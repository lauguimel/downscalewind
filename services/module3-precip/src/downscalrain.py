"""CNN patch-to-point model for daily precipitation downscaling.

The tabular XGBoost model remains the fast baseline. This module implements the
spatial branch: a small CNN receives a multi-source patch around a station/date
and predicts station-quality daily precipitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DownscalRainLossConfig:
    wet_threshold_mm: float = 0.2
    occurrence_weight: float = 1.0
    amount_weight: float = 1.0
    dry_amount_weight: float = 0.05
    heavy_rain_threshold_mm: float = 10.0
    heavy_rain_weight: float = 1.5


class ResidualBlock(nn.Module):
    """Small residual block used by the precipitation encoder."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels, dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class DownscalRainCNN(nn.Module):
    """Patch-to-point CNN with occurrence and amount heads.

    Parameters
    ----------
    in_channels:
        Number of gridded input channels in the patch.
    meta_dim:
        Number of station/date metadata features concatenated after the encoder.
    width:
        Base CNN width.
    depths:
        Number of residual blocks per downsampling stage.
    dropout:
        Dropout used in residual blocks and readout MLP.
    """

    def __init__(
        self,
        in_channels: int,
        meta_dim: int = 0,
        width: int = 32,
        depths: tuple[int, ...] = (2, 2, 2),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if meta_dim < 0:
            raise ValueError("meta_dim must be non-negative")

        self.in_channels = int(in_channels)
        self.meta_dim = int(meta_dim)
        self.width = int(width)
        self.depths = tuple(int(d) for d in depths)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            ResidualBlock(width, dropout),
        )

        channels = width
        stages = []
        for i, n_blocks in enumerate(self.depths):
            out_channels = width * (2 ** (i + 1))
            stages.append(EncoderStage(channels, out_channels, n_blocks, dropout))
            channels = out_channels
        self.encoder = nn.Sequential(*stages)

        encoded_dim = channels * 2  # global mean + station-centered feature
        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, max(16, width)),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, width), width),
                nn.SiLU(),
            )
            readout_dim = encoded_dim + width
        else:
            self.meta_mlp = None
            readout_dim = encoded_dim

        hidden = max(64, width * 4)
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.occurrence_head = nn.Linear(hidden // 2, 1)
        self.amount_head = nn.Linear(hidden // 2, 1)

    @staticmethod
    def _center_feature(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2] // 2
        w = x.shape[-1] // 2
        return x[..., h, w]

    def forward(self, patch: torch.Tensor, meta: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if patch.ndim != 4:
            raise ValueError(f"patch must have shape (B, C, H, W), got {tuple(patch.shape)}")
        if patch.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {patch.shape[1]}")

        x = self.encoder(self.stem(patch))
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        center = self._center_feature(x)
        features = [pooled, center]

        if self.meta_dim > 0:
            if meta is None:
                raise ValueError("meta tensor is required when meta_dim > 0")
            if meta.ndim != 2 or meta.shape[1] != self.meta_dim:
                raise ValueError(f"meta must have shape (B, {self.meta_dim}), got {tuple(meta.shape)}")
            assert self.meta_mlp is not None
            features.append(self.meta_mlp(meta))

        h = self.readout(torch.cat(features, dim=1))
        return {
            "wet_logit": self.occurrence_head(h).squeeze(-1),
            "log_amount": self.amount_head(h).squeeze(-1),
        }


def predict_rain_mm(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert occurrence and amount heads to expected daily rain in mm."""
    wet_prob = torch.sigmoid(outputs["wet_logit"])
    log_amount = F.softplus(outputs["log_amount"])
    amount = torch.expm1(log_amount)
    return wet_prob * amount


def downscalrain_loss(
    outputs: dict[str, torch.Tensor],
    rain_mm: torch.Tensor,
    config: DownscalRainLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Occurrence + amount loss for station daily precipitation."""
    cfg = config or DownscalRainLossConfig()
    rain_mm = rain_mm.float().clamp_min(0.0)
    wet = rain_mm > cfg.wet_threshold_mm
    wet_float = wet.float()

    occurrence_loss = F.binary_cross_entropy_with_logits(outputs["wet_logit"], wet_float)
    pred_log_amount = F.softplus(outputs["log_amount"])
    target_log_amount = torch.log1p(rain_mm)

    if wet.any():
        weights = torch.ones_like(rain_mm[wet])
        heavy = rain_mm[wet] >= cfg.heavy_rain_threshold_mm
        weights = torch.where(heavy, weights * cfg.heavy_rain_weight, weights)
        amount_raw = F.smooth_l1_loss(
            pred_log_amount[wet],
            target_log_amount[wet],
            reduction="none",
        )
        amount_loss = (amount_raw * weights).mean()
    else:
        amount_loss = pred_log_amount.mean() * 0.0

    if (~wet).any():
        dry_amount_loss = pred_log_amount[~wet].mean()
    else:
        dry_amount_loss = pred_log_amount.mean() * 0.0

    total = (
        cfg.occurrence_weight * occurrence_loss
        + cfg.amount_weight * amount_loss
        + cfg.dry_amount_weight * dry_amount_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "occurrence_loss": float(occurrence_loss.detach().cpu()),
        "amount_loss": float(amount_loss.detach().cpu()),
        "dry_amount_loss": float(dry_amount_loss.detach().cpu()),
    }
    return total, parts


def precipitation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wet_threshold_mm: float = 1.0,
    heavy_threshold_mm: float = 10.0,
) -> dict[str, float]:
    """Compute publication-relevant daily precipitation metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty array")

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else 0.0

    wet_true = y_true > wet_threshold_mm
    wet_pred = y_pred > wet_threshold_mm
    tp = float(np.sum(wet_true & wet_pred))
    fp = float(np.sum(~wet_true & wet_pred))
    fn = float(np.sum(wet_true & ~wet_pred))
    tn = float(np.sum(~wet_true & ~wet_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    heavy_true = y_true > heavy_threshold_mm
    heavy_pred = y_pred > heavy_threshold_mm
    heavy_tp = float(np.sum(heavy_true & heavy_pred))
    heavy_fn = float(np.sum(heavy_true & ~heavy_pred))
    heavy_recall = heavy_tp / (heavy_tp + heavy_fn) if (heavy_tp + heavy_fn) > 0 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": corr,
        "wet_precision": float(precision),
        "wet_recall": float(recall),
        "dry_false_alarm": float(false_alarm),
        "heavy_recall": float(heavy_recall),
    }


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract model kwargs from a saved training checkpoint."""
    keys = ("in_channels", "meta_dim", "width", "depths", "dropout")
    return {k: checkpoint[k] for k in keys if k in checkpoint}
