"""
model_unet3d.py — U-Net 3D for wind/T/q downscaling on regular grid.

Architecture:
    Input:  (B, C_in, Ny, Nx, Nz) — terrain + ERA5 profiles (or terrain only + FiLM)
    Output: (B, 5, Ny, Nx, Nz)    — Δu, Δv, Δw, ΔT, Δq (residual)

Two variants:
    V1 "volume":       C_in=7 (terrain, z0, era5_u, era5_v, era5_T, era5_q, era5_k)
    V2 "terrain_only": C_in=2 (terrain, z0) + ProfileEncoder → FiLM conditioning

3-level encoder-decoder with skip connections.
Typical input: (B, 7, 128, 128, 32) → output: (B, 5, 128, 128, 32)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer3D(nn.Module):
    """Feature-wise Linear Modulation for 3D volumes.

    Given a conditioning vector, produces per-channel scale and shift
    applied to a 3D feature map.
    """

    def __init__(self, n_features: int, n_cond: int):
        super().__init__()
        self.gamma = nn.Linear(n_cond, n_features)
        self.beta = nn.Linear(n_cond, n_features)
        # Initialize to identity transform
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, C, D, H, W)
        cond: (B, n_cond)
        """
        gamma = self.gamma(cond)[:, :, None, None, None]  # (B, C, 1, 1, 1)
        beta = self.beta(cond)[:, :, None, None, None]
        return x * (1 + gamma) + beta


class ProfileEncoder(nn.Module):
    """Encode 1D ERA5 profiles into a conditioning vector.

    Input:  (B, n_vars, n_levels) — e.g., (B, 5, 32)
    Output: (B, cond_dim)
    """

    def __init__(self, n_vars: int = 5, n_levels: int = 32, cond_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_vars, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, cond_dim),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        """profiles: (B, n_vars, n_levels) → (B, cond_dim)"""
        return self.net(profiles)


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """3-level U-Net for 3D wind/T/q field prediction.

    Supports optional FiLM conditioning at the bottleneck (for V2 variant).
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 5,
        base_features: int = 32,
        cond_dim: int = 0,
    ):
        """
        Parameters
        ----------
        in_channels : 7 for V1 (volume), 2 for V2 (terrain_only)
        out_channels : 5 (Δu, Δv, Δw, ΔT, Δq)
        base_features : base channel width
        cond_dim : if > 0, enable FiLM conditioning at bottleneck
        """
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        # Optional FiLM conditioning
        self.film = FiLMLayer3D(f * 8, cond_dim) if cond_dim > 0 else None

        # Decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)

        self.out_conv = nn.Conv3d(f, out_channels, 1)

        # Pooling
        self.pool = nn.MaxPool3d(2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, D, H, W) — input volume
        cond : (B, cond_dim) — optional FiLM conditioning vector

        Returns
        -------
        (B, C_out, D, H, W) — predicted residual fields
        """
        # Pad to ensure dimensions are divisible by 8
        orig_shape = x.shape[2:]
        pad = []
        for s in reversed(orig_shape):
            target = ((s + 7) // 8) * 8
            pad.extend([0, target - s])
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # FiLM conditioning
        if self.film is not None and cond is not None:
            b = self.film(b, cond)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)

        # Crop back to original size
        out = out[:, :, : orig_shape[0], : orig_shape[1], : orig_shape[2]]
        return out

    @staticmethod
    def _match_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Match spatial dimensions and concatenate skip connection."""
        d = min(up.shape[2], skip.shape[2])
        h = min(up.shape[3], skip.shape[3])
        w = min(up.shape[4], skip.shape[4])
        return torch.cat([up[:, :, :d, :h, :w], skip[:, :, :d, :h, :w]], dim=1)
