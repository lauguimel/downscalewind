"""
ann_correction.py — DEVINE-style ANN correction in front of the frozen
surrogate v2 ViT (Phase H', M_H'0).

Reference: Le Toumelin et al. 2024 NPG, Sect. 3.3.1 + Table 2.

Design choices (Table 2 transcription, adapted):
    - 2 hidden layers, units = [50, 10] (matches ANN_speed)
    - Activation = SELU (Selu in the paper, "Excluding the output neuron")
    - Output activation = Linear (raw delta, can be positive or negative)
    - Dropout 0.25 between layers (paper value for ANN_speed)
    - Initializer = Glorot Uniform (Xavier) on all hidden layers
    - LAST LAYER zero-initialised so the ANN starts as identity (delta=0) →
      the surrogate v2 sees the raw ERA5 vector at step 0, matching the
      DEVINE training trajectory ("starts as raw NWP, learns correction").
    - Output is ADDED to era5_flat (residual skip connection; eq. to the
      "+" symbols in the paper's Fig. 2 between ANN_speed and DEVINE).

This module does NOT bound the corrected vector — the surrogate v2 normalises
its own inputs internally and the loss at the central pixel is the supervision
signal that prevents pathological corrections.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ANNCorrection(nn.Module):
    """MLP that learns Δ ERA5 from (era5_flat, topo_features).

    Inputs:
        era5_flat: (B, era5_dim)   — surrogate v2 flat ERA5 vector
                                      (408 = 4 vars × 3×3 × 10 plevels
                                       + 4 surf × 3×3 + 10 plev + lat + z0)
        topo_features: (B, F)      — local topo + season + diurnal features
                                      (default F=8, see dataset_v2_obs_centered.py)
        terrain: (B, C, 180, 180)  — optional full terrain tensor, used only
                                      when use_terrain_encoder=True

    Output:
        era5_corrected: (B, era5_dim) = era5_flat + delta
    """

    def __init__(
        self,
        era5_dim: int = 408,
        topo_dim: int = 8,
        hidden_units: tuple[int, int] = (50, 10),
        dropout: float = 0.25,
        zero_init_output: bool = True,
        use_terrain_encoder: bool = False,
        terrain_latent_dim: int = 48,
        terrain_in_channels: int = 4,
    ) -> None:
        super().__init__()
        self.era5_dim = era5_dim
        self.topo_dim = topo_dim
        self.use_terrain_encoder = use_terrain_encoder
        self.terrain_latent_dim = terrain_latent_dim
        self.terrain_in_channels = terrain_in_channels

        in_dim = era5_dim + topo_dim
        if use_terrain_encoder:
            self.terrain_encoder = nn.Sequential(
                nn.Conv2d(terrain_in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 16),
                nn.SELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 32),
                nn.SELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, terrain_latent_dim),
                nn.SELU(),
            )
            in_dim += terrain_latent_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, era5_dim))  # linear output
        self.mlp = nn.Sequential(*layers)
        self._init_weights(zero_init_output=zero_init_output)

    def _init_weights(self, zero_init_output: bool) -> None:
        if self.use_terrain_encoder:
            for m in self.terrain_encoder.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_output:
            last = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        era5_flat: torch.Tensor,
        topo_features: torch.Tensor,
        terrain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if era5_flat.dim() != 2 or topo_features.dim() != 2:
            raise ValueError(
                f"Expected 2D inputs, got era5_flat {era5_flat.shape}, "
                f"topo_features {topo_features.shape}"
            )
        if era5_flat.shape[-1] != self.era5_dim:
            raise ValueError(
                f"era5_flat last dim {era5_flat.shape[-1]} != {self.era5_dim}"
            )
        if topo_features.shape[-1] != self.topo_dim:
            raise ValueError(
                f"topo_features last dim {topo_features.shape[-1]} != {self.topo_dim}"
            )
        parts = [era5_flat, topo_features]
        if self.use_terrain_encoder:
            if terrain is None:
                raise ValueError("terrain is required when use_terrain_encoder=True")
            if terrain.dim() != 4:
                raise ValueError(
                    f"Expected terrain shape (B, C, 180, 180), got {terrain.shape}"
                )
            if terrain.shape[0] != era5_flat.shape[0]:
                raise ValueError(
                    f"terrain batch {terrain.shape[0]} != era5 batch {era5_flat.shape[0]}"
                )
            if terrain.shape[1] != self.terrain_in_channels:
                raise ValueError(
                    f"terrain channels {terrain.shape[1]} != {self.terrain_in_channels}"
                )
            if terrain.shape[-2:] != (180, 180):
                raise ValueError(
                    f"terrain spatial shape {terrain.shape[-2:]} != (180, 180)"
                )
            parts.append(self.terrain_encoder(terrain))
        x = torch.cat(parts, dim=-1)
        delta = self.mlp(x)
        return era5_flat + delta  # residual skip connection


def devine_speed_loss(
    speed_pred: torch.Tensor,
    speed_obs: torch.Tensor,
    tau_under: float = 0.6,
    tau_over: float = 0.4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Custom asymmetric loss from Le Toumelin 2024 Eq. (3).

        L_speed = speed_obs * tau * mse(speed_obs, speed_pred)
        tau = 0.6  if speed_obs <= speed_pred  (model overestimates → fast cap)
        tau = 0.4  if speed_obs >  speed_pred  (model underestimates)

    NOTE: the paper says "0.6 for underestimations and 0.4 for overestimations",
    with the explicit case split `speed_obs <= speed_model -> tau=0.6` and
    `speed_obs > speed_model -> tau=0.4`. We follow the literal case split (this
    penalises overestimations more, contrary to the textual claim — see paper
    discussion). The DEVINE 2024 logic is: surrogate underestimates strong
    winds, so a stronger penalty on the "obs > pred" branch would force
    correction upward; the paper inverts via wording but the formula split is
    explicit and is what we replicate here.

    NAMING CAVEAT: here `over_mask = (speed_obs > speed_pred)` is the branch
    where the MODEL UNDER-predicts (obs above pred); it gets `tau_over`. The
    branch `speed_obs <= speed_pred` is where the MODEL OVER-predicts; it gets
    `tau_under`. With the M_I3 default (0.6/0.4) the over-prediction branch is
    penalised HARDER — good against the surrogate's strong-wind compression,
    but it over-generalises to calm and makes the ANN add wind everywhere
    (M_I4: Perdigão <3 m/s bias corr +1.6..+2.65 vs raw ~0). See
    `devine_speed_loss_regime` for the regime-aware fix.
    """
    if speed_pred.shape != speed_obs.shape:
        raise ValueError(
            f"shape mismatch: pred {speed_pred.shape} vs obs {speed_obs.shape}"
        )
    # paper Eq. (3) literal case split:
    #   tau = tau_under (0.6) if speed_obs <= speed_pred
    #   tau = tau_over  (0.4) if speed_obs >  speed_pred
    over_mask = (speed_obs > speed_pred).to(speed_pred.dtype)
    tau = tau_under * (1.0 - over_mask) + tau_over * over_mask
    mse = (speed_obs - speed_pred) ** 2
    weight = torch.clamp(speed_obs, min=eps)
    return (weight * tau * mse).mean()


def devine_speed_loss_regime(
    speed_pred: torch.Tensor,
    speed_obs: torch.Tensor,
    tau_under: float = 0.6,
    tau_over: float = 0.4,
    *,
    calm_threshold: float = 3.0,
    calm_width: float = 1.5,
    calm_over_penalty: float = 2.0,
    weight_floor: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Regime-aware asymmetric speed loss (M_I5b fix for low-wind over-add).

    Diagnosis (M_I4): the flat τ-asymmetry meant to fight the surrogate's
    strong-wind compression (slope 0.59) over-generalises to calm, where the
    raw surrogate is already good — the ANN then ADDS wind everywhere.

    Fix: make the penalty on the OVER-prediction branch (model predicts more
    than obs, i.e. `speed_pred > speed_obs`) magnitude-aware. In CALM
    conditions, over-prediction is the failure mode we want to suppress, so we
    *raise* the τ on that branch toward `calm_over_penalty`. In strong wind we
    leave the original asymmetry untouched so the held-out / steep gain is
    preserved.

    Branches (note: `obs > pred` ⇒ model UNDER-predicts):
        under-pred (obs > pred): τ = tau_over     (unchanged everywhere)
        over-pred  (obs <= pred): τ = tau_under in strong wind,
                                   blended up to (tau_under * calm_over_penalty)
                                   as obs → 0, gated smoothly by a sigmoid on
                                   (calm_threshold - speed_obs)/calm_width.

    Two other low-wind safeguards vs the base loss:
      - `weight_floor`: the base loss multiplies MSE by `speed_obs`, which
        ~zeroes the gradient in calm (so calm errors are invisible to the
        optimiser). We floor the weight at `weight_floor` (default 1.0) so calm
        over-prediction actually contributes to the loss.

    Reduces to the base asymmetric loss when calm_over_penalty=1.0 and
    weight_floor is set to 0 (with weight = speed_obs).
    """
    if speed_pred.shape != speed_obs.shape:
        raise ValueError(
            f"shape mismatch: pred {speed_pred.shape} vs obs {speed_obs.shape}"
        )
    under_mask = (speed_obs > speed_pred).to(speed_pred.dtype)   # model under-predicts
    over_mask = 1.0 - under_mask                                 # model over-predicts

    # Smooth calm gate in [0, 1]: ~1 when obs << threshold, ~0 when obs >> threshold.
    calm_gate = torch.sigmoid((calm_threshold - speed_obs) / max(calm_width, eps))
    # τ on the over-prediction branch: tau_under in strong wind, up to
    # tau_under*calm_over_penalty in calm.
    tau_over_branch = tau_under * (1.0 + (calm_over_penalty - 1.0) * calm_gate)

    tau = tau_over * under_mask + tau_over_branch * over_mask
    mse = (speed_obs - speed_pred) ** 2
    weight = torch.clamp(speed_obs, min=weight_floor)
    return (weight * tau * mse).mean()
