"""
Shared losses for campaign-v2 surrogate training.

The optional AGL weight is a per-voxel tensor with shape
(B, 1, Ny, Nx, Nz). It reweights only pointwise data terms; spectral and
divergence regularisers stay global.
"""
from __future__ import annotations

import torch


def weighted_mean(values: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of pointwise values, optionally weighted over spatial voxels."""
    if weight is None:
        return values.mean()
    if weight.ndim == values.ndim - 1:
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and values.shape[1] != 1:
        denom = weight.sum() * values.shape[1]
    else:
        denom = weight.expand_as(values).sum()
    return (values * weight).sum() / denom.clamp_min(1e-12)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return weighted_mean((pred - target).pow(2), weight)


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    return weighted_mean(torch.sqrt((pred - target) ** 2 + eps ** 2), weight)


def amplitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral amplitude loss on z-mean 2D fields."""
    pred_2d = pred.mean(dim=-1)
    target_2d = target.mean(dim=-1)
    pred_amp = torch.fft.rfft2(pred_2d, dim=(-2, -1)).abs()
    target_amp = torch.fft.rfft2(target_2d, dim=(-2, -1)).abs()
    return (pred_amp - target_amp).abs().mean()


def divergence_loss(pred: torch.Tensor, dx: float = 33.333) -> torch.Tensor:
    """Soft div-free penalty assuming uniform vertical spacing (cheap approx)."""
    u, v, w = pred[:, 0], pred[:, 1], pred[:, 2]
    du_dx = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dx)
    dv_dy = (v[:, 2:, :, :] - v[:, :-2, :, :]) / (2.0 * dx)
    dw_dz = (w[:, :, :, 2:] - w[:, :, :, :-2]) / 2.0
    ny = min(du_dx.shape[1], dv_dy.shape[1])
    nx = min(du_dx.shape[2], dv_dy.shape[2])
    nz = min(du_dx.shape[3], dw_dz.shape[3])
    div = (
        du_dx[:, 1:ny + 1, :nx, 1:nz + 1]
        + dv_dy[:, :ny, 1:nx + 1, 1:nz + 1]
        + dw_dz[:, 1:ny + 1, 1:nx + 1, :nz]
    )
    return div.pow(2).mean()


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    *,
    weight: torch.Tensor | None = None,
    w_amp: float = 0.1,
    w_div: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if kind == "mse":
        l = mse_loss(pred, target, weight)
        return l, {"mse": l.item()}
    if kind == "charbonnier":
        l = charbonnier_loss(pred, target, weight)
        return l, {"char": l.item()}
    if kind == "s4":
        l_c = charbonnier_loss(pred, target, weight)
        l_a = amplitude_loss(pred, target)
        l_d = divergence_loss(pred)
        l = l_c + w_amp * l_a + w_div * l_d
        return l, {"char": l_c.item(), "amp": l_a.item(), "div": l_d.item()}
    raise ValueError(f"unknown loss-type {kind}")
