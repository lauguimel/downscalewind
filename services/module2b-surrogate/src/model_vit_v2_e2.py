from __future__ import annotations

import torch
import torch.nn as nn

from .model_vit import (
    PatchEmbed2D,
    TransformerBlock,
    CrossAttentionBlock,
    ERA5TokenEncoder,
    _init_weights,
)
from .model_vit_v2 import GeoFiLMVerticalHead, UpsampleDecoder2D_V2


class TerrainViT_V2_E2(nn.Module):
    """Cross-attention ViT with strategy A: sparse additive OBS token embedding.

    I use strategy A because the observation is a single sparse station-like
    value on the 180x180 surface grid, so routing it only to the patch that
    contains `(i, j)` preserves the existing terrain/ERA5 contract and avoids
    changing `era5_flat_dim` or the patch input channel count. The OBS branch
    is a small MLP over `(value, mask)` whose output is multiplied by the mask
    before being added to the selected patch token. Therefore mask=0 is exactly
    a no-op, so a partially loaded surface checkpoint reproduces the base model
    when OBS is dropped. The final OBS projection is initialized at small scale
    to make mask=1 informative but non-disruptive at the start of fine-tuning.
    """

    def __init__(self, img_size=180, patch_size=12, nz=40,
                 embed_dim=384, depth=12, n_heads=8,
                 mlp_ratio=4.0, drop=0.1, feat_dim=64,
                 era5_input_dim=400, n_output_vars=5,
                 n_cross_layers=4, n_era5_tokens=16,
                 terrain_in_channels=2, geo_channels=0,
                 obs_value_dim=1, obs_mask_dim=1, obs_init_std=1e-3):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        pg = img_size // patch_size
        self.patch_grid = pg
        self.img_size = img_size
        self.patch_size = patch_size
        self.obs_value_dim = obs_value_dim
        self.obs_mask_dim = obs_mask_dim
        n_self = depth - n_cross_layers

        self.patch_embed = PatchEmbed2D(terrain_in_channels, embed_dim,
                                        img_size, patch_size)
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_value_dim + obs_mask_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.era5_tokens = ERA5TokenEncoder(era5_input_dim, embed_dim, n_era5_tokens)

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, n_heads, drop)
            for _ in range(n_cross_layers)])
        self.self_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(n_self)])
        self.norm = nn.LayerNorm(embed_dim)

        self.upsample = UpsampleDecoder2D_V2(embed_dim, feat_dim, pg, out_size=img_size)
        self.heads = nn.ModuleList([
            GeoFiLMVerticalHead(feat_dim, nz, era5_input_dim,
                                geo_channels=geo_channels)
            for _ in range(n_output_vars)])
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.obs_mlp[-1].weight, std=obs_init_std)
        nn.init.zeros_(self.obs_mlp[-1].bias)

    def _add_obs_token(self, tokens: torch.Tensor, obs_value: torch.Tensor | None,
                       obs_mask: torch.Tensor | None,
                       obs_ij: torch.Tensor | None) -> torch.Tensor:
        if obs_value is None or obs_mask is None or obs_ij is None:
            return tokens
        bsz, _, emb = tokens.shape
        obs_value = obs_value.to(device=tokens.device, dtype=tokens.dtype).view(bsz, -1)
        obs_mask = obs_mask.to(device=tokens.device, dtype=tokens.dtype).view(bsz, -1)
        gate = obs_mask[:, :1]
        obs_in = torch.cat([obs_value[:, :self.obs_value_dim],
                            obs_mask[:, :self.obs_mask_dim]], dim=1)
        delta = self.obs_mlp(obs_in) * gate

        ij = obs_ij.to(device=tokens.device).long().view(bsz, 2)
        patch_i = torch.clamp(ij[:, 0], 0, self.img_size - 1) // self.patch_size
        patch_j = torch.clamp(ij[:, 1], 0, self.img_size - 1) // self.patch_size
        patch_idx = patch_i * self.patch_grid + patch_j
        add = torch.zeros_like(tokens)
        add.scatter_add_(1, patch_idx.view(bsz, 1, 1).expand(bsz, 1, emb),
                         delta.view(bsz, 1, emb))
        return tokens + add

    def forward(self, terrain: torch.Tensor, era5: torch.Tensor,
                geo: torch.Tensor | None = None,
                obs_value: torch.Tensor | None = None,
                obs_mask: torch.Tensor | None = None,
                obs_ij: torch.Tensor | None = None) -> torch.Tensor:
        t_tokens = self.patch_embed(terrain)
        t_tokens = self._add_obs_token(t_tokens, obs_value, obs_mask, obs_ij)
        e_tokens = self.era5_tokens(era5)
        for blk in self.cross_blocks:
            t_tokens = blk(t_tokens, e_tokens)
        for blk in self.self_blocks:
            t_tokens = blk(t_tokens)
        t_tokens = self.norm(t_tokens)

        pg = self.patch_grid
        feat2d = t_tokens.transpose(1, 2).view(t_tokens.shape[0], -1, pg, pg)
        feat2d = self.upsample(feat2d)
        return torch.cat([h(feat2d, era5, geo) for h in self.heads], dim=1)


_PRESETS_V2_E2 = {
    "small": dict(embed_dim=256, depth=8, n_heads=8, drop=0.1, feat_dim=48),
    "base": dict(embed_dim=384, depth=12, n_heads=8, drop=0.1, feat_dim=64),
    "large": dict(embed_dim=512, depth=16, n_heads=16, drop=0.1, feat_dim=96,
                  n_cross_layers=6, n_era5_tokens=24),
}


def build_vit_v2_e2(preset: str = "base", **overrides) -> nn.Module:
    if preset not in _PRESETS_V2_E2:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(_PRESETS_V2_E2)}")
    cfg = {**_PRESETS_V2_E2[preset], **overrides}
    return TerrainViT_V2_E2(**cfg)
