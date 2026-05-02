"""
model_vit_v2.py — TerrainViT adapted to the campaign-v2 native 180×180×40 grid.

Key changes vs. the 9k variant in `model_vit.py`:
  - patch_size=12 → pg=15 (180/12)
  - UpsampleDecoder2D upsamples 15→120, then F.interpolate to 180
  - FiLMVerticalHead with nz=40
  - ERA5 input dim ≈ 400 (3×3 grid × 4 vars × 10 levels + surface)

Reuses the building blocks of model_vit.py (TransformerBlock, CrossAttentionBlock,
PatchEmbed2D, FiLMVerticalHead, ERA5TokenEncoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_vit import (
    PatchEmbed2D,
    TransformerBlock,
    CrossAttentionBlock,
    FiLMVerticalHead,
    ERA5TokenEncoder,
    _init_weights,
)


class UpsampleDecoder2D_V2(nn.Module):
    """(B, embed_dim, pg, pg) → (B, feat_dim, out_size, out_size).

    8× transposed conv stack, then F.interpolate to the target size.
    """
    def __init__(self, embed_dim: int = 384, feat_dim: int = 64,
                 patch_grid: int = 15, out_size: int = 180):
        super().__init__()
        self.out_size = out_size
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, feat_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)            # (B, feat_dim, pg*8, pg*8) = (..., 120, 120)
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return x                       # (..., 180, 180)


class TerrainViT_V2_S3(nn.Module):
    """Cross-attention ViT for 180×180×40 grid.

    terrain (B, 2, 180, 180) + era5 (B, era5_dim) → (B, n_out, 180, 180, 40)
    """
    def __init__(self, img_size=180, patch_size=12, nz=40,
                 embed_dim=384, depth=12, n_heads=8,
                 mlp_ratio=4.0, drop=0.1, feat_dim=64,
                 era5_input_dim=400, n_output_vars=5,
                 n_cross_layers=4, n_era5_tokens=16):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        pg = img_size // patch_size
        self.patch_grid = pg
        self.img_size = img_size
        n_self = depth - n_cross_layers

        self.patch_embed = PatchEmbed2D(2, embed_dim, img_size, patch_size)
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
            FiLMVerticalHead(feat_dim, nz, era5_input_dim)
            for _ in range(n_output_vars)])
        self.apply(_init_weights)

    def forward(self, terrain: torch.Tensor, era5: torch.Tensor) -> torch.Tensor:
        t_tokens = self.patch_embed(terrain)
        e_tokens = self.era5_tokens(era5)
        for blk in self.cross_blocks:
            t_tokens = blk(t_tokens, e_tokens)
        for blk in self.self_blocks:
            t_tokens = blk(t_tokens)
        t_tokens = self.norm(t_tokens)

        pg = self.patch_grid
        feat2d = t_tokens.transpose(1, 2).view(t_tokens.shape[0], -1, pg, pg)
        feat2d = self.upsample(feat2d)                # (B, feat_dim, 180, 180)
        return torch.cat([h(feat2d, era5) for h in self.heads], dim=1)


_PRESETS_V2 = {
    "small": dict(embed_dim=256, depth=8, n_heads=8, drop=0.1, feat_dim=48),
    "base":  dict(embed_dim=384, depth=12, n_heads=8, drop=0.1, feat_dim=64),
    "large": dict(embed_dim=512, depth=16, n_heads=16, drop=0.1, feat_dim=96,
                  n_cross_layers=6, n_era5_tokens=24),
}


def build_vit_v2(preset: str = "base", **overrides) -> nn.Module:
    if preset not in _PRESETS_V2:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(_PRESETS_V2)}")
    cfg = {**_PRESETS_V2[preset], **overrides}
    return TerrainViT_V2_S3(**cfg)
