"""
model_vit.py — Vision Transformer for terrain-aware 3D wind downscaling.

Inspired by FuXi-CFD (Lin et al., Nature Communications 2026):
  - Shared ViT encoder (patch embedding + transformer blocks)
  - Task-specific decoder heads (one per output variable)
  - Inputs: HR terrain (128x128) + HR z0 (128x128) + LR ERA5 profiles (nz,)
  - Outputs: u, v, w, T, q residual fields (128x128x32)

Architecture:
  1. Terrain/z0 maps → 2D patch embedding (patch_size=8 → 16x16 patches)
  2. ERA5 profiles → broadcast to 3D, project, add to patch embeddings
  3. Transformer encoder (N blocks, multi-head self-attention)
  4. Reshape to spatial grid → per-variable 3D decoder heads
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    """Embed 2D terrain maps into patch tokens.

    Input:  (B, C_terrain, H, W) — e.g. (B, 2, 128, 128)
    Output: (B, n_patches, embed_dim) + positional encoding
    """

    def __init__(self, in_channels: int = 2, embed_dim: int = 384,
                 img_size: int = 128, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x + self.pos_embed


class ERA5ProfileEncoder(nn.Module):
    """Encode 1D ERA5 profiles into a conditioning vector added to each patch.

    Input:  (B, n_vars, nz) — e.g. (B, 5, 32)
    Output: (B, embed_dim) — broadcast-added to all patch tokens
    """

    def __init__(self, n_vars: int = 5, nz: int = 32, embed_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vars * nz, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        # profiles: (B, n_vars, nz)
        x = profiles.flatten(1)  # (B, n_vars * nz)
        return self.net(x)  # (B, embed_dim)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with GELU MLP."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderHead3D(nn.Module):
    """Decode patch tokens back to a 3D volume for one variable.

    Reshapes (B, n_patches, embed_dim) → (B, 1, ny, nx, nz).
    Uses transposed convolutions to upsample from patch grid to full resolution.
    """

    def __init__(self, embed_dim: int = 384, patch_grid: int = 16,
                 nz: int = 32, patch_size: int = 8):
        super().__init__()
        self.patch_grid = patch_grid
        self.nz = nz

        # Project each patch token to a 3D volume patch
        self.token_proj = nn.Linear(embed_dim, patch_size * patch_size * nz)

        # Refine with 3D convolutions
        self.refine = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        pg = self.patch_grid
        ps = int(math.isqrt(tokens.shape[-1] // self.nz))  # recover patch_size

        # Project tokens to spatial patches
        x = self.token_proj(tokens)  # (B, n_patches, ps*ps*nz)
        x = x.view(B, pg, pg, ps, ps, self.nz)  # unfold patches

        # Rearrange to full spatial grid: (B, 1, ny, nx, nz)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, pg, ps, pg, ps, nz)
        x = x.view(B, 1, pg * ps, pg * ps, self.nz)

        # Refine
        x = self.refine(x)
        return x  # (B, 1, ny, nx, nz)


class TerrainViT(nn.Module):
    """Vision Transformer for terrain-aware 3D wind field downscaling.

    Input channels:
      - terrain: (B, 128, 128) — elevation [m]
      - z0:      (B, 128, 128) — roughness [m]
      - ERA5 profiles: (B, 5, 32) — u, v, T, q, k at nz levels

    Output: (B, 5, 128, 128, 32) — residual fields (u, v, w, T, q)
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        n_terrain_channels: int = 2,
        n_era5_vars: int = 5,
        nz: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        n_output_vars: int = 5,
    ):
        super().__init__()
        self.nz = nz
        self.n_output_vars = n_output_vars
        self.patch_grid = img_size // patch_size

        # Patch embedding for 2D terrain maps
        self.patch_embed = PatchEmbed2D(
            in_channels=n_terrain_channels,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
        )

        # ERA5 profile conditioning
        self.era5_encoder = ERA5ProfileEncoder(
            n_vars=n_era5_vars, nz=nz, embed_dim=embed_dim)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Per-variable decoder heads
        self.heads = nn.ModuleList([
            DecoderHead3D(embed_dim, self.patch_grid, nz, patch_size)
            for _ in range(n_output_vars)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, terrain: torch.Tensor, era5: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        terrain : (B, 2, 128, 128) — [terrain_norm, z0_norm]
        era5    : (B, 5, 32) — ERA5 profiles [u, v, T, q, k] normalized

        Returns
        -------
        (B, 5, 128, 128, 32) — predicted residual fields
        """
        # Patch embedding
        tokens = self.patch_embed(terrain)  # (B, n_patches, embed_dim)

        # Add ERA5 conditioning to all tokens
        era5_cond = self.era5_encoder(era5)  # (B, embed_dim)
        tokens = tokens + era5_cond.unsqueeze(1)

        # Transformer encoder
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Decode per variable
        outputs = [head(tokens) for head in self.heads]
        return torch.cat(outputs, dim=1)  # (B, 5, ny, nx, nz)


def build_vit(preset: str = "base", **overrides) -> TerrainViT:
    """Build a TerrainViT with preset configurations.

    Presets:
      - "small": embed_dim=256, depth=8,  heads=8   (~25M params)
      - "base":  embed_dim=384, depth=12, heads=8   (~55M params)
      - "large": embed_dim=512, depth=16, heads=16  (~110M params)
    """
    configs = {
        "small": dict(embed_dim=256, depth=8, n_heads=8, drop=0.1),
        "base": dict(embed_dim=384, depth=12, n_heads=8, drop=0.1),
        "large": dict(embed_dim=512, depth=16, n_heads=16, drop=0.1),
    }
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs)}")
    cfg = {**configs[preset], **overrides}
    return TerrainViT(**cfg)
