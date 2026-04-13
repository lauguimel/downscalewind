"""
model_vit.py — TerrainViT variants for 3D wind downscaling.

Three architectures, all using S4 training recipe:

  S1 "film":   v1 encoder (terrain patches + ERA5 additive) + v2 decoder + FiLM vertical
  S2 "factor": Factored 2D transformer + 1D vertical MLP conditioned by ERA5
  S3 "cross":  Cross-attention between terrain tokens and ERA5 profile tokens

Shared components: TransformerBlock, UpsampleDecoder2D, PatchEmbed2D.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared components ───────────────────────────────────────────────────────

class PatchEmbed2D(nn.Module):
    """2D terrain patch embedding with positional encoding."""

    def __init__(self, in_channels: int = 2, embed_dim: int = 384,
                 img_size: int = 128, patch_size: int = 8):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to key/value from another sequence."""

    def __init__(self, dim: int, n_heads: int = 8, drop: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * 4, dim), nn.Dropout(drop),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        h, _ = self.attn(q_n, kv_n, kv_n)
        x = q + h
        return x + self.mlp(self.norm2(x))


class UpsampleDecoder2D(nn.Module):
    """Upsample tokens back to full 2D resolution via transposed convolutions.

    (B, embed_dim, pg, pg) → (B, feat_dim, H, W)
    """

    def __init__(self, embed_dim: int = 384, feat_dim: int = 64,
                 patch_grid: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, feat_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ERA5Encoder(nn.Module):
    """Encode ERA5 grid/profiles → (embed_dim,) for additive conditioning.

    Accepts either:
      - (B, 4, 3, 3, 32) — 3×3 spatial grid, 4 vars, 32 z-levels
      - (B, 4, 1, 1, 32) — 1D profiles (backward compat)
      - (B, 5, 32) — legacy 1D profiles
    All are flattened and projected.
    """

    def __init__(self, era5_input_dim: int = 1440, embed_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(era5_input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, era5: torch.Tensor) -> torch.Tensor:
        return self.net(era5.flatten(1))


def _init_weights(module):
    """Standard weight init for transformers."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ═══════════════════════════════════════════════════════════════════════════
# S1 — FiLM vertical: v1 encoder + v2 decoder + FiLM conditioning per z
# ═══════════════════════════════════════════════════════════════════════════

class FiLMVerticalHead(nn.Module):
    """3D decoder head with FiLM conditioning from ERA5 profiles.

    2D features → expand to 3D → FiLM modulation per z-level → conv3d.
    ERA5 profiles provide per-level gamma/beta.
    """

    def __init__(self, feat_dim: int = 64, nz: int = 32,
                 era5_input_dim: int = 1440, hidden: int = 32):
        super().__init__()
        self.nz = nz
        # Expand 2D → 3D: learn a vertical basis
        self.vert_basis = nn.Parameter(torch.randn(1, feat_dim, 1, 1, nz) * 0.02)
        # FiLM from ERA5: flatten → per-level (gamma, beta)
        self.film_net = nn.Sequential(
            nn.Linear(era5_input_dim, 128), nn.GELU(),
            nn.Linear(128, feat_dim * nz * 2),  # gamma + beta per (feat, z)
        )
        # 3D refinement
        self.refine = nn.Sequential(
            nn.Conv3d(feat_dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, feat2d: torch.Tensor,
                era5: torch.Tensor) -> torch.Tensor:
        """
        feat2d: (B, feat_dim, H, W)
        era5:   (B, n_vars, nz)
        """
        B, C, H, W = feat2d.shape
        # Expand to 3D: (B, C, H, W, 1) * (1, C, 1, 1, nz) → (B, C, H, W, nz)
        vol = feat2d.unsqueeze(-1) * self.vert_basis

        # FiLM modulation
        film = self.film_net(era5.flatten(1))  # (B, C*nz*2)
        film = film.view(B, C, self.nz, 2)
        gamma = film[:, :, :, 0]  # (B, C, nz)
        beta = film[:, :, :, 1]
        # Apply: (B, C, H, W, nz) * (B, C, 1, 1, nz) + ...
        vol = vol * (1 + gamma[:, :, None, None, :]) + beta[:, :, None, None, :]

        return self.refine(vol)  # (B, 1, H, W, nz)


class TerrainViT_S1(nn.Module):
    """S1: v1 encoder + v2 decoder + FiLM vertical."""

    def __init__(self, img_size=128, patch_size=8, nz=32,
                 embed_dim=384, depth=12, n_heads=8,
                 mlp_ratio=4.0, drop=0.1, feat_dim=64,
                 era5_input_dim=1440, n_output_vars=5):
        super().__init__()
        pg = img_size // patch_size
        self.patch_grid = pg

        self.patch_embed = PatchEmbed2D(2, embed_dim, img_size, patch_size)
        self.era5_enc = ERA5Encoder(era5_input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = UpsampleDecoder2D(embed_dim, feat_dim, pg)
        self.heads = nn.ModuleList([
            FiLMVerticalHead(feat_dim, nz, era5_input_dim)
            for _ in range(n_output_vars)])
        self.apply(_init_weights)

    def forward(self, terrain, era5):
        tokens = self.patch_embed(terrain)
        tokens = tokens + self.era5_enc(era5).unsqueeze(1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        pg = self.patch_grid
        feat2d = tokens.transpose(1, 2).view(
            tokens.shape[0], -1, pg, pg)
        feat2d = self.upsample(feat2d)
        return torch.cat([h(feat2d, era5) for h in self.heads], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# S2 — Factored 2D+1D: 2D transformer + vertical MLP conditioned by ERA5
# ═══════════════════════════════════════════════════════════════════════════

class VerticalMLPHead(nn.Module):
    """Per-variable head: 2D features + ERA5 profile → 3D via vertical MLP.

    For each spatial location, MLP(feat_vec, ERA5_profile) → nz output values.
    """

    def __init__(self, feat_dim: int = 64, nz: int = 32,
                 era5_input_dim: int = 1440, hidden: int = 128):
        super().__init__()
        self.nz = nz
        in_dim = feat_dim + era5_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, nz),
        )

    def forward(self, feat2d: torch.Tensor,
                era5: torch.Tensor) -> torch.Tensor:
        """
        feat2d: (B, feat_dim, H, W)
        era5:   (B, n_vars, nz)
        """
        B, C, H, W = feat2d.shape
        # Concat feat + ERA5 at each spatial location
        era5_flat = era5.flatten(1)  # (B, n_vars*nz)
        era5_bc = era5_flat[:, :, None, None].expand(B, -1, H, W)
        # (B, feat+era5, H, W)
        x = torch.cat([feat2d, era5_bc], dim=1)
        # (B, H, W, feat+era5) → MLP → (B, H, W, nz)
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        # (B, 1, H, W, nz)
        return x.permute(0, 3, 1, 2).unsqueeze(1).permute(0, 1, 3, 4, 2)


class TerrainViT_S2(nn.Module):
    """S2: Factored — 2D transformer + per-location vertical MLP."""

    def __init__(self, img_size=128, patch_size=8, nz=32,
                 embed_dim=384, depth=12, n_heads=8,
                 mlp_ratio=4.0, drop=0.1, feat_dim=64,
                 era5_input_dim=1440, n_output_vars=5):
        super().__init__()
        pg = img_size // patch_size
        self.patch_grid = pg

        self.patch_embed = PatchEmbed2D(2, embed_dim, img_size, patch_size)
        self.era5_enc = ERA5Encoder(era5_input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = UpsampleDecoder2D(embed_dim, feat_dim, pg)
        self.heads = nn.ModuleList([
            VerticalMLPHead(feat_dim, nz, era5_input_dim)
            for _ in range(n_output_vars)])
        self.apply(_init_weights)

    def forward(self, terrain, era5):
        tokens = self.patch_embed(terrain)
        tokens = tokens + self.era5_enc(era5).unsqueeze(1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        pg = self.patch_grid
        feat2d = tokens.transpose(1, 2).view(
            tokens.shape[0], -1, pg, pg)
        feat2d = self.upsample(feat2d)
        return torch.cat([h(feat2d, era5) for h in self.heads], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# S3 — Cross-attention: terrain tokens × ERA5 profile tokens
# ═══════════════════════════════════════════════════════════════════════════

class ERA5TokenEncoder(nn.Module):
    """Encode ERA5 profiles into a sequence of tokens for cross-attention.

    (B, n_vars, nz) → (B, n_tokens, embed_dim)
    Uses Conv1d to create tokens from profile slices.
    """

    def __init__(self, era5_input_dim: int = 1440,
                 embed_dim: int = 384, n_tokens: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(era5_input_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim * n_tokens),
        )
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        B = profiles.shape[0]
        x = self.proj(profiles.flatten(1))
        x = x.view(B, self.n_tokens, self.embed_dim)
        return x + self.pos_embed


class TerrainViT_S3(nn.Module):
    """S3: Cross-attention — terrain queries attend to ERA5 key/values."""

    def __init__(self, img_size=128, patch_size=8, nz=32,
                 embed_dim=384, depth=12, n_heads=8,
                 mlp_ratio=4.0, drop=0.1, feat_dim=64,
                 era5_input_dim=1440, n_output_vars=5,
                 n_cross_layers=4, n_era5_tokens=16):
        super().__init__()
        pg = img_size // patch_size
        self.patch_grid = pg
        n_self = depth - n_cross_layers

        self.patch_embed = PatchEmbed2D(2, embed_dim, img_size, patch_size)
        self.era5_tokens = ERA5TokenEncoder(
            era5_input_dim, embed_dim, n_era5_tokens)

        # Cross-attention layers first (ERA5 → terrain)
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, n_heads, drop)
            for _ in range(n_cross_layers)])
        # Self-attention layers
        self.self_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(n_self)])
        self.norm = nn.LayerNorm(embed_dim)

        self.upsample = UpsampleDecoder2D(embed_dim, feat_dim, pg)
        self.heads = nn.ModuleList([
            FiLMVerticalHead(feat_dim, nz, era5_input_dim)
            for _ in range(n_output_vars)])
        self.apply(_init_weights)

    def forward(self, terrain, era5):
        t_tokens = self.patch_embed(terrain)     # (B, 256, dim)
        e_tokens = self.era5_tokens(era5)        # (B, 16, dim)

        # Cross-attention: terrain attends to ERA5
        for blk in self.cross_blocks:
            t_tokens = blk(t_tokens, e_tokens)
        # Self-attention on enriched terrain tokens
        for blk in self.self_blocks:
            t_tokens = blk(t_tokens)
        t_tokens = self.norm(t_tokens)

        pg = self.patch_grid
        feat2d = t_tokens.transpose(1, 2).view(
            t_tokens.shape[0], -1, pg, pg)
        feat2d = self.upsample(feat2d)
        return torch.cat([h(feat2d, era5) for h in self.heads], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════

_PRESETS = {
    "small": dict(embed_dim=256, depth=8, n_heads=8, drop=0.1, feat_dim=48),
    "base": dict(embed_dim=384, depth=12, n_heads=8, drop=0.1, feat_dim=64),
}

_VARIANTS = {
    "film": TerrainViT_S1,
    "factor": TerrainViT_S2,
    "cross": TerrainViT_S3,
}


def build_vit(variant: str = "film", preset: str = "base",
              **overrides) -> nn.Module:
    """Build a TerrainViT variant.

    Variants: "film" (S1), "factor" (S2), "cross" (S3)
    Presets:  "small" (~10-15M), "base" (~20-30M)
    """
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(_VARIANTS)}")
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(_PRESETS)}")
    cfg = {**_PRESETS[preset], **overrides}
    return _VARIANTS[variant](**cfg)
