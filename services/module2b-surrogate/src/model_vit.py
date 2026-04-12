"""
model_vit.py — Vision Transformer for terrain-aware 3D wind downscaling.

V2: FuXi-CFD fusion architecture (Lin et al., Nature Communications 2026).

Key changes vs V1:
  - ERA5 profiles broadcast to 2D and CONCATENATED with terrain/z0 (not added)
  - Downsample conv layer fuses all inputs before patchification
  - Heavy 3D decoder with progressive upsampling (not linear projection)
  - Per-variable decoder heads with shared trunk

Architecture (Fig. 8 of FuXi-CFD):
  1. Input: terrain(128,128) + z0(128,128) + ERA5 broadcast(5ch, 128,128) → 7ch
  2. Downsample conv layers → reduced spatial dim
  3. Reshape to patch tokens + positional embedding
  4. Transformer encoder (N blocks)
  5. Reshape back to 2D feature map → upsample to full res
  6. Expand to 3D (vertical) → per-variable conv3d heads
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleFusion(nn.Module):
    """Fuse multi-channel 2D input and downsample to patch grid.

    FuXi-CFD style: conv layers reduce (B, C_in, H, W) to (B, embed_dim, pg, pg)
    where pg = H / total_stride.

    Input:  (B, 7, 128, 128) — terrain + z0 + 5 ERA5 channels
    Output: (B, embed_dim, pg, pg)
    """

    def __init__(self, in_channels: int = 7, embed_dim: int = 384,
                 img_size: int = 128, patch_size: int = 8):
        super().__init__()
        # Progressive downsampling: 128 → 64 → 32 → 16 (for patch_size=8)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),  # /2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        x = self.layers(x)  # (B, embed_dim, pg, pg)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
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


class UpsampleDecoder2D(nn.Module):
    """Upsample 2D feature map from patch grid to full resolution.

    (B, embed_dim, pg, pg) → (B, feat_dim, H, W) via transposed convolutions.
    """

    def __init__(self, embed_dim: int = 384, feat_dim: int = 64,
                 patch_grid: int = 16, img_size: int = 128):
        super().__init__()
        # pg=16 → 32 → 64 → 128
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),  # ×2
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # ×4
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, feat_dim, 4, stride=2, padding=1),  # ×8
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # (B, feat_dim, H, W)


class DecoderHead3D(nn.Module):
    """Expand 2D features to 3D volume for one output variable.

    (B, feat_dim, H, W) → (B, 1, H, W, nz) via vertical expansion + conv3d.
    """

    def __init__(self, feat_dim: int = 64, nz: int = 32):
        super().__init__()
        self.nz = nz
        # Vertical expansion: project each spatial location to nz values
        self.vert_proj = nn.Linear(feat_dim, feat_dim * nz)

        # 3D refinement (operates on the full 3D volume)
        self.refine = nn.Sequential(
            nn.Conv3d(feat_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        """feat2d: (B, feat_dim, H, W) → (B, 1, H, W, nz)"""
        B, C, H, W = feat2d.shape
        # (B, C, H, W) → (B, H, W, C)
        x = feat2d.permute(0, 2, 3, 1)
        # (B, H, W, C) → (B, H, W, C*nz)
        x = self.vert_proj(x)
        # (B, H, W, C, nz) → (B, C, H, W, nz)
        x = x.view(B, H, W, C, self.nz).permute(0, 3, 1, 2, 4)
        # 3D convolution refinement → (B, 1, H, W, nz)
        x = self.refine(x)
        return x


class TerrainViT(nn.Module):
    """Vision Transformer for terrain-aware 3D wind field downscaling.

    V2 architecture — FuXi-CFD style input fusion.

    Input:
      - terrain_z0: (B, 2, 128, 128) — normalized elevation + roughness
      - era5:       (B, 5, 32) — ERA5 profiles [u, v, T, q, k]

    The ERA5 profiles are broadcast to (B, 5, 128, 128) taking the mean
    across z-levels per variable, then concatenated with terrain → 7 channels.

    Output: (B, 5, 128, 128, 32) — predicted residual fields (u, v, w, T, q)
    """

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 8,
        n_era5_vars: int = 5,
        nz: int = 32,
        embed_dim: int = 384,
        depth: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        n_output_vars: int = 5,
        decoder_feat_dim: int = 64,
    ):
        super().__init__()
        self.nz = nz
        self.n_output_vars = n_output_vars
        self.n_era5_vars = n_era5_vars
        self.img_size = img_size
        self.patch_grid = img_size // patch_size

        # ERA5 profile → per-level spatial maps
        # Project each (n_vars, nz) profile into n_era5_vars 2D channels
        self.era5_proj = nn.Sequential(
            nn.Linear(n_era5_vars * nz, 128),
            nn.GELU(),
            nn.Linear(128, n_era5_vars),
        )

        # Input fusion: 2 (terrain) + n_era5_vars channels → downsample → tokens
        self.downsample = DownsampleFusion(
            in_channels=2 + n_era5_vars,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
        )

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Shared 2D upsampler (tokens → full-res 2D feature map)
        self.upsample = UpsampleDecoder2D(
            embed_dim=embed_dim,
            feat_dim=decoder_feat_dim,
            patch_grid=self.patch_grid,
            img_size=img_size,
        )

        # Per-variable 3D decoder heads
        self.heads = nn.ModuleList([
            DecoderHead3D(feat_dim=decoder_feat_dim, nz=nz)
            for _ in range(n_output_vars)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        B = terrain.shape[0]
        H, W = terrain.shape[2], terrain.shape[3]

        # ERA5 profiles → summary channels broadcast to 2D
        era5_flat = era5.flatten(1)  # (B, n_vars * nz)
        era5_2d = self.era5_proj(era5_flat)  # (B, n_era5_vars)
        era5_spatial = era5_2d[:, :, None, None].expand(
            B, self.n_era5_vars, H, W)  # (B, 5, 128, 128)

        # Concatenate terrain + ERA5 → (B, 7, 128, 128)
        x = torch.cat([terrain, era5_spatial], dim=1)

        # Downsample + patchify → tokens
        tokens = self.downsample(x)  # (B, n_patches, embed_dim)

        # Transformer encoder
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Reshape tokens to 2D feature map
        pg = self.patch_grid
        feat2d = tokens.transpose(1, 2).view(
            B, -1, pg, pg)  # (B, embed_dim, pg, pg)

        # Shared 2D upsampling
        feat2d = self.upsample(feat2d)  # (B, feat_dim, H, W)

        # Per-variable 3D decoding
        outputs = [head(feat2d) for head in self.heads]
        return torch.cat(outputs, dim=1)  # (B, 5, H, W, nz)


def build_vit(preset: str = "base", **overrides) -> TerrainViT:
    """Build a TerrainViT with preset configurations.

    Presets:
      - "small":  embed_dim=256, depth=8,  heads=8,  feat=48  (~15M params)
      - "base":   embed_dim=384, depth=12, heads=8,  feat=64  (~45M params)
      - "large":  embed_dim=512, depth=16, heads=16, feat=96  (~100M params)
    """
    configs = {
        "small": dict(embed_dim=256, depth=8, n_heads=8, drop=0.1,
                       decoder_feat_dim=48),
        "base": dict(embed_dim=384, depth=12, n_heads=8, drop=0.1,
                      decoder_feat_dim=64),
        "large": dict(embed_dim=512, depth=16, n_heads=16, drop=0.1,
                       decoder_feat_dim=96),
    }
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs)}")
    cfg = {**configs[preset], **overrides}
    return TerrainViT(**cfg)
