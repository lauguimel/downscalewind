"""
model_unet3d.py — U-Net 3D for wind downscaling on regular grid.

Architecture:
    Input:  (B, C_in, Ny, Nx, Nz) — terrain + inflow profile
    Output: (B, 3, Ny, Nx, Nz)    — u, v, w velocity

3-level encoder-decoder with skip connections.
Typical input: (B, 3, 33, 33, 18) → output: (B, 3, 33, 33, 18)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """3-level U-Net for 3D wind field prediction."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f * 2)
        self.enc3 = ConvBlock3D(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(f * 8, f * 4)  # concat with enc3

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(f * 4, f * 2)  # concat with enc2

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(f * 2, f)  # concat with enc1

        self.out_conv = nn.Conv3d(f, out_channels, 1)

        # Pooling
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, D, H, W) — input volume

        Returns
        -------
        (B, C_out, D, H, W) — predicted velocity field
        """
        # Pad to ensure dimensions are divisible by 8 (3 pooling levels)
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
        out = out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
        return out

    @staticmethod
    def _match_and_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Match spatial dimensions and concatenate skip connection."""
        # Crop up to match skip (or vice versa)
        d = min(up.shape[2], skip.shape[2])
        h = min(up.shape[3], skip.shape[3])
        w = min(up.shape[4], skip.shape[4])
        return torch.cat([up[:, :, :d, :h, :w], skip[:, :, :d, :h, :w]], dim=1)
