"""
model_factored.py — Factored 2D+1D architecture for wind/T/q downscaling.

Physically motivated: horizontal terrain effects (advection, deflection) are
decoupled from vertical mixing (stratification, shear). Much lighter than 3D conv.

Architecture:
    Input: (B, C_in, Ny, Nx, Nz)

    1. Horizontal encoder (shared 2D U-Net across z-levels):
       For each z: (B, C_in, Ny, Nx) → (B, F, Ny, Nx)

    2. Vertical mixer (1D conv per column):
       For each (x,y): (B, F, Nz) → (B, F', Nz)
       Learns vertical dependencies (shear, stratification)

    3. Horizontal decoder (shared 2D across z-levels):
       For each z: (B, F', Ny, Nx) → (B, C_out, Ny, Nx)

    Output: (B, C_out, Ny, Nx, Nz)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """Two 2D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VerticalMixer(nn.Module):
    """1D convolutions along z-axis for vertical mixing.

    Processes each (x,y) column independently.
    """

    def __init__(self, channels: int, n_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*Ny*Nx, F, Nz) → (B*Ny*Nx, F, Nz)"""
        return x + self.net(x)  # residual


class FactoredUNet(nn.Module):
    """Factored 2D+1D architecture.

    Parameters
    ----------
    in_channels : 7 for volume variant
    out_channels : 5 (u, v, w, T, q)
    base_features : width of 2D encoder
    vertical_layers : number of 1D conv layers in vertical mixer
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 5,
        base_features: int = 32,
        vertical_layers: int = 4,
    ):
        super().__init__()
        f = base_features

        # ── 2D Encoder (shared across z) ──
        self.enc1 = ConvBlock2D(in_channels, f)
        self.enc2 = ConvBlock2D(f, f * 2)
        self.enc3 = ConvBlock2D(f * 2, f * 4)
        self.pool = nn.MaxPool2d(2)

        # ── Vertical mixer ──
        self.vmix = VerticalMixer(f * 4, n_layers=vertical_layers)

        # ── 2D Decoder (shared across z) ──
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock2D(f * 4, f * 2)  # cat with skip
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock2D(f * 2, f)  # cat with skip

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def _encode_2d(self, x: torch.Tensor) -> tuple:
        """x: (B*Nz, C_in, Ny, Nx) → features + skips"""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        return e3, e2, e1

    def _decode_2d(self, e3: torch.Tensor, e2: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
        """Decode with skip connections → (B*Nz, C_out, Ny, Nx)"""
        d2 = self.up2(e3)
        # Match spatial dims
        d2 = d2[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = d1[:, :, :e1.shape[2], :e1.shape[3]]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C_in, Ny, Nx, Nz)
        Returns: (B, C_out, Ny, Nx, Nz)
        """
        B, C, Ny, Nx, Nz = x.shape

        # Pad Ny, Nx to multiples of 4 (for 2 pooling layers)
        pad_y = (4 - Ny % 4) % 4
        pad_x = (4 - Nx % 4) % 4
        if pad_y > 0 or pad_x > 0:
            x = F.pad(x, [0, 0, 0, pad_x, 0, pad_y])
            Ny_p, Nx_p = Ny + pad_y, Nx + pad_x
        else:
            Ny_p, Nx_p = Ny, Nx

        # ── Step 1: 2D encode per z-level ──
        # Reshape: (B, C, Ny, Nx, Nz) → (B*Nz, C, Ny, Nx)
        x_2d = x.permute(0, 4, 1, 2, 3).reshape(B * Nz, C, Ny_p, Nx_p)
        e3, e2, e1 = self._encode_2d(x_2d)

        # e3: (B*Nz, F*4, Ny//4, Nx//4)
        F4, Hy, Hx = e3.shape[1], e3.shape[2], e3.shape[3]

        # ── Step 2: Vertical mixing at bottleneck ──
        # Reshape: (B*Nz, F*4, Hy, Hx) → (B, Nz, F*4, Hy, Hx) → (B*Hy*Hx, F*4, Nz)
        e3_vol = e3.reshape(B, Nz, F4, Hy, Hx).permute(0, 3, 4, 2, 1).reshape(B * Hy * Hx, F4, Nz)
        e3_mixed = self.vmix(e3_vol)

        # Reshape back: (B*Hy*Hx, F*4, Nz) → (B*Nz, F*4, Hy, Hx)
        e3_out = e3_mixed.reshape(B, Hy, Hx, F4, Nz).permute(0, 4, 3, 1, 2).reshape(B * Nz, F4, Hy, Hx)

        # ── Step 3: 2D decode per z-level ──
        out_2d = self._decode_2d(e3_out, e2, e1)  # (B*Nz, C_out, Ny, Nx)

        C_out = out_2d.shape[1]
        # Reshape: (B*Nz, C_out, Ny, Nx) → (B, Nz, C_out, Ny, Nx) → (B, C_out, Ny, Nx, Nz)
        out = out_2d.reshape(B, Nz, C_out, Ny_p, Nx_p).permute(0, 2, 3, 4, 1)

        # Crop padding
        out = out[:, :, :Ny, :Nx, :]
        return out
