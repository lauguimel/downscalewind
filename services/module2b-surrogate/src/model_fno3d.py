"""
model_fno3d.py — Fourier Neural Operator 3D for wind/T/q downscaling.

FNO learns the solution operator in Fourier space: global receptive field,
resolution-invariant (train at 128³, infer at 256³).

Architecture:
    Input:  (B, C_in, Ny, Nx, Nz)
    → Lift:  Conv3d 1×1×1 → (B, width, Ny, Nx, Nz)
    → N × [SpectralConv3d + Conv3d skip + GELU]
    → Project: Conv3d 1×1×1 → (B, C_out, Ny, Nx, Nz)

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """3D Fourier layer: multiply truncated Fourier modes by learnable weights."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Ny modes
        self.modes2 = modes2  # Nx modes
        self.modes3 = modes3  # Nz modes (rfft → half spectrum)

        scale = 1.0 / (in_channels * out_channels)
        # 4 octants of the 3D Fourier space (due to symmetry of rfft)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def _compl_mul3d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (batch, in, x, y, z) × (in, out, x, y, z) → (batch, out, x, y, z)"""
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # 3D rFFT (real-to-complex on last dim)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft = torch.zeros(
            B, self.out_channels, x.shape[-3], x.shape[-2], x.shape[-1] // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # 4 octants
        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1
        )
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2
        )
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4
        )

        return torch.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]))


class FNO3D(nn.Module):
    """3D Fourier Neural Operator for wind/T/q downscaling.

    Parameters
    ----------
    in_channels : input channels (7 for volume variant)
    out_channels : output channels (5: u, v, w, T, q)
    width : hidden channel dimension
    modes : (modes_y, modes_x, modes_z) Fourier modes to keep
    n_layers : number of Fourier layers
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 5,
        width: int = 32,
        modes: tuple[int, int, int] = (16, 16, 8),
        n_layers: int = 4,
    ):
        super().__init__()
        self.width = width
        self.n_layers = n_layers

        # Lifting: project input channels → width
        self.lift = nn.Conv3d(in_channels, width, 1)

        # Fourier layers
        self.spectral_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(
                SpectralConv3d(width, width, modes[0], modes[1], modes[2])
            )
            self.skip_convs.append(nn.Conv3d(width, width, 1))
            self.norms.append(nn.InstanceNorm3d(width))

        # Projection: width → out_channels (2-layer MLP per voxel)
        self.proj1 = nn.Conv3d(width, width * 2, 1)
        self.proj2 = nn.Conv3d(width * 2, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C_in, Ny, Nx, Nz)
        Returns: (B, C_out, Ny, Nx, Nz)
        """
        # Pad to power-of-2-friendly sizes for FFT efficiency
        orig_shape = x.shape[2:]
        pad = []
        for s in reversed(orig_shape):
            target = ((s + 7) // 8) * 8
            pad.extend([0, target - s])
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)

        x = self.lift(x)

        for i in range(self.n_layers):
            x_s = self.spectral_convs[i](x)
            x_l = self.skip_convs[i](x)
            x = self.norms[i](x_s + x_l)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        # Crop to original size
        x = x[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
        return x
