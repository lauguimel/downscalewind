from __future__ import annotations

import os

import numpy as np
import torch

from .dataset_v2 import parse_agl_levels
from .dataset_v2_vit import WindV2DatasetViT


class WindV2DatasetViT_E2(WindV2DatasetViT):
    """Append OBS fields to the parent tuple.

    Return order:
      terrain, era5, [geo], target, [weight], obs_value, obs_mask, obs_ij,
      case_name

    `obs_value` is the normalized horizontal target speed at a random `(i, j)`
    and `self.obs_k`; `obs_mask` is 1 when present and 0 when dropped;
    `obs_ij` is an int64 tensor `(i, j)` in native 180x180 pixel coordinates.
    """

    def __init__(self, data_dir, splits_yaml, split="train", *,
                 obs_dropout=0.5, obs_height_m=10.0,
                 obs_agl_level_idx=None, **kwargs):
        data_dir = os.path.realpath(data_dir)
        splits_yaml = os.path.realpath(splits_yaml)
        super().__init__(data_dir, splits_yaml, split, **kwargs)
        self.obs_dropout = float(obs_dropout)
        self.obs_height_m = float(obs_height_m)
        if obs_agl_level_idx is not None:
            self.obs_k = int(obs_agl_level_idx)
        elif self.target_agl_levels is not None:
            self.obs_k = int(np.argmin(np.abs(self.target_agl_levels - self.obs_height_m)))
        else:
            raise ValueError("obs_agl_level_idx is required when target_agl_levels is native")
        nz = len(parse_agl_levels(self.target_agl_levels)) if self.target_agl_levels is not None else None
        if nz is not None and not 0 <= self.obs_k < nz:
            raise ValueError(f"obs_k={self.obs_k} outside target nz={nz}")

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        target_idx = 3 if self.return_geo else 2
        target = base[target_idx]

        if np.random.rand() < self.obs_dropout:
            obs_value = torch.zeros(1, dtype=torch.float32)
            obs_mask = torch.zeros(1, dtype=torch.float32)
            obs_ij = torch.zeros(2, dtype=torch.long)
        else:
            i = int(np.random.randint(0, target.shape[1]))
            j = int(np.random.randint(0, target.shape[2]))
            u = float(target[0, i, j, self.obs_k])
            v = float(target[1, i, j, self.obs_k])
            obs_value = torch.tensor([np.sqrt(u * u + v * v)], dtype=torch.float32)
            obs_mask = torch.ones(1, dtype=torch.float32)
            obs_ij = torch.tensor([i, j], dtype=torch.long)
        return (*base[:-1], obs_value, obs_mask, obs_ij, base[-1])
