"""Dataset utilities for DownscalRain CNN patch-to-point training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PatchDatasetStats:
    patch_mean: list[float]
    patch_std: list[float]
    meta_mean: list[float]
    meta_std: list[float]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PatchDatasetStats":
        return cls(**json.loads(Path(path).read_text()))


def _load_array(path: Path, mmap_mode: str | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=True)


def _as_str_array(values: Any, n: int, default: str = "") -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=object)
    arr = np.asarray(values)
    if len(arr) != n:
        raise ValueError(f"expected array length {n}, got {len(arr)}")
    return arr.astype(object)


class RainPatchDataset(Dataset):
    """Torch dataset backed by a directory or NPZ patch dataset.

    Directory format:
      - `patches.npy`: float array `(N, C, H, W)`
      - `meta.npy`: float array `(N, M)`; may be absent
      - `rain.npy`: float array `(N,)`
      - `station_id.npy`: optional string/object array `(N,)`
      - `date.npy`: optional string/object array `(N,)`
      - `channels.json`: optional list of channel names
      - `meta_columns.json`: optional list of metadata names

    NPZ format uses the same keys. Directory format is preferred for large
    datasets because `patches.npy` can be memory-mapped.
    """

    def __init__(
        self,
        path: str | Path,
        indices: Sequence[int] | np.ndarray | None = None,
        stats: PatchDatasetStats | None = None,
        mmap: bool = True,
    ) -> None:
        self.path = Path(path)
        mmap_mode = "r" if mmap else None

        if self.path.is_dir():
            self.patches = _load_array(self.path / "patches.npy", mmap_mode=mmap_mode)
            self.rain = _load_array(self.path / "rain.npy", mmap_mode=mmap_mode).astype(np.float32)
            meta_path = self.path / "meta.npy"
            if meta_path.exists():
                self.meta = _load_array(meta_path, mmap_mode=mmap_mode).astype(np.float32)
            else:
                self.meta = np.zeros((len(self.rain), 0), dtype=np.float32)
            station_path = self.path / "station_id.npy"
            date_path = self.path / "date.npy"
            self.station_ids = _as_str_array(
                _load_array(station_path, mmap_mode=None) if station_path.exists() else None,
                len(self.rain),
            )
            self.dates = _as_str_array(
                _load_array(date_path, mmap_mode=None) if date_path.exists() else None,
                len(self.rain),
            )
            self.channels = _read_json_list(self.path / "channels.json")
            self.meta_columns = _read_json_list(self.path / "meta_columns.json")
        else:
            data = np.load(self.path, allow_pickle=True)
            self.patches = data["patches"].astype(np.float32)
            self.rain = data["rain"].astype(np.float32)
            self.meta = data["meta"].astype(np.float32) if "meta" in data else np.zeros((len(self.rain), 0), dtype=np.float32)
            self.station_ids = _as_str_array(data["station_id"] if "station_id" in data else None, len(self.rain))
            self.dates = _as_str_array(data["date"] if "date" in data else None, len(self.rain))
            self.channels = list(data["channels"].astype(str)) if "channels" in data else []
            self.meta_columns = list(data["meta_columns"].astype(str)) if "meta_columns" in data else []

        if self.patches.ndim != 4:
            raise ValueError(f"patches must have shape (N, C, H, W), got {self.patches.shape}")
        if len(self.patches) != len(self.rain):
            raise ValueError("patches and rain must have the same first dimension")
        if self.meta.ndim != 2 or len(self.meta) != len(self.rain):
            raise ValueError("meta must have shape (N, M)")

        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.rain)), dtype=np.int64)
        self.stats = stats

    @property
    def n_channels(self) -> int:
        return int(self.patches.shape[1])

    @property
    def patch_size(self) -> tuple[int, int]:
        return int(self.patches.shape[2]), int(self.patches.shape[3])

    @property
    def meta_dim(self) -> int:
        return int(self.meta.shape[1])

    def subset(self, indices: Sequence[int] | np.ndarray, stats: PatchDatasetStats | None = None) -> "RainPatchDataset":
        return RainPatchDataset(self.path, indices=indices, stats=stats or self.stats)

    def with_stats(self, stats: PatchDatasetStats) -> "RainPatchDataset":
        return RainPatchDataset(self.path, indices=self.indices, stats=stats)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        patch = np.asarray(self.patches[idx], dtype=np.float32)
        meta = np.asarray(self.meta[idx], dtype=np.float32)
        rain = np.float32(self.rain[idx])

        if self.stats is not None:
            patch_mean = np.asarray(self.stats.patch_mean, dtype=np.float32)[:, None, None]
            patch_std = np.asarray(self.stats.patch_std, dtype=np.float32)[:, None, None]
            patch = (patch - patch_mean) / patch_std
            if meta.size:
                meta_mean = np.asarray(self.stats.meta_mean, dtype=np.float32)
                meta_std = np.asarray(self.stats.meta_std, dtype=np.float32)
                meta = (meta - meta_mean) / meta_std

        return {
            "patch": torch.from_numpy(patch.copy()),
            "meta": torch.from_numpy(meta.copy()),
            "rain": torch.tensor(rain, dtype=torch.float32),
            "station_id": str(self.station_ids[idx]),
            "date": str(self.dates[idx]),
        }


def _read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text()))


def compute_stats(
    dataset: RainPatchDataset,
    indices: Sequence[int] | np.ndarray | None = None,
    chunk_size: int = 2048,
    eps: float = 1e-6,
) -> PatchDatasetStats:
    """Compute channel and metadata normalization stats on selected samples."""
    idx = np.asarray(indices if indices is not None else dataset.indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("cannot compute stats on an empty index set")

    c = dataset.n_channels
    m = dataset.meta_dim
    patch_sum = np.zeros(c, dtype=np.float64)
    patch_sumsq = np.zeros(c, dtype=np.float64)
    patch_count = 0
    meta_sum = np.zeros(m, dtype=np.float64)
    meta_sumsq = np.zeros(m, dtype=np.float64)

    for start in range(0, idx.size, chunk_size):
        batch_idx = idx[start : start + chunk_size]
        patch = np.asarray(dataset.patches[batch_idx], dtype=np.float64)
        patch_sum += patch.sum(axis=(0, 2, 3))
        patch_sumsq += np.square(patch).sum(axis=(0, 2, 3))
        patch_count += patch.shape[0] * patch.shape[2] * patch.shape[3]
        if m:
            meta = np.asarray(dataset.meta[batch_idx], dtype=np.float64)
            meta_sum += meta.sum(axis=0)
            meta_sumsq += np.square(meta).sum(axis=0)

    patch_mean = patch_sum / patch_count
    patch_var = np.maximum(patch_sumsq / patch_count - np.square(patch_mean), eps)
    patch_std = np.sqrt(patch_var)

    if m:
        meta_mean = meta_sum / idx.size
        meta_var = np.maximum(meta_sumsq / idx.size - np.square(meta_mean), eps)
        meta_std = np.sqrt(meta_var)
    else:
        meta_mean = np.zeros(0, dtype=np.float64)
        meta_std = np.ones(0, dtype=np.float64)

    return PatchDatasetStats(
        patch_mean=patch_mean.astype(float).tolist(),
        patch_std=patch_std.astype(float).tolist(),
        meta_mean=meta_mean.astype(float).tolist(),
        meta_std=meta_std.astype(float).tolist(),
    )


def station_group_split(
    station_ids: Sequence[Any],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split rows by station id to avoid station leakage."""
    station_ids_arr = np.asarray(station_ids).astype(str)
    unique = np.unique(station_ids_arr)
    if unique.size < 3:
        raise ValueError("need at least 3 unique stations for train/val/test split")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(unique.size * test_fraction)))
    n_val = max(1, int(round(unique.size * val_fraction)))
    if n_test + n_val >= unique.size:
        n_test = 1
        n_val = 1

    test_stations = set(unique[:n_test])
    val_stations = set(unique[n_test : n_test + n_val])
    split = {"train": [], "val": [], "test": []}
    for i, sid in enumerate(station_ids_arr):
        if sid in test_stations:
            split["test"].append(i)
        elif sid in val_stations:
            split["val"].append(i)
        else:
            split["train"].append(i)
    return {name: np.asarray(values, dtype=np.int64) for name, values in split.items()}


def save_split_manifest(split: dict[str, np.ndarray], path: str | Path) -> None:
    serializable = {name: values.astype(int).tolist() for name, values in split.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))


def load_split_manifest(path: str | Path) -> dict[str, np.ndarray]:
    raw = json.loads(Path(path).read_text())
    return {name: np.asarray(values, dtype=np.int64) for name, values in raw.items()}


def write_patch_dataset(
    output_dir: str | Path,
    patches: np.ndarray,
    rain: np.ndarray,
    meta: np.ndarray | None = None,
    station_ids: Sequence[Any] | None = None,
    dates: Sequence[Any] | None = None,
    channels: Sequence[str] | None = None,
    meta_columns: Sequence[str] | None = None,
) -> None:
    """Write a directory-format patch dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patches = np.asarray(patches, dtype=np.float32)
    np.save(output_dir / "patches.npy", patches)
    write_patch_dataset_metadata(
        output_dir=output_dir,
        rain=rain,
        meta=meta,
        station_ids=station_ids,
        dates=dates,
        channels=channels,
        meta_columns=meta_columns,
    )


def write_patch_dataset_metadata(
    output_dir: str | Path,
    rain: np.ndarray,
    meta: np.ndarray | None = None,
    station_ids: Sequence[Any] | None = None,
    dates: Sequence[Any] | None = None,
    channels: Sequence[str] | None = None,
    meta_columns: Sequence[str] | None = None,
) -> None:
    """Write directory-format labels and metadata next to an existing patches.npy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rain = np.asarray(rain, dtype=np.float32)
    if meta is None:
        meta = np.zeros((len(rain), 0), dtype=np.float32)
    meta = np.asarray(meta, dtype=np.float32)

    np.save(output_dir / "rain.npy", rain)
    np.save(output_dir / "meta.npy", meta)
    np.save(output_dir / "station_id.npy", _as_str_array(station_ids, len(rain)))
    np.save(output_dir / "date.npy", _as_str_array(dates, len(rain)))
    (output_dir / "channels.json").write_text(json.dumps(list(channels or []), indent=2))
    (output_dir / "meta_columns.json").write_text(json.dumps(list(meta_columns or []), indent=2))
