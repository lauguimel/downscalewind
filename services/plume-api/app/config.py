"""Plume API settings — loaded from env vars / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLUME_", env_file=".env", extra="ignore")

    # Model
    model_path: Path = Path("models/fno3d_9k.ts.pt")
    model_version: str = "fno3d-9k-v1"

    # Domain / grid (must match training)
    grid_nx: int = 128
    grid_ny: int = 128
    grid_nz: int = 32
    domain_km: float = 4.0

    # Normalization (must match dataset_sf.py)
    terrain_scale: float = 500.0
    z0_scale: float = 1.0
    wind_scale: float = 20.0
    t_scale: float = 30.0
    q_scale: float = 0.01

    # Cache
    cache_dir: Path = Path("cache")
    terrain_cache_ttl_days: int = 365

    # Rate limiting
    rate_limit_per_day: int = 100
    rate_limit_burst_per_min: int = 10

    # External APIs
    openmeteo_base_url: str = "https://api.open-meteo.com/v1"

    # Demo data
    demo_data_dir: Path = Path("app/demo_data")


settings = Settings()
