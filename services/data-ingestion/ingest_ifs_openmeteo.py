"""
ingest_ifs_openmeteo.py — Ingestion IFS ECMWF via Open-Meteo pour DownscaleWind.

Télécharge les données IFS (ECMWF) depuis l'API Open-Meteo Historical Forecast
sur une grille centrée sur un site de référence.

Usage :
    # Inférence opérationnelle (IFS 0.25°, depuis 2024-02-03)
    python ingest_ifs_openmeteo.py --site perdigao \\
        --start 2024-02-03 --end 2024-12-31 \\
        --model ecmwf_ifs025 \\
        --output ../../data/raw/ifs_perdigao.zarr

    # Caractérisation domain shift ERA5→IFS sur période Perdigão 2017
    python ingest_ifs_openmeteo.py --site perdigao \\
        --start 2017-01-01 --end 2017-12-31 \\
        --model ecmwf_ifs_analysis_long_window \\
        --output ../../data/raw/ifs_hres_perdigao.zarr

Notes :
    - Open-Meteo fournit wind_speed + wind_direction, pas u/v directement.
      Reconstruction : u = -speed*sin(dir_rad), v = -speed*cos(dir_rad)
      (convention météo : vent VENANT DE la direction indiquée).
    - L'humidité spécifique (q) n'est pas disponible → stockée comme NaN.
      Conversion RH → q possible en post-traitement via shared.data_io.
    - Pas d'authentification requise. Limite de taux non documentée → délai 0.1s
      entre requêtes pour éviter le rate-limiting.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import yaml
from retry_requests import retry

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import (
    create_empty_store,
    open_store,
    sha256_file,
    wind_speed_direction_to_uv,
    relative_humidity_to_specific_humidity,
)
from utils.checkpointing import Checkpointer

log = get_logger("ingest_ifs_openmeteo")

# ── Constantes ────────────────────────────────────────────────────────────────

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]

# Modèles supportés et leur résolution temporelle
MODELS = {
    "ecmwf_ifs025": {
        "description":    "IFS ECMWF 0.25°",
        "available_from": "2024-02-03",
        "temporal_res_h": 3,
    },
    "ecmwf_ifs_analysis_long_window": {
        "description":    "IFS HRES 9km analysis",
        "available_from": "2017-01-01",
        "temporal_res_h": 1,
    },
    "ecmwf_ifs04": {
        "description":    "IFS ECMWF 0.4°",
        "available_from": "2022-11-07",
        "temporal_res_h": 3,
    },
}

# Variables demandées pour chaque niveau de pression
def _pressure_vars(levels: list[int]) -> list[str]:
    """Génère la liste des variables Open-Meteo pour les niveaux de pression."""
    vars_ = []
    for l in levels:
        vars_.extend([
            f"wind_speed_{l}hPa",
            f"wind_direction_{l}hPa",
            f"temperature_{l}hPa",
            f"geopotential_height_{l}hPa",
            f"relative_humidity_{l}hPa",
        ])
    return vars_


# ── Initialisation du client Open-Meteo ───────────────────────────────────────

def _build_client() -> openmeteo_requests.Client:
    """Client Open-Meteo avec cache SQLite et retry automatique."""
    cache_session = requests_cache.CachedSession(
        "http_cache", expire_after=3600  # cache 1h
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.5)
    return openmeteo_requests.Client(session=retry_session)


# ── Téléchargement pour un point géographique ─────────────────────────────────

def _fetch_point(
    client,
    lat: float,
    lon: float,
    model: str,
    start_date: str,
    end_date: str,
    levels: list[int],
) -> dict[str, pd.Series]:
    """
    Télécharge les données IFS pour un point géographique.

    Returns:
        dict {variable_name: pd.Series avec index DatetimeIndex UTC}
    """
    hourly_vars = _pressure_vars(levels) + [
        "wind_speed_10m", "wind_direction_10m", "temperature_2m",
    ]

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "models":     model,
        "hourly":     hourly_vars,
        "start_date": start_date,
        "end_date":   end_date,
        "wind_speed_unit": "ms",  # m/s (pas km/h)
        "temperature_unit": "kelvin",
        "timezone":   "UTC",
    }

    responses = client.weather_api(API_URL, params=params)
    response = responses[0]

    hourly = response.Hourly()
    # Reconstruire l'index temporel
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    result = {}
    for i, var in enumerate(hourly_vars):
        result[var] = pd.Series(hourly.Variables(i).ValuesAsNumpy(), index=timestamps)

    return result


# ── Reconstruction u/v depuis speed+direction ─────────────────────────────────

def _extract_uv_for_level(
    data: dict[str, pd.Series],
    level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait et reconstitue u/v pour un niveau de pression donné.

    Convention météo : vent VENANT DE direction_deg
        u = -speed * sin(dir_rad)  → composante zonale (Est positif)
        v = -speed * cos(dir_rad)  → composante méridionale (Nord positif)
    """
    speed = data[f"wind_speed_{level}hPa"].values.astype(np.float32)
    direction = data[f"wind_direction_{level}hPa"].values.astype(np.float32)
    return wind_speed_direction_to_uv(speed, direction)


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--site",   required=True, help="Identifiant du site (ex: perdigao)")
@click.option("--start",  required=True, help="Date de début (YYYY-MM-DD)")
@click.option("--end",    required=True, help="Date de fin (YYYY-MM-DD, inclusif)")
@click.option("--model",  default="ecmwf_ifs025",
              type=click.Choice(list(MODELS.keys())),
              help="Modèle IFS à télécharger")
@click.option("--output", required=True, help="Chemin du store Zarr de sortie")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Répertoire des configurations de sites")
@click.option("--checkpoint-dir", default=None,
              help="Répertoire des sentinelles")
@click.option("--chunk-days", default=30,
              help="Nombre de jours par chunk de téléchargement")
@click.option("--dry-run", is_flag=True,
              help="Affiche les requêtes sans télécharger")
def main(site, start, end, model, output, config_dir, checkpoint_dir, chunk_days, dry_run):
    """
    Télécharge les données IFS ECMWF via Open-Meteo pour un site et une période.

    Les composantes u/v sont reconstruites depuis la vitesse et la direction
    météorologique (convention : vent VENANT DE). L'humidité spécifique n'est
    pas disponible dans l'API Open-Meteo — les champs q restent à NaN.
    """
    model_info = MODELS[model]
    log.info("Démarrage ingestion IFS Open-Meteo", extra={
        "site": site, "model": model, "start": start, "end": end,
        "model_description": model_info["description"],
        "dry_run": dry_run,
    })

    # ── Configuration du site ─────────────────────────────────────────────────
    config_path = Path(config_dir) / f"{site}.yaml"
    if not config_path.exists():
        log.error("Configuration de site introuvable", extra={"path": str(config_path)})
        sys.exit(1)

    with open(config_path) as f:
        site_cfg = yaml.safe_load(f)

    domain = site_cfg["era5_domain"]
    step = domain["grid_spacing_deg"]
    lats = np.arange(domain["north"], domain["south"] - step/2, -step, dtype=np.float32)
    lons = np.arange(domain["west"],  domain["east"]  + step/2,  step, dtype=np.float32)
    n_lat, n_lon = len(lats), len(lons)

    log.info("Grille IFS", extra={
        "n_lat": n_lat, "n_lon": n_lon, "n_points": n_lat * n_lon,
        "temporal_res_h": model_info["temporal_res_h"],
    })

    if dry_run:
        log.info("Mode dry-run — aucun téléchargement effectué")
        _print_sample_request(lats[0], lons[0], model, start, end)
        return

    # ── Checkpointer ──────────────────────────────────────────────────────────
    cp_dir = checkpoint_dir or str(Path(output) / ".checkpoints")
    cp = Checkpointer(cp_dir)

    # ── Découpage en chunks temporels ─────────────────────────────────────────
    chunks = list(_iter_date_chunks(start, end, chunk_days))
    log.info("Plan de téléchargement", extra={
        "n_chunks": len(chunks), "chunk_days": chunk_days,
        "n_grid_points": n_lat * n_lon,
    })

    # ── Téléchargement ─────────────────────────────────────────────────────────
    client = _build_client()

    for chunk_start, chunk_end in chunks:
        chunk_key = f"ifs_{model}_{chunk_start}_{chunk_end}"

        if cp.is_done(chunk_key):
            log.info("Chunk déjà traité", extra={"key": chunk_key})
            continue

        log.info("Début chunk", extra={"key": chunk_key,
                                       "n_points": n_lat * n_lon})

        # Nombre de pas de temps dans ce chunk
        n_days_chunk = (
            pd.Timestamp(chunk_end) - pd.Timestamp(chunk_start)
        ).days + 1
        res_h = model_info["temporal_res_h"]
        n_times_chunk = n_days_chunk * 24 // res_h

        # Initialiser les arrays de ce chunk [n_times, n_levels, n_lat, n_lon]
        n_levels = len(PRESSURE_LEVELS)
        arr_u   = np.full((n_times_chunk, n_levels, n_lat, n_lon), np.nan, np.float32)
        arr_v   = np.full_like(arr_u, np.nan)
        arr_z   = np.full_like(arr_u, np.nan)
        arr_t   = np.full_like(arr_u, np.nan)
        arr_rh  = np.full_like(arr_u, np.nan)
        arr_u10 = np.full((n_times_chunk, n_lat, n_lon), np.nan, np.float32)
        arr_v10 = np.full_like(arr_u10, np.nan)
        arr_t2m = np.full_like(arr_u10, np.nan)
        timestamps = None

        # Télécharger point par point
        for i_lat, lat in enumerate(lats):
            for i_lon, lon in enumerate(lons):
                log.info("Requête point", extra={
                    "lat": float(lat), "lon": float(lon),
                    "chunk": chunk_key,
                    "progress": f"{i_lat * n_lon + i_lon + 1}/{n_lat * n_lon}",
                })

                try:
                    data = _fetch_point(
                        client, float(lat), float(lon),
                        model, chunk_start, chunk_end,
                        PRESSURE_LEVELS,
                    )
                except Exception as e:
                    log.error("Erreur requête point", extra={
                        "lat": float(lat), "lon": float(lon),
                        "error": str(e),
                    })
                    raise

                if timestamps is None:
                    timestamps = data[f"wind_speed_{PRESSURE_LEVELS[0]}hPa"].index

                for i_lev, level in enumerate(PRESSURE_LEVELS):
                    u, v = _extract_uv_for_level(data, level)
                    arr_u[:len(u), i_lev, i_lat, i_lon]  = u
                    arr_v[:len(v), i_lev, i_lat, i_lon]  = v
                    arr_z[:, i_lev, i_lat, i_lon] = (
                        data[f"geopotential_height_{level}hPa"].values[:n_times_chunk]
                        * 9.80665  # m → m²/s² (géopotentiel)
                    )
                    arr_t[:, i_lev, i_lat, i_lon] = (
                        data[f"temperature_{level}hPa"].values[:n_times_chunk]
                    )
                    arr_rh[:, i_lev, i_lat, i_lon] = (
                        data[f"relative_humidity_{level}hPa"].values[:n_times_chunk]
                    )

                # Surface
                spd_10m = data["wind_speed_10m"].values[:n_times_chunk]
                dir_10m = data["wind_direction_10m"].values[:n_times_chunk]
                u10, v10 = wind_speed_direction_to_uv(spd_10m, dir_10m)
                arr_u10[:len(u10), i_lat, i_lon] = u10
                arr_v10[:len(v10), i_lat, i_lon] = v10
                arr_t2m[:, i_lat, i_lon] = data["temperature_2m"].values[:n_times_chunk]

                # Délai pour éviter le rate-limiting
                time.sleep(0.1)

        # Conversion RH → q approximée (Tetens, valide T > 0°C)
        # Nota: les niveaux de pression sont en hPa
        p_broadcast = np.array(PRESSURE_LEVELS, dtype=np.float32)[
            np.newaxis, :, np.newaxis, np.newaxis
        ]  # [1, n_levels, 1, 1]
        arr_q = relative_humidity_to_specific_humidity(
            rh=arr_rh,
            t_kelvin=arr_t,
            p_hpa=np.broadcast_to(p_broadcast, arr_rh.shape),
        )

        # Écriture dans le store Zarr
        output_path = Path(output)
        if not output_path.exists():
            # Créer le store lors du premier chunk
            store = create_empty_store(
                path=output_path,
                n_times=_count_total_times(start, end, res_h),
                levels_hpa=PRESSURE_LEVELS,
                lats=lats.tolist(),
                lons=lons.tolist(),
                source=f"ifs_{model}",
                site=site,
                time_step_hours=res_h,
                overwrite=False,
            )
            store.attrs["model"]     = model
            store.attrs["ifs_note"]  = (
                "u/v reconstruits depuis wind_speed+wind_direction (convention météo). "
                "q approximé depuis RH via Tetens (valide T > 0°C)."
            )
        else:
            store = open_store(output_path, mode="r+")

        # Trouver l'offset temporel pour ce chunk
        time_offset = _time_offset_for_chunk(start, chunk_start, res_h)
        n_t = arr_u.shape[0]
        ts_values = timestamps.values[:n_t].astype("datetime64[ns]").astype(np.int64)

        store["coords/time"][time_offset:time_offset + n_t] = ts_values
        store["pressure/u"][time_offset:time_offset + n_t]  = arr_u
        store["pressure/v"][time_offset:time_offset + n_t]  = arr_v
        store["pressure/z"][time_offset:time_offset + n_t]  = arr_z
        store["pressure/t"][time_offset:time_offset + n_t]  = arr_t
        store["pressure/q"][time_offset:time_offset + n_t]  = arr_q
        store["surface/u10"][time_offset:time_offset + n_t] = arr_u10
        store["surface/v10"][time_offset:time_offset + n_t] = arr_v10
        store["surface/t2m"][time_offset:time_offset + n_t] = arr_t2m

        cp.mark_done(chunk_key, extra_meta={
            "n_times": n_t, "time_offset": time_offset,
            "model": model, "chunk_start": chunk_start, "chunk_end": chunk_end,
        })

        log.info("Chunk traité avec succès", extra={
            "key": chunk_key, "n_times": n_t,
        })

    log.info("Ingestion IFS terminée", extra={
        "model": model, "output": str(output),
        "n_chunks": len(chunks),
    })


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _iter_date_chunks(start: str, end: str, chunk_days: int):
    """Découpe [start, end] en chunks de chunk_days jours."""
    current = pd.Timestamp(start)
    end_ts  = pd.Timestamp(end)
    while current <= end_ts:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def _count_total_times(start: str, end: str, res_h: int) -> int:
    """Nombre total de pas de temps entre start et end."""
    delta = pd.Timestamp(end) - pd.Timestamp(start)
    return (delta.days + 1) * 24 // res_h


def _time_offset_for_chunk(global_start: str, chunk_start: str, res_h: int) -> int:
    """Index du premier pas de temps d'un chunk dans le store global."""
    delta = pd.Timestamp(chunk_start) - pd.Timestamp(global_start)
    return delta.days * 24 // res_h


def _print_sample_request(lat, lon, model, start, end):
    """Affiche un exemple de requête API pour debug."""
    vars_ = _pressure_vars(PRESSURE_LEVELS)[:4]
    log.info("Exemple de requête API", extra={
        "url": API_URL,
        "params": {
            "latitude": lat, "longitude": lon,
            "models": model,
            "hourly": vars_ + ["..."],
            "start_date": start, "end_date": end,
        },
    })


if __name__ == "__main__":
    main()
