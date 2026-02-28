"""
ingest_era5_hourly.py — Ingestion ERA5 horaire pour l'entraînement du module 1.

Télécharge les données ERA5 réanalyse à résolution horaire (24 h/jour) sur les
niveaux de pression, pour servir de vérité terrain au module 1 (6h → 1h).

Usage :
    python ingest_era5_hourly.py --site perdigao --start 2016-01 --end 2016-12 \\
                                 --output ../../data/raw/era5_hourly_perdigao.zarr

Différences avec ingest_era5.py (6h) :
  - Tous les 24 créneaux horaires téléchargés (HOURS_1H)
  - Uniquement les variables de pression (pas de surface — non utilisées par M1)
  - time_step_hours=1 dans le store Zarr

Prérequis : identiques à ingest_era5.py (compte Copernicus CDS, ~/.cdsapirc)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import create_empty_store, sha256_file, open_store
from utils.checkpointing import Checkpointer

log = get_logger("ingest_era5_hourly")

# ── Constantes ────────────────────────────────────────────────────────────────

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]

PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
]

# Toutes les heures de la journée (24h)
HOURS_1H = [f"{h:02d}:00" for h in range(24)]


# ── Fonctions CDS avec retry ──────────────────────────────────────────────────

def _build_cds_client():
    import cdsapi
    api_key = os.environ.get("CDS_API_KEY")
    api_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
    if api_key:
        return cdsapi.Client(url=api_url, key=api_key, quiet=True)
    return cdsapi.Client(quiet=True)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=60, min=60, max=240),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _download_cds(client, dataset: str, request: dict, target: str) -> None:
    client.retrieve(dataset, request, target)


# ── Génération des mois ───────────────────────────────────────────────────────

def _iter_months(start: str, end: str):
    y0, m0 = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


# ── Conversion NetCDF → arrays numpy ─────────────────────────────────────────

def _nc_to_arrays_pressure(nc_path: str) -> dict:
    ds = xr.open_dataset(nc_path)

    # Tri N→S (latitude décroissante)
    if ds.latitude[0] < ds.latitude[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Tri pression décroissante (1000→200 hPa)
    if "level" in ds.dims and ds.level[0] < ds.level[-1]:
        ds = ds.isel(level=slice(None, None, -1))

    result = {
        "times":  ds.time.values,
        "levels": ds.level.values.astype(np.float32),
        "lats":   ds.latitude.values.astype(np.float32),
        "lons":   ds.longitude.values.astype(np.float32),
        "u": ds["u"].values.astype(np.float32),
        "v": ds["v"].values.astype(np.float32),
        "z": ds["z"].values.astype(np.float32),
        "t": ds["t"].values.astype(np.float32),
        "q": ds["q"].values.astype(np.float32),
    }
    ds.close()
    return result


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--site",           required=True, help="Identifiant du site (ex: perdigao)")
@click.option("--start",          required=True, help="Début de période (YYYY-MM)")
@click.option("--end",            required=True, help="Fin de période (YYYY-MM, inclusif)")
@click.option("--output",         required=True, help="Chemin du store Zarr de sortie")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Répertoire des configurations de sites")
@click.option("--checkpoint-dir", default=None,
              help="Répertoire des sentinelles (défaut: output/.checkpoints)")
@click.option("--dry-run",        is_flag=True, help="Affiche les requêtes sans télécharger")
def main(site, start, end, output, config_dir, checkpoint_dir, dry_run):
    """
    Télécharge ERA5 horaire (niveaux de pression) pour l'entraînement du module 1.

    Stratégie : téléchargement mensuel avec checkpointing SHA256 (même pattern
    que ingest_era5.py). Un mois à 1h = 24 × ~30 jours = ~720 pas de temps,
    bien en dessous de la limite CDS (~1000 éléments/requête).
    """
    log.info("Démarrage ingestion ERA5 horaire", extra={
        "site": site, "start": start, "end": end, "output": output,
    })

    # ── Configuration du site ─────────────────────────────────────────────────
    config_path = Path(config_dir) / f"{site}.yaml"
    if not config_path.exists():
        log.error("Configuration de site introuvable", extra={"path": str(config_path)})
        sys.exit(1)

    with open(config_path) as f:
        site_cfg = yaml.safe_load(f)

    domain = site_cfg["era5_domain"]
    north, west, south, east = domain["north"], domain["west"], domain["south"], domain["east"]
    step = domain["grid_spacing_deg"]

    lats = np.arange(north, south - step / 2, -step, dtype=np.float32)
    lons = np.arange(west,  east  + step / 2,  step, dtype=np.float32)

    log.info("Grille ERA5", extra={
        "n_lat": len(lats), "n_lon": len(lons),
        "lats": f"{lats[0]:.2f}→{lats[-1]:.2f}",
        "lons": f"{lons[0]:.2f}→{lons[-1]:.2f}",
    })

    # ── Checkpointer ──────────────────────────────────────────────────────────
    cp_dir = checkpoint_dir or str(Path(output) / ".checkpoints_hourly")
    cp = Checkpointer(cp_dir)

    # ── Calcul du nombre total de pas de temps ────────────────────────────────
    months = list(_iter_months(start, end))
    n_times_total = sum(24 * _days_in_month(y, m) for y, m in months)

    log.info("Plan de téléchargement", extra={
        "n_months": len(months), "n_times_total": n_times_total,
        "hours_per_day": 24,
    })

    if dry_run:
        log.info("Mode dry-run — aucun téléchargement effectué")
        return

    # ── Initialisation du store Zarr ──────────────────────────────────────────
    output_path = Path(output)
    if not output_path.exists():
        store = create_empty_store(
            path=output_path,
            n_times=n_times_total,
            levels_hpa=PRESSURE_LEVELS,
            lats=lats.tolist(),
            lons=lons.tolist(),
            source="era5_hourly",
            site=site,
            time_step_hours=1,
            overwrite=False,
        )
        log.info("Store Zarr créé", extra={"path": str(output_path)})
    else:
        store = open_store(output_path, mode="r+")
        log.info("Store Zarr existant — mode append", extra={"path": str(output_path)})

    # ── Client CDS ────────────────────────────────────────────────────────────
    cds = _build_cds_client()
    time_offset = 0

    # ── Boucle mensuelle ──────────────────────────────────────────────────────
    for year, month in months:
        month_key = f"era5h_{year}_{month:02d}"
        n_days = _days_in_month(year, month)
        n_times_month = 24 * n_days

        if cp.is_done(month_key):
            log.info("Mois déjà traité", extra={"key": month_key, "n_times": n_times_month})
            time_offset += n_times_month
            continue

        log.info("Téléchargement", extra={
            "key": month_key, "n_days": n_days, "n_times": n_times_month,
        })

        with tempfile.TemporaryDirectory(prefix="era5h_") as tmpdir:
            tmpdir = Path(tmpdir)
            nc_pres = str(tmpdir / "pressure.nc")

            days = [f"{d:02d}" for d in range(1, n_days + 1)]
            request_pres = {
                "product_type": "reanalysis",
                "variable":     PRESSURE_VARIABLES,
                "pressure_level": [str(lev) for lev in PRESSURE_LEVELS],
                "year":   str(year),
                "month":  f"{month:02d}",
                "day":    days,
                "time":   HOURS_1H,
                "area":   [north, west, south, east],
                "format": "netcdf",
            }

            log.info("Requête CDS pression horaire", extra={"month": month_key})

            try:
                _download_cds(
                    cds,
                    "reanalysis-era5-pressure-levels",
                    request_pres,
                    nc_pres,
                )
            except Exception as e:
                log.error("Échec téléchargement", extra={"key": month_key, "error": str(e)})
                raise

            sha_pres = sha256_file(nc_pres)
            log.info("Téléchargement terminé", extra={
                "key": month_key, "sha256": sha_pres[:16] + "...",
                "size_mb": round(Path(nc_pres).stat().st_size / 1e6, 1),
            })

            pres_data = _nc_to_arrays_pressure(nc_pres)

            assert len(pres_data["times"]) == n_times_month, (
                f"Attendu {n_times_month} pas, obtenu {len(pres_data['times'])}"
            )

            for i, ts in enumerate(pres_data["times"]):
                idx = time_offset + i
                store["coords/time"][idx] = ts.astype("datetime64[ns]").astype(np.int64)
                store["pressure/u"][idx]  = pres_data["u"][i]
                store["pressure/v"][idx]  = pres_data["v"][i]
                store["pressure/z"][idx]  = pres_data["z"][i]
                store["pressure/t"][idx]  = pres_data["t"][i]
                store["pressure/q"][idx]  = pres_data["q"][i]
                # surface non utilisée par le module 1 — laissée à zéro

            cp.mark_done(
                month_key,
                sha256=sha_pres,
                extra_meta={
                    "n_times":     n_times_month,
                    "time_offset": time_offset,
                    "year": year, "month": month,
                },
            )
            log.info("Mois traité avec succès", extra={
                "key": month_key, "n_times": n_times_month,
            })

        time_offset += n_times_month

    log.info("Ingestion ERA5 horaire terminée", extra={
        "n_months": len(months), "n_times_total": time_offset, "output": str(output_path),
    })


if __name__ == "__main__":
    main()
