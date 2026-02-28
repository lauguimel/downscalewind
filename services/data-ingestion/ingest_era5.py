"""
ingest_era5.py — Ingestion ERA5 via CDS API pour DownscaleWind.

Télécharge les données ERA5 réanalyse (niveaux de pression + surface) sur un
domaine centré sur un site de référence (défini dans configs/sites/*.yaml).

Usage :
    python ingest_era5.py --site perdigao --start 2016-01 --end 2017-12 \\
                          --output ../../data/raw/era5_perdigao.zarr

Prérequis :
    - Compte Copernicus CDS et clé API dans ~/.cdsapirc
    - Ou variables d'env CDS_API_KEY et CDS_API_URL

Stratégie :
    Téléchargement mensuel (évite la limite CDS de ~1000 éléments par requête).
    Chaque mois est checkpointé (sentinel SHA256). Reprise automatique après
    interruption. Les fichiers NetCDF temporaires sont supprimés après conversion.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
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

# Chemin vers le package partagé (si pas installé en editable)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import (
    create_empty_store,
    sha256_file,
    open_store,
    VARIABLE_META,
)
from utils.checkpointing import Checkpointer

log = get_logger("ingest_era5")

# ── Constantes ────────────────────────────────────────────────────────────────

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]

PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
]

SURFACE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

# Mapping noms CDS → noms courts DownscaleWind
CDS_TO_SHORT = {
    "u_component_of_wind":      "u",
    "v_component_of_wind":      "v",
    "geopotential":             "z",
    "temperature":              "t",
    "specific_humidity":        "q",
    "10m_u_component_of_wind":  "u10",
    "10m_v_component_of_wind":  "v10",
    "2m_temperature":           "t2m",
}

# Heures disponibles ERA5 à 6h
HOURS_6H = ["00:00", "06:00", "12:00", "18:00"]


# ── Fonctions CDS avec retry ──────────────────────────────────────────────────

def _build_cds_client():
    """Initialise le client CDS API avec support des variables d'env."""
    import cdsapi

    # Support des variables d'env (pour Docker)
    api_key = os.environ.get("CDS_API_KEY")
    api_url = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")

    if api_key:
        return cdsapi.Client(url=api_url, key=api_key, quiet=True)
    # Sinon, utilise ~/.cdsapirc
    return cdsapi.Client(quiet=True)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=60, min=60, max=240),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _download_cds(
    client,
    dataset: str,
    request: dict,
    target: str,
) -> None:
    """Télécharge depuis CDS avec retry exponentiel (60s, 120s, 240s)."""
    client.retrieve(dataset, request, target)


# ── Génération des mois ───────────────────────────────────────────────────────

def _iter_months(start: str, end: str):
    """Génère tous les (année, mois) entre start et end inclus (format YYYY-MM)."""
    y0, m0 = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ── Conversion NetCDF → arrays numpy ─────────────────────────────────────────

def _nc_to_arrays_pressure(nc_path: str) -> dict[str, np.ndarray]:
    """
    Charge un fichier NetCDF ERA5 (niveaux de pression) et retourne les arrays.

    Returns:
        dict avec clés : 'u', 'v', 'z', 't', 'q', 'times', 'levels', 'lats', 'lons'
    """
    ds = xr.open_dataset(nc_path)

    # Trier par latitude décroissante (ERA5 standard : 90→-90)
    if ds.latitude[0] < ds.latitude[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # New CDS API (v2 beta) uses "pressure_level" instead of "level"
    level_dim = "level" if "level" in ds.dims else "pressure_level"

    # Trier par pression décroissante (1000→200 hPa)
    if level_dim in ds.dims and ds[level_dim][0] < ds[level_dim][-1]:
        ds = ds.isel({level_dim: slice(None, None, -1)})

    # New CDS API (v2 beta) uses "valid_time" instead of "time"
    time_coord = "time" if "time" in ds.coords else "valid_time"

    result = {
        "times":  ds[time_coord].values,
        "levels": ds[level_dim].values.astype(np.float32),
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


def _nc_to_arrays_surface(nc_path: str) -> dict[str, np.ndarray]:
    """Charge les variables de surface depuis un fichier NetCDF ERA5."""
    ds = xr.open_dataset(nc_path)

    if ds.latitude[0] < ds.latitude[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # New CDS API (v2 beta) uses "valid_time" instead of "time"
    time_coord = "time" if "time" in ds.coords else "valid_time"

    result = {
        "times": ds[time_coord].values,
        "lats":  ds.latitude.values.astype(np.float32),
        "lons":  ds.longitude.values.astype(np.float32),
        "u10": ds["u10"].values.astype(np.float32),
        "v10": ds["v10"].values.astype(np.float32),
        "t2m": ds["t2m"].values.astype(np.float32),
    }
    ds.close()
    return result


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--site",   required=True, help="Identifiant du site (ex: perdigao)")
@click.option("--start",  required=True, help="Début de période (YYYY-MM)")
@click.option("--end",    required=True, help="Fin de période (YYYY-MM, inclusif)")
@click.option("--output", required=True, help="Chemin du store Zarr de sortie")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Répertoire des configurations de sites")
@click.option("--checkpoint-dir", default=None,
              help="Répertoire des sentinelles (défaut: output/.checkpoints)")
@click.option("--dry-run", is_flag=True,
              help="Affiche les requêtes sans télécharger")
def main(site, start, end, output, config_dir, checkpoint_dir, dry_run):
    """
    Télécharge ERA5 réanalyse pour un site et une période donnés.

    Téléchargement mensuel avec checkpointing SHA256. Les données sont stockées
    dans un store Zarr unifié avec métadonnées CF-conventions.
    """
    log.info("Démarrage ingestion ERA5", extra={
        "site": site, "start": start, "end": end, "output": output,
        "dry_run": dry_run,
    })

    # ── Chargement de la configuration du site ────────────────────────────────
    config_path = Path(config_dir) / f"{site}.yaml"
    if not config_path.exists():
        log.error("Configuration de site introuvable", extra={"path": str(config_path)})
        sys.exit(1)

    with open(config_path) as f:
        site_cfg = yaml.safe_load(f)

    domain = site_cfg["era5_domain"]
    north, west, south, east = domain["north"], domain["west"], domain["south"], domain["east"]

    log.info("Configuration chargée", extra={
        "site": site, "domain": f"N{north}/W{-west}/S{south}/E{east}",
    })

    # ── Grille ERA5 ───────────────────────────────────────────────────────────
    step = domain["grid_spacing_deg"]
    lats = np.arange(north, south - step/2, -step, dtype=np.float32)
    lons = np.arange(west,  east  + step/2,  step, dtype=np.float32)
    n_lat, n_lon = len(lats), len(lons)

    log.info("Grille ERA5", extra={
        "n_lat": n_lat, "n_lon": n_lon,
        "lats": f"{lats[0]:.2f}→{lats[-1]:.2f}",
        "lons": f"{lons[0]:.2f}→{lons[-1]:.2f}",
    })

    # ── Checkpointer ──────────────────────────────────────────────────────────
    cp_dir = checkpoint_dir or str(Path(output) / ".checkpoints")
    cp = Checkpointer(cp_dir)

    # ── Calcul du nombre total de pas de temps ────────────────────────────────
    months = list(_iter_months(start, end))
    n_months = len(months)
    # ERA5 à 6h → 4 pas/jour, ~30 jours/mois → ~120 pas/mois
    n_times_total = sum(
        4 * _days_in_month(y, m) for y, m in months
    )

    log.info("Plan de téléchargement", extra={
        "n_months": n_months,
        "n_times_total": n_times_total,
        "pressure_levels": PRESSURE_LEVELS,
    })

    if dry_run:
        log.info("Mode dry-run — aucun téléchargement effectué")
        return

    # ── Initialisation du store Zarr ──────────────────────────────────────────
    output_path = Path(output)
    if not output_path.exists():
        log.info("Création du store Zarr", extra={"path": str(output_path)})
        store = create_empty_store(
            path=output_path,
            n_times=n_times_total,
            levels_hpa=PRESSURE_LEVELS,
            lats=lats.tolist(),
            lons=lons.tolist(),
            source="era5",
            site=site,
            time_step_hours=6,
            overwrite=False,
        )
    else:
        import shutil

        def _store_is_valid(path: Path) -> bool:
            """Vérifie que le store a bien les arrays pressure/u et coords/time."""
            try:
                s = open_store(path, mode="r")
                return "pressure/u" in s and "coords/time" in s
            except Exception:
                return False

        if not _store_is_valid(output_path):
            log.warning("Store incomplet ou corrompu — recréation", extra={"path": str(output_path)})
            shutil.rmtree(output_path)
            cp.dir.mkdir(parents=True, exist_ok=True)  # rmtree a effacé .checkpoints/
            store = create_empty_store(
                path=output_path,
                n_times=n_times_total,
                levels_hpa=PRESSURE_LEVELS,
                lats=lats.tolist(),
                lons=lons.tolist(),
                source="era5",
                site=site,
                time_step_hours=6,
                overwrite=True,
            )
        else:
            try:
                log.info("Store Zarr existant — mode append", extra={"path": str(output_path)})
                store = open_store(output_path, mode="r+")
            except Exception:
                log.warning("Store inaccessible en r+ — recréation", extra={"path": str(output_path)})
                shutil.rmtree(output_path)
                cp.dir.mkdir(parents=True, exist_ok=True)  # rmtree a effacé .checkpoints/
                store = create_empty_store(
                    path=output_path,
                    n_times=n_times_total,
                    levels_hpa=PRESSURE_LEVELS,
                    lats=lats.tolist(),
                    lons=lons.tolist(),
                    source="era5",
                    site=site,
                    time_step_hours=6,
                    overwrite=True,
                )

    # ── Client CDS ────────────────────────────────────────────────────────────
    cds = _build_cds_client()
    time_offset = 0  # index courant dans le store

    # ── Boucle mensuelle ──────────────────────────────────────────────────────
    for year, month in months:
        month_key = f"era5_{year}_{month:02d}"
        n_days = _days_in_month(year, month)
        n_times_month = 4 * n_days

        if cp.is_done(month_key):
            log.info("Mois déjà traité, passage au suivant", extra={
                "key": month_key, "n_times": n_times_month,
            })
            time_offset += n_times_month
            continue

        log.info("Début téléchargement", extra={
            "key": month_key, "n_days": n_days, "n_times": n_times_month,
        })

        with tempfile.TemporaryDirectory(prefix="era5_") as tmpdir:
            tmpdir = Path(tmpdir)
            nc_pres = str(tmpdir / "pressure.nc")
            nc_surf = str(tmpdir / "surface.nc")

            # ── Requête niveaux de pression ───────────────────────────────────
            days = [f"{d:02d}" for d in range(1, n_days + 1)]
            request_pres = {
                "product_type": "reanalysis",
                "variable":     PRESSURE_VARIABLES,
                "pressure_level": [str(l) for l in PRESSURE_LEVELS],
                "year":   str(year),
                "month":  f"{month:02d}",
                "day":    days,
                "time":   HOURS_6H,
                "area":   [north, west, south, east],  # N/W/S/E
                "format": "netcdf",
            }

            log.info("Requête CDS (niveaux de pression)", extra={
                "dataset": "reanalysis-era5-pressure-levels",
                "month": month_key,
            })

            try:
                _download_cds(
                    cds,
                    "reanalysis-era5-pressure-levels",
                    request_pres,
                    nc_pres,
                )
            except Exception as e:
                log.error("Échec téléchargement pression", extra={
                    "key": month_key, "error": str(e),
                })
                raise

            sha_pres = sha256_file(nc_pres)
            log.info("Fichier pression téléchargé", extra={
                "key": month_key, "sha256": sha_pres[:16] + "...",
                "size_mb": round(Path(nc_pres).stat().st_size / 1e6, 1),
            })

            # ── Requête surface ───────────────────────────────────────────────
            request_surf = {
                "product_type": "reanalysis",
                "variable":     SURFACE_VARIABLES,
                "year":  str(year),
                "month": f"{month:02d}",
                "day":   days,
                "time":  HOURS_6H,
                "area":  [north, west, south, east],
                "format": "netcdf",
            }

            log.info("Requête CDS (surface)", extra={
                "dataset": "reanalysis-era5-single-levels",
                "month": month_key,
            })

            try:
                _download_cds(
                    cds,
                    "reanalysis-era5-single-levels",
                    request_surf,
                    nc_surf,
                )
            except Exception as e:
                log.error("Échec téléchargement surface", extra={
                    "key": month_key, "error": str(e),
                })
                raise

            sha_surf = sha256_file(nc_surf)

            # ── Conversion et écriture Zarr ────────────────────────────────────
            log.info("Conversion NetCDF → Zarr", extra={"key": month_key})

            pres_data = _nc_to_arrays_pressure(nc_pres)
            surf_data = _nc_to_arrays_surface(nc_surf)

            # Vérification cohérence temporelle
            assert len(pres_data["times"]) == n_times_month, (
                f"Attendu {n_times_month} pas, obtenu {len(pres_data['times'])}"
            )

            for i, ts in enumerate(pres_data["times"]):
                idx = time_offset + i
                store["coords/time"][idx] = ts.astype("datetime64[ns]").astype(np.int64)
                store["pressure/u"][idx]   = pres_data["u"][i]
                store["pressure/v"][idx]   = pres_data["v"][i]
                store["pressure/z"][idx]   = pres_data["z"][i]
                store["pressure/t"][idx]   = pres_data["t"][i]
                store["pressure/q"][idx]   = pres_data["q"][i]
                store["surface/u10"][idx]  = surf_data["u10"][i]
                store["surface/v10"][idx]  = surf_data["v10"][i]
                store["surface/t2m"][idx]  = surf_data["t2m"][i]

            # ── Checkpoint ────────────────────────────────────────────────────
            cp.mark_done(
                month_key,
                sha256=sha_pres,
                extra_meta={
                    "sha256_surface": sha_surf,
                    "n_times":        n_times_month,
                    "time_offset":    time_offset,
                    "year": year, "month": month,
                },
            )

            log.info("Mois traité avec succès", extra={
                "key": month_key, "n_times": n_times_month,
            })

        time_offset += n_times_month

    log.info("Ingestion ERA5 terminée", extra={
        "n_months": n_months,
        "n_times_total": time_offset,
        "output": str(output_path),
    })


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _days_in_month(year: int, month: int) -> int:
    """Retourne le nombre de jours dans un mois donné."""
    import calendar
    return calendar.monthrange(year, month)[1]


if __name__ == "__main__":
    main()
