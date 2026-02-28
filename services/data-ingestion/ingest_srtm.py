"""
ingest_srtm.py — Ingestion topographie COP-DEM GLO-30 (30m) pour DownscaleWind.

Télécharge les tuiles Copernicus DEM Global-30m (COP-DEM GLO-30) depuis le
registre AWS Open Data (public, sans authentification).

Qualité : COP-DEM GLO-30 est supérieur à SRTM1 pour l'Europe — résolution
identique (1 arc-seconde ≈ 30m) mais moins d'artefacts radar sur les flancs boisés.

Usage :
    python ingest_srtm.py --site perdigao \\
                          --output ../../data/raw/srtm_perdigao_30m.tif

Sortie :
    - GeoTIFF 30m en EPSG:4326 couvrant le domaine du site
    - Métadonnées : source, date de téléchargement, SHA256
    - Log JSON par tuile téléchargée (checkpointing)

Le resampling à la résolution CFD cible est fait dans generate_mesh.py,
pas ici. Ce script sauvegarde à la résolution native (30m).
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import rasterio
import rasterio.merge
import rasterio.mask
from rasterio.transform import from_bounds
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import sha256_file
from utils.checkpointing import Checkpointer

log = get_logger("ingest_srtm")

# ── Constantes ────────────────────────────────────────────────────────────────

# COP-DEM GLO-30 sur AWS Open Data (public, sans auth)
# Format : https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif
COPDEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

# Table de correspondance classe CGLS-LC100 → z₀ (utilisée aussi dans ingest_landcover)
# Définie ici pour référence, utilisée par ingest_landcover.py
LC100_Z0_TABLE: dict[int, float] = {
    10: 1.00,   # forêt dense
    20: 0.10,   # arbustes
    30: 0.03,   # herbacé / prairies
    40: 0.05,   # cultures
    50: 0.50,   # zones urbaines
    60: 0.01,   # sol nu / rochers
    70: 0.001,  # neige / glace
    80: 0.0002, # eau
    90: 0.03,   # zones humides
    95: 0.10,   # mangroves
    100: 0.03,  # mousse / lichens
}


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def _tile_name(lat: int, lon: int) -> str:
    """
    Génère le nom de tuile COP-DEM GLO-30 pour un coin SW à (lat, lon).

    Convention : lat/lon = coin inférieur gauche de la tuile 1°×1°.
    Exemple : lat=39, lon=-8 → "Copernicus_DSM_COG_10_N39_00_W008_00_DEM"

    Args:
        lat: Latitude entière (°N) du coin SW de la tuile
        lon: Longitude entière (°E) du coin SW de la tuile

    Returns:
        Nom de la tuile (sans extension)
    """
    lat_hemi = "N" if lat >= 0 else "S"
    lon_hemi = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{lat_hemi}{lat_str}_00_{lon_hemi}{lon_str}_00_DEM"


def _tile_url(lat: int, lon: int) -> str:
    """Retourne l'URL de téléchargement d'une tuile COP-DEM GLO-30."""
    name = _tile_name(lat, lon)
    return f"{COPDEM_BASE_URL}/{name}/{name}.tif"


def _tiles_for_domain(north: float, west: float, south: float, east: float) -> list[tuple[int, int]]:
    """
    Retourne la liste des tuiles (lat, lon) couvrant le domaine.

    Args:
        north, west, south, east: Limites du domaine en degrés

    Returns:
        Liste de (lat_sw, lon_sw) pour chaque tuile 1°×1° nécessaire
    """
    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append((lat, lon))
    return tiles


def _download_tile(url: str, dest: Path, timeout: int = 120) -> None:
    """
    Télécharge une tuile COP-DEM depuis AWS.

    Args:
        url:  URL de la tuile
        dest: Fichier de destination
        timeout: Timeout HTTP en secondes
    """
    import urllib.request

    log.info("Téléchargement tuile", extra={"url": url, "dest": str(dest)})
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        # Certaines tuiles peuvent être manquantes (océan, pôles)
        raise RuntimeError(f"Téléchargement échoué pour {url}: {e}") from e


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--site",       required=True, help="Identifiant du site (ex: perdigao)")
@click.option("--output",     required=True, help="GeoTIFF de sortie (30m)")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Répertoire des configurations de sites")
@click.option("--checkpoint-dir", default=None,
              help="Répertoire des sentinelles (défaut: output_dir/.checkpoints)")
@click.option("--dry-run",    is_flag=True, help="Affiche les URLs sans télécharger")
def main(site, output, config_dir, checkpoint_dir, dry_run):
    """
    Télécharge le COP-DEM GLO-30 pour le domaine d'un site et exporte en GeoTIFF.

    Source : Copernicus DEM GLO-30 via AWS Open Data (public, sans authentification).
    Résolution : ~30m (1 arc-seconde).
    """
    log.info("Démarrage ingestion DEM", extra={"site": site, "output": output})

    # ── Configuration du site ─────────────────────────────────────────────────
    config_path = Path(config_dir) / f"{site}.yaml"
    if not config_path.exists():
        log.error("Configuration introuvable", extra={"path": str(config_path)})
        sys.exit(1)

    with open(config_path) as f:
        site_cfg = yaml.safe_load(f)

    domain = site_cfg["era5_domain"]
    north = domain["north"]
    west  = domain["west"]
    south = domain["south"]
    east  = domain["east"]

    log.info("Domaine", extra={
        "north": north, "west": west, "south": south, "east": east,
    })

    # ── Identification des tuiles ─────────────────────────────────────────────
    tiles = _tiles_for_domain(north, west, south, east)
    log.info("Tuiles nécessaires", extra={"n_tiles": len(tiles), "tiles": tiles})

    if dry_run:
        for lat, lon in tiles:
            print(f"  {_tile_url(lat, lon)}")
        return

    # ── Checkpointer ─────────────────────────────────────────────────────────
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cp_dir = checkpoint_dir or str(output_path.parent / ".checkpoints_srtm")
    cp = Checkpointer(cp_dir)

    # ── Téléchargement des tuiles ─────────────────────────────────────────────
    tile_paths: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="copdem_") as tmpdir:
        tmpdir = Path(tmpdir)

        for lat, lon in tiles:
            tile_key = f"copdem_{lat}_{lon}"
            tile_file = tmpdir / f"{_tile_name(lat, lon)}.tif"
            url = _tile_url(lat, lon)

            if cp.is_done(tile_key):
                log.info("Tuile déjà téléchargée", extra={"key": tile_key})
                # Re-télécharger dans le tmpdir pour la fusion (fichier temp)
                try:
                    _download_tile(url, tile_file)
                    tile_paths.append(tile_file)
                except RuntimeError:
                    log.warning("Tuile absente (probable zone océanique)", extra={
                        "key": tile_key, "url": url,
                    })
                continue

            try:
                _download_tile(url, tile_file)
                sha = sha256_file(tile_file)
                cp.mark_done(tile_key, sha256=sha, extra_meta={
                    "url": url, "lat": lat, "lon": lon,
                })
                tile_paths.append(tile_file)
                log.info("Tuile téléchargée", extra={
                    "key": tile_key, "sha256": sha[:16] + "...",
                    "size_mb": round(tile_file.stat().st_size / 1e6, 1),
                })
            except RuntimeError as e:
                log.warning("Tuile ignorée", extra={
                    "key": tile_key, "reason": str(e),
                })
                # Peut se produire pour les tuiles entièrement en mer

        if not tile_paths:
            log.error("Aucune tuile téléchargée avec succès")
            sys.exit(1)

        # ── Fusion des tuiles ─────────────────────────────────────────────────
        log.info("Fusion des tuiles", extra={"n_tiles": len(tile_paths)})

        datasets = [rasterio.open(str(p)) for p in tile_paths]
        merged_array, merged_transform = rasterio.merge.merge(datasets)
        profile = datasets[0].profile.copy()

        for ds in datasets:
            ds.close()

        # ── Clip au domaine du site ───────────────────────────────────────────
        # Créer un rectangle de clip légèrement plus large que le domaine ERA5
        # (0.05° de marge pour éviter les effets de bord)
        margin = 0.05
        clip_bbox = [
            west  - margin,
            south - margin,
            east  + margin,
            north + margin,
        ]

        # Calculer les indices pixel correspondant au bbox dans le raster merged
        # merged_transform : (pixel_width, 0, left_edge, 0, pixel_height, top_edge)
        px_w = merged_transform.a   # pixel width (°)
        px_h = merged_transform.e   # pixel height (°, négatif)
        left = merged_transform.c
        top  = merged_transform.f

        col_start = max(0, int((clip_bbox[0] - left) / px_w))
        row_start = max(0, int((top - clip_bbox[3]) / (-px_h)))
        col_end   = min(merged_array.shape[2], int((clip_bbox[2] - left) / px_w) + 1)
        row_end   = min(merged_array.shape[1], int((top - clip_bbox[1]) / (-px_h)) + 1)

        clipped = merged_array[:, row_start:row_end, col_start:col_end]
        new_left  = left + col_start * px_w
        new_top   = top  - row_start * (-px_h)
        new_transform = from_bounds(
            new_left, new_top + px_h * clipped.shape[1],
            new_left + px_w * clipped.shape[2], new_top,
            clipped.shape[2], clipped.shape[1],
        )

        # ── Écriture GeoTIFF ──────────────────────────────────────────────────
        profile.update({
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width":  clipped.shape[2],
            "count":  1,
            "dtype":  "float32",
            "crs":    "EPSG:4326",
            "transform": new_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        })

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(clipped[0].astype(np.float32), 1)
            dst.update_tags(
                source="Copernicus DEM GLO-30 (AWS Open Data)",
                site=site,
                domain=f"N{north}/W{-west}/S{south}/E{east}",
                resolution_m="~30 (1 arc-second)",
                n_tiles=str(len(tile_paths)),
            )

        sha_out = sha256_file(output_path)
        log.info("DEM exporté", extra={
            "output": str(output_path),
            "shape": f"{clipped.shape[1]}×{clipped.shape[2]}",
            "resolution_m": "~30",
            "sha256": sha_out[:16] + "...",
            "size_mb": round(output_path.stat().st_size / 1e6, 1),
        })

    # ── Vérification finale ───────────────────────────────────────────────────
    with rasterio.open(str(output_path)) as src:
        arr = src.read(1)
        valid = arr[arr != src.nodata] if src.nodata else arr.flatten()
        log.info("Statistiques DEM", extra={
            "min_m": float(np.nanmin(valid)),
            "max_m": float(np.nanmax(valid)),
            "mean_m": round(float(np.nanmean(valid)), 1),
            "nodata_fraction": round(float(np.sum(arr == src.nodata) / arr.size), 4)
            if src.nodata else 0.0,
        })


if __name__ == "__main__":
    main()
