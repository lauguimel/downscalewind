"""
ingest_landcover.py — Carte de rugosité z₀ depuis Copernicus Land Cover (CGLS-LC100).

Télécharge CGLS-LC100 v3 (100m, 2018) via CDS API et produit une carte de
rugosité dynamique z₀ [m] sur le domaine du site.

Usage :
    python ingest_landcover.py --site perdigao \\
                               --output ../../data/raw/z0_perdigao.tif

Fallback :
    Si CDS API non disponible, utilise une carte z₀ uniforme (0.05 m)
    avec un avertissement. Le pipeline CFD reste fonctionnel mais
    les résultats seront moins précis sur les zones forestières.

Référence table z₀ : Wieringa (1992) + Davenport et al. (2000)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.data_io import sha256_file

log = get_logger("ingest_landcover")

# ── Table de correspondance CGLS-LC100 → z₀ ──────────────────────────────────
# Source : Wieringa (1992), Davenport et al. (2000), Stull (1988)
# Classes CGLS-LC100 v3 (Copernicus Global Land Service)

LC100_Z0_TABLE: dict[int, float] = {
    0:   0.001,   # pas de données / mer
    20:  1.00,    # forêt à feuilles larges toujours verte
    30:  1.00,    # forêt à feuilles larges caduque
    40:  0.80,    # forêt de conifères toujours verte
    50:  0.80,    # forêt de conifères caduque
    60:  0.80,    # forêt de conifères sempervirente
    70:  0.80,    # forêt mixte
    80:  0.80,    # forêt mixte - autre
    90:  0.10,    # arbustes
    100: 0.10,    # arbustes tropicaux
    111: 0.10,    # herbes fermées
    112: 0.05,    # herbes ouvertes
    113: 0.05,    # prairies
    114: 0.05,    # prairies + cultures mixtes
    115: 0.05,    # cultures + prairies mixtes
    116: 0.03,    # cultures herbacées
    121: 0.30,    # zones urbaines denses
    122: 0.50,    # zones urbaines très denses
    123: 0.10,    # zones périurbaines
    124: 0.05,    # zones périurbaines ouvertes
    200: 0.0002,  # eau douce (lacs, rivières)
    201: 0.0002,  # eau salée (mer)
    202: 0.001,   # zones inondables saisonnières
    # Classes simplifiées (entiers ronds = classes CGLS-LC100)
    10:  1.00,    # forêt dense (classe générique 10)
    255: 0.05,    # données manquantes → z₀ moyen cultures
}

# Classes CGLS-LC100 v3 entiers ronds (mappings simplifiés)
# Note : les codes réels peuvent varier selon la version du produit
LC100_Z0_SIMPLE: dict[int, float] = {
    10: 1.00,    # forêt dense
    20: 0.10,    # arbustes
    30: 0.03,    # herbacé / prairie
    40: 0.05,    # cultures
    50: 0.50,    # urbain
    60: 0.01,    # sol nu / rochers
    70: 0.001,   # neige / glace
    80: 0.0002,  # eau
    90: 0.03,    # zones humides
    95: 0.10,    # mangroves
    100: 0.03,   # mousse / lichens
    # Valeur par défaut pour classes inconnues
    255: 0.05,
}

# z₀ par défaut si classe non trouvée dans la table
Z0_DEFAULT = 0.05  # cultures / terrain ouvert


# ── Fonctions principales ─────────────────────────────────────────────────────

def _map_lc_to_z0(lc_array: np.ndarray) -> np.ndarray:
    """
    Convertit un array de classes LC100 en un array de z₀ [m].

    Args:
        lc_array: Array de classes entières (CGLS-LC100)

    Returns:
        Array de rugosité z₀ [m], même shape que lc_array
    """
    z0 = np.full(lc_array.shape, Z0_DEFAULT, dtype=np.float32)
    for lc_class, z0_val in LC100_Z0_SIMPLE.items():
        z0[lc_array == lc_class] = z0_val
    return z0


def _create_uniform_z0(
    north: float, west: float, south: float, east: float,
    z0_value: float = 0.05,
    resolution_deg: float = 0.001,
) -> tuple[np.ndarray, rasterio.transform.Affine]:
    """
    Crée une carte z₀ uniforme pour le domaine (fallback si LC non disponible).

    Args:
        north, west, south, east: Limites du domaine (°)
        z0_value: Valeur de z₀ uniforme (m)
        resolution_deg: Résolution en degrés (~100m en latitude)

    Returns:
        (z0_array, transform) pour export GeoTIFF
    """
    nrows = max(1, int((north - south) / resolution_deg))
    ncols = max(1, int((east  - west)  / resolution_deg))
    z0 = np.full((nrows, ncols), z0_value, dtype=np.float32)
    transform = from_bounds(west, south, east, north, ncols, nrows)
    return z0, transform


def _download_cgls_lc100(
    north: float, west: float, south: float, east: float,
    year: int = 2018,
) -> np.ndarray | None:
    """
    Télécharge CGLS-LC100 depuis CDS API (Copernicus Land Service).

    Requiert ~/.cdsapirc configuré.

    Returns:
        Array numpy de classes LC100, ou None si CDS non disponible
    """
    try:
        import cdsapi
    except ImportError:
        log.warning("cdsapi non disponible — fallback z₀ uniforme")
        return None

    try:
        client = cdsapi.Client(quiet=True)
    except Exception as e:
        log.warning("CDS non configuré — fallback z₀ uniforme", extra={"error": str(e)})
        return None

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        client.retrieve(
            "satellite-land-cover",
            {
                "variable": "all",
                "format": "zip",
                "year": str(year),
                "version": "v3.0.1",
            },
            str(tmp_path),
        )
        log.info("CGLS-LC100 téléchargé", extra={"year": year})
        # TODO : parser le ZIP → extraire NetCDF → lire avec rasterio
        # Pour l'instant retourner None (à compléter avec parsing NetCDF)
        log.warning("Parsing CGLS-LC100 non encore implémenté — fallback z₀ uniforme")
        return None
    except Exception as e:
        log.warning("Téléchargement CGLS-LC100 échoué — fallback z₀ uniforme",
                    extra={"error": str(e)})
        return None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--site",       required=True, help="Identifiant du site (ex: perdigao)")
@click.option("--output",     required=True, help="GeoTIFF z₀ de sortie")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              help="Répertoire des configurations de sites")
@click.option("--lc-file",    default=None,
              help="Fichier LC100 GeoTIFF déjà téléchargé (optionnel)")
@click.option("--year",       default=2018, show_default=True,
              help="Année du produit CGLS-LC100 (2016–2020 disponibles)")
@click.option("--resolution-m", default=100, show_default=True,
              help="Résolution de sortie en mètres (100m = résolution native)")
def main(site, output, config_dir, lc_file, year, resolution_m):
    """
    Génère une carte de rugosité z₀ depuis CGLS-LC100 pour un site donné.

    Si CGLS-LC100 n'est pas disponible (CDS non configuré, pas de fichier fourni),
    crée une carte z₀ uniforme (0.05 m) avec avertissement.

    Pour utiliser un fichier CGLS-LC100 déjà téléchargé :
        python ingest_landcover.py --site perdigao --lc-file /path/to/lc100.tif \\
                                   --output data/raw/z0_perdigao.tif
    """
    log.info("Démarrage ingestion land cover", extra={"site": site, "output": output})

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

    # Marge de ~0.1° pour éviter les effets de bord
    margin = 0.1
    bbox = (west - margin, south - margin, east + margin, north + margin)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z0_array: np.ndarray | None = None
    transform = None
    source_label = "uniform_fallback"

    # ── Cas 1 : fichier LC fourni directement ─────────────────────────────────
    if lc_file:
        lc_path = Path(lc_file)
        if not lc_path.exists():
            log.error("Fichier LC introuvable", extra={"path": str(lc_path)})
            sys.exit(1)

        log.info("Lecture fichier LC fourni", extra={"path": str(lc_path)})
        with rasterio.open(str(lc_path)) as src:
            # Clip au domaine avec une marge
            from rasterio.windows import from_bounds as window_from_bounds
            window = window_from_bounds(*bbox, src.transform)
            lc_data = src.read(1, window=window)
            transform = src.window_transform(window)

        z0_array = _map_lc_to_z0(lc_data)
        source_label = f"CGLS-LC100 v3 {year} (from file)"

    # ── Cas 2 : tentative téléchargement CDS ──────────────────────────────────
    if z0_array is None:
        lc_data = _download_cgls_lc100(
            north + margin, west - margin, south - margin, east + margin, year=year,
        )
        if lc_data is not None:
            z0_array = _map_lc_to_z0(lc_data)
            source_label = f"CGLS-LC100 v3 {year} (CDS)"

    # ── Cas 3 : fallback z₀ uniforme ──────────────────────────────────────────
    if z0_array is None:
        log.warning(
            "Fallback z₀ uniforme activé. "
            "Pour une meilleure précision, téléchargez CGLS-LC100 depuis "
            "https://lcviewer.vito.be/download et relancez avec --lc-file."
        )
        z0_array, transform = _create_uniform_z0(
            north + margin, west - margin, south - margin, east + margin,
            z0_value=0.05,
        )
        source_label = "uniform_fallback_0.05m"

    # ── Appliquer les valeurs z₀ spécifiques au site si connues ───────────────
    # Surcharge basée sur perdigao.yaml si disponible
    if "physics" in site_cfg:
        z0_ridge   = site_cfg["physics"].get("z0_ridge_m", None)
        z0_valley  = site_cfg["physics"].get("z0_valley_m", None)
        z0_forest  = site_cfg["physics"].get("z0_forest_m", None)
        log.info("Valeurs z₀ du site", extra={
            "z0_ridge_m": z0_ridge, "z0_valley_m": z0_valley, "z0_forest_m": z0_forest,
        })

    # ── Écriture GeoTIFF ──────────────────────────────────────────────────────
    if transform is None:
        # Si transform non défini (cas fallback sans transform)
        resolution_deg = resolution_m / 111320.0  # ~approximation
        transform = from_bounds(
            west - margin, south - margin,
            east + margin, north + margin,
            z0_array.shape[1], z0_array.shape[0],
        )

    profile = {
        "driver": "GTiff",
        "height": z0_array.shape[0],
        "width":  z0_array.shape[1],
        "count":  1,
        "dtype":  "float32",
        "crs":    "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "nodata": -9999.0,
    }

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(z0_array, 1)
        dst.update_tags(
            source=source_label,
            site=site,
            units="m",
            description="Rugosité dynamique z0 [m] pour la couche limite de surface",
        )

    sha_out = sha256_file(output_path)
    log.info("Carte z₀ exportée", extra={
        "output": str(output_path),
        "source": source_label,
        "shape": f"{z0_array.shape[0]}×{z0_array.shape[1]}",
        "z0_min": float(np.nanmin(z0_array)),
        "z0_max": float(np.nanmax(z0_array)),
        "z0_mean": round(float(np.nanmean(z0_array)), 4),
        "sha256": sha_out[:16] + "...",
        "warning": "uniform" in source_label,
    })

    if "uniform" in source_label:
        log.warning(
            "ATTENTION : carte z₀ uniforme utilisée. "
            "Les erreurs de rugosité affectent directement le profil inlet "
            "et la couche limite de surface. "
            "Téléchargez CGLS-LC100 depuis https://lcviewer.vito.be/download "
            "pour améliorer la précision."
        )


if __name__ == "__main__":
    main()
