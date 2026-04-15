"""
build_sites_complex_terrain.py — Generate site list for complex_terrain_v1 campaign.

Produces ~750 sites across 4 categories, all in complex terrain where CFD adds
value over ERA5 25km. Replaces the 9k campaign which contained 450 offshore/coastal
wind farms of limited scientific value.

Categories:
    D_fire     : 250 Mediterranean fire-prone (FR/ES/PT/IT/GR, climate Csa/Csb)
    E_mountain : 200 high-relief mountains (Alps, Pyrenees, Dolomites, Apennines)
    F_wind_onshore: 150 wind farms in complex terrain (onshore, not flat/coastal)
    C_morpho   : 150 random complex-terrain sampling (stratified slope × aspect)

Outputs:
    sites.csv (for run_matrix builder)
    manifests/sites.yaml (authoritative metadata)

Usage
-----
    python services/module2a-cfd/build_sites_complex_terrain.py \\
        --srtm data/raw/srtm_europe.tif \\
        --worldcover data/raw/worldcover_europe.tif \\
        --koppen data/raw/koppen_geiger.tif \\
        --n-d-fire 250 --n-e-mountain 200 --n-f-wind 150 --n-c-morpho 150 \\
        --out data/campaign/complex_terrain_v1/sites.csv \\
        --manifest data/campaign/complex_terrain_v1/manifests/sites.yaml \\
        --seed 42

If --koppen is missing, climate zone is inferred from lat/lon bbox heuristics.
If --worldcover is missing, land cover filtering is skipped.
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geographic bounding boxes for category sampling
# ─────────────────────────────────────────────────────────────────────────────

# Mediterranean fire-prone areas (D_fire) — bbox [lat_min, lon_min, lat_max, lon_max]
D_FIRE_BBOXES = {
    "FR_Provence":       (43.0,  4.0, 44.5,   7.5),
    "FR_Cevennes":       (44.0,  3.3, 44.7,   4.5),
    "FR_Languedoc":      (42.8,  2.3, 44.0,   3.7),
    "FR_Corse":          (41.4,  8.5, 43.0,   9.6),
    "ES_Catalogne":      (40.5,  0.0, 42.8,   3.2),
    "ES_Aragon_S":       (40.0, -1.5, 41.5,   0.5),
    "ES_ComValenciana":  (38.5, -1.0, 40.3,   0.5),
    "ES_Andalucia_E":    (36.8, -3.5, 38.2,  -1.5),
    "ES_SierraNevada":   (36.9, -3.8, 37.4,  -2.6),
    "PT_Alentejo":       (37.5, -8.8, 39.3,  -7.0),
    "PT_Centro":         (39.3, -8.5, 40.8,  -7.0),
    "PT_SerraEstrela":   (40.2, -7.8, 40.5,  -7.2),
    "IT_Sicilia":        (37.0, 12.5, 38.5,  15.5),
    "IT_Sardegna":       (39.0,  8.3, 41.3,   9.9),
    "IT_Calabria":       (38.0, 15.6, 40.1,  17.2),
    "GR_Peloponnese":    (36.5, 21.1, 38.3,  23.2),
    "GR_CentralGreece":  (38.0, 21.0, 39.5,  24.0),
    "GR_Attica":         (37.7, 22.8, 38.6,  24.5),
    "HR_Dalmatia":       (42.5, 15.5, 45.0,  18.0),
}

# High-relief mountain areas (E_mountain)
E_MOUNTAIN_BBOXES = {
    "FR_AlpesFR":        (44.0,  5.8, 46.5,   7.8),
    "FR_Vercors":        (44.7,  5.1, 45.3,   5.8),
    "FR_Ecrins":         (44.7,  6.0, 45.1,   6.6),
    "IT_AlpesIT":        (45.5,  7.0, 47.0,  11.5),
    "IT_Dolomites":      (46.1, 11.2, 46.9,  12.7),
    "IT_Apennines_N":    (43.7, 10.5, 44.8,  12.2),
    "IT_Apennines_C":    (42.0, 13.0, 43.5,  14.5),
    "CH_Valais":         (45.9,  6.8, 46.5,   8.5),
    "CH_Graubunden":     (46.3,  8.8, 47.1,  10.5),
    "CH_Bernese":        (46.3,  7.3, 46.8,   8.5),
    "AT_Tirol":          (46.8, 10.0, 47.5,  12.5),
    "FR_Pyrenees":       (42.5, -1.5, 43.2,   3.0),
    "ES_Pyrenees":       (42.3, -1.0, 42.9,   2.5),
    "ES_PicosEuropa":    (42.9, -5.2, 43.3,  -4.3),
    "SI_JulianAlps":     (46.2, 13.4, 46.6,  14.2),
    "RO_Carpathians":    (45.0, 23.0, 47.5,  26.0),
    "BG_Rila":           (41.9, 23.2, 42.4,  24.0),
    "SK_Tatras":         (49.0, 19.7, 49.4,  20.5),
}

# Wind farms in complex terrain (F_wind_onshore) — deliberately excludes offshore/coastal plains
F_WIND_BBOXES = {
    "FR_Aveyron":        (43.9,  2.3, 44.6,   3.3),
    "FR_Lozere":         (44.1,  3.1, 44.7,   3.9),
    "FR_Tarn":           (43.5,  2.0, 44.1,   2.8),
    "ES_Galicia":        (42.3, -8.8, 43.7,  -7.0),
    "ES_Cantabria":      (42.7, -4.5, 43.3,  -3.3),
    "ES_CastillaLeon":   (40.5, -5.5, 42.0,  -3.0),
    "PT_NortePT":        (41.2, -8.3, 41.9,  -7.2),
    "IT_Appennini_Wind": (40.5, 14.5, 42.0,  16.0),
    "IT_Basilicata":     (40.2, 15.2, 41.2,  16.8),
    "GR_Thrace":         (40.8, 25.0, 41.5,  26.5),
    "UK_Scotland":       (56.0, -5.0, 58.5,  -3.0),   # Highlands wind
    "IE_Ireland_W":      (53.0, -10.0, 54.5, -8.5),
    "DE_Schwarzwald":    (47.6,  7.9, 48.6,   8.7),
}

# Random complex terrain (C_morpho) — broader Europe, filtered by relief later
C_MORPHO_BBOX_EUROPE = (36.0, -10.0, 55.0, 25.0)  # continental Europe, SRTM-covered


# Köppen-Geiger approximate classification by bbox (heuristic fallback if raster missing)
KOPPEN_BBOX_HEURISTIC = {
    # Mediterranean (Csa/Csb)
    **{k: "Csa" for k in D_FIRE_BBOXES.keys()},
    # Oceanic mountain (Cfb/Dfc)
    **{k: "Cfb" for k in ["UK_Scotland", "IE_Ireland_W", "DE_Schwarzwald"]},
    # Alpine (ET/Dfc)
    **{k: "Dfc" for k in E_MOUNTAIN_BBOXES.keys()},
    # Continental onshore wind zones
    "FR_Aveyron": "Cfb", "FR_Lozere": "Cfb", "FR_Tarn": "Csa",
    "ES_Galicia": "Cfb", "ES_Cantabria": "Cfb", "ES_CastillaLeon": "Csa",
    "PT_NortePT": "Csb", "IT_Appennini_Wind": "Csa", "IT_Basilicata": "Csa",
    "GR_Thrace": "Csa",
}


@dataclass
class Site:
    site_id: str
    group: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    country: str = "XX"
    climate_zone: str = "Cfb"
    std_elev_local_m: float = 0.0
    mean_slope_deg: float = 0.0
    subregion: str = ""

    def to_csv_row(self) -> dict:
        return {
            "site_id": self.site_id,
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "group": self.group,
            "elev_m": round(self.elevation_m, 1),
            "country": self.country,
            "climate_zone": self.climate_zone,
            "std_elev": round(self.std_elev_local_m, 1),
            "slope_deg": round(self.mean_slope_deg, 2),
            "subregion": self.subregion,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SRTM utilities
# ─────────────────────────────────────────────────────────────────────────────

def sample_srtm_stats(
    srtm_path: Path,
    lat: float,
    lon: float,
    radius_m: float = 2500.0,
) -> tuple[float, float, float]:
    """Return (elev_center, std_elev_local, mean_slope_deg) at (lat, lon).

    Samples SRTM in a box of ~radius_m around the point. Returns NaN triple if
    outside raster or too few valid pixels.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        return (np.nan, np.nan, np.nan)

    try:
        with rasterio.open(srtm_path) as src:
            # Convert radius to degrees (approx)
            deg_per_m = 1.0 / 111_320.0
            dlat = radius_m * deg_per_m
            dlon = radius_m * deg_per_m / max(np.cos(np.radians(lat)), 0.1)
            bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
            win = from_bounds(*bbox, transform=src.transform)
            arr = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            arr = np.where(arr < -500, np.nan, arr)  # sentinel cleanup
            if np.isfinite(arr).sum() < 10:
                return (np.nan, np.nan, np.nan)
            elev_center = float(np.nanmean(arr[arr.shape[0] // 2 - 1:arr.shape[0] // 2 + 2,
                                              arr.shape[1] // 2 - 1:arr.shape[1] // 2 + 2]))
            std_elev = float(np.nanstd(arr))
            # Mean slope via central differences
            px_m = 30.0  # SRTM 30m
            gy, gx = np.gradient(arr, px_m, px_m)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            mean_slope = float(np.nanmean(slope))
            return (elev_center, std_elev, mean_slope)
    except Exception as exc:
        logger.debug("SRTM sample failed at (%.3f, %.3f): %s", lat, lon, exc)
        return (np.nan, np.nan, np.nan)


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (lat, lon) center of bbox."""
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def country_from_subregion(subregion: str) -> str:
    """First 2 letters of subregion name are ISO country (by naming convention)."""
    return subregion.split("_")[0].upper()[:2] if subregion else "XX"


# ─────────────────────────────────────────────────────────────────────────────
# Sampling strategies per category
# ─────────────────────────────────────────────────────────────────────────────

def sample_category(
    category: str,
    bboxes: dict[str, tuple[float, float, float, float]],
    n_total: int,
    srtm_path: Optional[Path],
    rng: random.Random,
    min_elev: float = 0.0,
    min_std_elev: float = 0.0,
    min_slope_deg: float = 0.0,
    max_attempts_factor: int = 40,
) -> list[Site]:
    """Sample n_total sites across the given bboxes (uniform per subregion).

    For each candidate, query SRTM and apply terrain filters. Retries up to
    max_attempts_factor × n_total before giving up on a subregion.
    """
    subregions = list(bboxes.keys())
    # Allocate roughly equally across subregions
    per_subregion = max(1, n_total // len(subregions))
    remainder = n_total - per_subregion * len(subregions)
    # Distribute remainder across first 'remainder' subregions
    alloc = {sr: per_subregion + (1 if i < remainder else 0)
             for i, sr in enumerate(subregions)}

    sites: list[Site] = []
    for sr in subregions:
        bbox = bboxes[sr]
        target = alloc[sr]
        attempts = 0
        max_attempts = max_attempts_factor * target
        obtained = 0
        while obtained < target and attempts < max_attempts:
            attempts += 1
            lat = rng.uniform(bbox[0], bbox[2])
            lon = rng.uniform(bbox[1], bbox[3])
            if srtm_path is not None:
                elev, std_elev, slope = sample_srtm_stats(srtm_path, lat, lon)
                if not np.isfinite(elev):
                    continue
                if elev < min_elev:
                    continue
                if std_elev < min_std_elev:
                    continue
                if slope < min_slope_deg:
                    continue
            else:
                # Without SRTM, accept all (use bbox center elev as 0)
                elev, std_elev, slope = (np.nan, np.nan, np.nan)

            site_id = f"ct_{category.lower()}_{len(sites):04d}"
            climate = KOPPEN_BBOX_HEURISTIC.get(sr, "Cfb")
            site = Site(
                site_id=site_id,
                group=category,
                lat=lat, lon=lon,
                elevation_m=elev if np.isfinite(elev) else 0.0,
                std_elev_local_m=std_elev if np.isfinite(std_elev) else 0.0,
                mean_slope_deg=slope if np.isfinite(slope) else 0.0,
                country=country_from_subregion(sr),
                climate_zone=climate,
                subregion=sr,
            )
            sites.append(site)
            obtained += 1
        if obtained < target:
            logger.warning(
                "%s/%s: obtained %d/%d sites (attempts exhausted, relaxed filters ?)",
                category, sr, obtained, target)
    return sites


def sample_c_morpho(
    n_total: int,
    srtm_path: Optional[Path],
    rng: random.Random,
    min_std_elev: float = 50.0,
    min_slope_deg: float = 3.0,
) -> list[Site]:
    """Stratified random sampling over continental Europe with relief filter."""
    bbox = C_MORPHO_BBOX_EUROPE
    sites: list[Site] = []
    attempts = 0
    max_attempts = 60 * n_total
    while len(sites) < n_total and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(bbox[0], bbox[2])
        lon = rng.uniform(bbox[1], bbox[3])
        if srtm_path is not None:
            elev, std_elev, slope = sample_srtm_stats(srtm_path, lat, lon)
            if not np.isfinite(elev):
                continue
            if std_elev < min_std_elev or slope < min_slope_deg:
                continue
        else:
            elev, std_elev, slope = (np.nan, np.nan, np.nan)

        # Heuristic country from lat/lon
        country = "XX"
        if 41.0 <= lat <= 51.5 and -5.5 <= lon <= 10.0:
            country = "FR"
        elif 35.5 <= lat <= 44.0 and -10.0 <= lon <= 3.5:
            country = "ES"
        elif 36.5 <= lat <= 47.5 and 6.5 <= lon <= 19.0:
            country = "IT"
        elif 45.5 <= lat <= 55.0 and 5.5 <= lon <= 15.5:
            country = "DE"
        elif 45.5 <= lat <= 48.0 and 9.0 <= lon <= 17.0:
            country = "AT"
        elif 49.0 <= lat <= 55.5 and -8.5 <= lon <= 2.0:
            country = "UK"
        elif 36.0 <= lat <= 42.0 and 19.0 <= lon <= 28.5:
            country = "GR"
        elif 42.0 <= lat <= 51.5 and 14.0 <= lon <= 24.0:
            country = "PL"

        # Climate heuristic
        if lat < 45.0 and -10.0 <= lon <= 25.0:
            climate = "Csa"  # Med/sub-Med
        elif lat >= 50.0:
            climate = "Cfb"  # oceanic/continental
        else:
            climate = "Cfb"

        site_id = f"ct_c_morpho_{len(sites):04d}"
        sites.append(Site(
            site_id=site_id,
            group="C_morpho",
            lat=lat, lon=lon,
            elevation_m=elev if np.isfinite(elev) else 0.0,
            std_elev_local_m=std_elev if np.isfinite(std_elev) else 0.0,
            mean_slope_deg=slope if np.isfinite(slope) else 0.0,
            country=country,
            climate_zone=climate,
            subregion="europe_random",
        ))
    if len(sites) < n_total:
        logger.warning("C_morpho: obtained %d/%d sites", len(sites), n_total)
    return sites


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(sites: list[Site], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(sites[0].to_csv_row().keys()) if sites else [
        "site_id", "lat", "lon", "group", "elev_m", "country",
        "climate_zone", "std_elev", "slope_deg", "subregion"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sites:
            writer.writerow(s.to_csv_row())
    logger.info("Wrote %d sites to %s", len(sites), out_csv)


def write_manifest(sites: list[Site], out_yaml: Path, campaign: str) -> None:
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "campaign": campaign,
        "n_sites": len(sites),
        "n_by_group": {},
        "sites": [],
    }
    for s in sites:
        g = s.group
        manifest["n_by_group"][g] = manifest["n_by_group"].get(g, 0) + 1
        manifest["sites"].append({
            "site_id": s.site_id,
            "group": s.group,
            "lat": round(s.lat, 5),
            "lon": round(s.lon, 5),
            "elevation_m": round(s.elevation_m, 1),
            "country": s.country,
            "climate_zone": s.climate_zone,
            "std_elev_local_m": round(s.std_elev_local_m, 1),
            "mean_slope_deg": round(s.mean_slope_deg, 2),
            "subregion": s.subregion,
            "era5_source": f"era5_campaign_v3/era5_{s.site_id}.zarr",
        })
    with open(out_yaml, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    logger.info("Wrote manifest to %s", out_yaml)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srtm", type=Path, default=None, help="SRTM Europe GeoTIFF")
    parser.add_argument("--worldcover", type=Path, default=None)  # reserved
    parser.add_argument("--koppen", type=Path, default=None)       # reserved
    parser.add_argument("--n-d-fire", type=int, default=250)
    parser.add_argument("--n-e-mountain", type=int, default=200)
    parser.add_argument("--n-f-wind", type=int, default=150)
    parser.add_argument("--n-c-morpho", type=int, default=150)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=str, default="complex_terrain_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.srtm is None or not args.srtm.exists():
        logger.warning("SRTM not provided — terrain filters disabled (all candidates accepted)")
        srtm_path = None
    else:
        srtm_path = args.srtm

    rng = random.Random(args.seed)

    all_sites: list[Site] = []

    # D_fire: accept any elevation, require std_elev > 20m (Mediterranean plains excluded)
    logger.info("Sampling D_fire (%d sites)...", args.n_d_fire)
    all_sites += sample_category(
        "D_fire", D_FIRE_BBOXES, args.n_d_fire,
        srtm_path=srtm_path, rng=rng,
        min_elev=50.0, min_std_elev=20.0, min_slope_deg=2.0,
    )

    # E_mountain: require altitude > 800m or std_elev > 200m (clearly mountain)
    logger.info("Sampling E_mountain (%d sites)...", args.n_e_mountain)
    all_sites += sample_category(
        "E_mountain", E_MOUNTAIN_BBOXES, args.n_e_mountain,
        srtm_path=srtm_path, rng=rng,
        min_elev=800.0, min_std_elev=150.0, min_slope_deg=5.0,
    )

    # F_wind_onshore: onshore, require slope and elevation > 200m
    logger.info("Sampling F_wind_onshore (%d sites)...", args.n_f_wind)
    all_sites += sample_category(
        "F_wind_onshore", F_WIND_BBOXES, args.n_f_wind,
        srtm_path=srtm_path, rng=rng,
        min_elev=200.0, min_std_elev=50.0, min_slope_deg=3.0,
    )

    # C_morpho: random complex terrain across Europe
    logger.info("Sampling C_morpho (%d sites)...", args.n_c_morpho)
    all_sites += sample_c_morpho(
        args.n_c_morpho, srtm_path=srtm_path, rng=rng,
        min_std_elev=50.0, min_slope_deg=3.0,
    )

    logger.info("TOTAL sites: %d", len(all_sites))
    by_group: dict[str, int] = {}
    for s in all_sites:
        by_group[s.group] = by_group.get(s.group, 0) + 1
    for g, n in sorted(by_group.items()):
        logger.info("  %s: %d", g, n)

    write_csv(all_sites, args.out)
    write_manifest(all_sites, args.manifest, args.campaign)


if __name__ == "__main__":
    main()
