"""
build_run_matrix_9k.py — Build 9000-case run matrix for extended campaign.

Groups:
  B1: 300 wind farm centres (top clusters by capacity)
  B2: 150 wind farm overlap points (1 km offset for coherence training)
  C:  150 morphological diversity sites (k-means on SRTM terrain features)

Each site gets 15 ERA5 timestamps selected by k-means on wind speed + direction.

Usage:
    python build_run_matrix_9k.py \
        --wind-farms data/campaign/wind_farms_europe.csv \
        --srtm data/raw/srtm_europe.tif \
        --era5-zarr data/raw/era5_perdigao.zarr \
        --existing-sites data/campaign/sites/sites.csv \
        --output-dir data/campaign/9k
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("build_9k")

# SRTM coverage for the campaign
LAT_MIN, LAT_MAX = 36.0, 55.0
LON_MIN, LON_MAX = -10.0, 10.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ── Group B1: Wind farm cluster centres ────────────────────────────────────

def cluster_farms(farms: list[dict], radius_km: float = 5.0) -> list[list[int]]:
    """Greedy clustering of wind farms within radius_km."""
    coords = [(float(r["latitude"]), float(r["longitude"])) for r in farms]
    assigned = [False] * len(farms)
    clusters = []
    for i in range(len(farms)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(farms)):
            if assigned[j]:
                continue
            if haversine_km(*coords[i], *coords[j]) < radius_km:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def select_wind_farm_centres(farms: list[dict], n_sites: int = 300) -> list[dict]:
    """Select top-N wind farm clusters by total capacity, return centroid."""
    clusters = cluster_farms(farms)

    # Compute cluster stats
    cluster_info = []
    for cl in clusters:
        lats = [float(farms[i]["latitude"]) for i in cl]
        lons = [float(farms[i]["longitude"]) for i in cl]
        caps = [float(farms[i]["capacity_mw"]) for i in cl]
        cluster_info.append({
            "lat": np.mean(lats),
            "lon": np.mean(lons),
            "capacity_mw": sum(caps),
            "n_farms": len(cl),
            "name": farms[cl[0]]["name"],
            "country": farms[cl[0]]["country"],
        })

    # Sort by capacity, take top N
    cluster_info.sort(key=lambda x: -x["capacity_mw"])
    selected = cluster_info[:n_sites]
    log.info("B1: selected %d wind farm centres (%.0f–%.0f MW)",
             len(selected), selected[-1]["capacity_mw"], selected[0]["capacity_mw"])
    return selected


# ── Group B2: Overlap points (1 km offset) ────────────────────────────────

def generate_overlap_points(centres: list[dict], n_overlap: int = 150) -> list[dict]:
    """Generate overlap points: 1 km offset from the largest centres."""
    # Take the top N centres by capacity for overlap
    top = sorted(centres, key=lambda x: -x["capacity_mw"])[:n_overlap]
    overlaps = []
    for c in top:
        # Offset ~1 km east (in degrees: 1km ≈ 0.009° lon at mid-latitudes)
        km_to_deg_lon = 1.0 / (111.32 * math.cos(math.radians(c["lat"])))
        km_to_deg_lat = 1.0 / 110.574
        overlaps.append({
            "lat": c["lat"] + 1.0 * km_to_deg_lat,
            "lon": c["lon"] + 1.0 * km_to_deg_lon,
            "capacity_mw": c["capacity_mw"],
            "name": f"{c['name']}_overlap",
            "country": c["country"],
            "overlap_of": c["name"],
        })
    log.info("B2: generated %d overlap points (1 km offset)", len(overlaps))
    return overlaps


# ── Group C: Morphological diversity sites ─────────────────────────────────

def select_morpho_sites(
    srtm_path: Path,
    existing_sites_csv: Path,
    n_sites: int = 150,
) -> list[dict]:
    """Select terrain-diverse sites via k-means on SRTM features.

    Avoids existing sites (group A) and wind farm locations.
    """
    import rasterio
    from sklearn.cluster import MiniBatchKMeans

    log.info("C: sampling %d morpho sites from SRTM...", n_sites)

    # Load existing sites to avoid
    existing = set()
    if existing_sites_csv.exists():
        with open(existing_sites_csv) as f:
            for row in csv.DictReader(f):
                # Round to 0.1° to define exclusion zones
                key = (round(float(row["lat"]), 1), round(float(row["lon"]), 1))
                existing.add(key)

    with rasterio.open(srtm_path) as src:
        # Sample terrain features on a coarse grid (~10 km spacing)
        step_deg = 0.1  # ~10 km
        lats = np.arange(LAT_MIN + 0.5, LAT_MAX - 0.5, step_deg)
        lons = np.arange(LON_MIN + 0.5, LON_MAX - 0.5, step_deg)

        features = []
        coords_list = []

        for lat in lats:
            for lon in lons:
                key = (round(lat, 1), round(lon, 1))
                if key in existing:
                    continue

                # Read 7×7 km patch (~0.07° at mid-latitudes)
                half_deg = 0.035
                window = rasterio.windows.from_bounds(
                    lon - half_deg, lat - half_deg,
                    lon + half_deg, lat + half_deg,
                    src.transform)
                try:
                    patch = src.read(1, window=window)
                except Exception:
                    continue

                if patch.size < 10 or np.all(patch == src.nodata):
                    continue

                valid = patch[patch != src.nodata]
                if len(valid) < 10:
                    continue

                std_elev = float(np.std(valid))
                mean_elev = float(np.mean(valid))
                relief = float(np.ptp(valid))

                # Skip ocean/flat
                if std_elev < 5 or relief < 20:
                    continue

                # Terrain features for clustering
                dx = np.gradient(valid.reshape(patch.shape) if len(valid) == patch.size
                                 else patch, axis=1)
                dy = np.gradient(valid.reshape(patch.shape) if len(valid) == patch.size
                                 else patch, axis=0)
                slope = np.sqrt(dx**2 + dy**2)
                mean_slope = float(np.mean(slope[slope > 0])) if np.any(slope > 0) else 0

                features.append([std_elev, mean_slope, relief, mean_elev])
                coords_list.append((lat, lon))

    if len(features) < n_sites:
        log.warning("Only %d candidate sites (need %d)", len(features), n_sites)
        n_sites = len(features)

    features = np.array(features)
    # Normalize
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # K-means clustering → pick 1 site per cluster (closest to centroid)
    kmeans = MiniBatchKMeans(n_clusters=n_sites, random_state=42, n_init=3)
    labels = kmeans.fit_predict(features_norm)

    sites = []
    for k in range(n_sites):
        mask = labels == k
        if not mask.any():
            continue
        cluster_features = features_norm[mask]
        dists = np.linalg.norm(cluster_features - kmeans.cluster_centers_[k], axis=1)
        best = np.argmin(dists)
        idx = np.where(mask)[0][best]
        lat, lon = coords_list[idx]
        sites.append({
            "lat": lat,
            "lon": lon,
            "std_elev": features[idx, 0],
            "mean_slope": features[idx, 1],
            "relief": features[idx, 2],
            "cluster_id": k,
        })

    log.info("C: selected %d morpho sites (std_elev %.0f–%.0f m)",
             len(sites), min(s["std_elev"] for s in sites),
             max(s["std_elev"] for s in sites))
    return sites


# ── ERA5 timestamp selection (k-means on wind rose) ───────────────────────

def select_timestamps(era5_zarr: Path, n_ts: int = 15) -> list[str]:
    """Select diverse timestamps from ERA5 via k-means on 10m wind."""
    import zarr
    from sklearn.cluster import MiniBatchKMeans

    store = zarr.open_group(str(era5_zarr), mode="r")
    times = store["coords/time"][:]
    u10 = store["surface/u10"][:, 0, 0]  # single grid point
    v10 = store["surface/v10"][:, 0, 0]

    speed = np.sqrt(u10**2 + v10**2)
    direction = np.degrees(np.arctan2(-u10, -v10)) % 360

    # Features: speed, sin(dir), cos(dir)
    features = np.column_stack([
        speed,
        np.sin(np.radians(direction)),
        np.cos(np.radians(direction)),
    ])
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    kmeans = MiniBatchKMeans(n_clusters=n_ts, random_state=42, n_init=3)
    labels = kmeans.fit_predict(features_norm)

    selected = []
    for k in range(n_ts):
        mask = labels == k
        if not mask.any():
            continue
        dists = np.linalg.norm(features_norm[mask] - kmeans.cluster_centers_[k], axis=1)
        best = np.argmin(dists)
        idx = np.where(mask)[0][best]
        ts = np.datetime64(int(times[idx]), "ns")
        selected.append(str(ts)[:19].replace("T", " "))

    log.info("Selected %d timestamps", len(selected))
    return sorted(selected)


# ── Main: build run matrix ─────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Build 9000-case run matrix")
    parser.add_argument("--wind-farms", required=True, type=Path)
    parser.add_argument("--srtm", required=True, type=Path)
    parser.add_argument("--era5-zarr", required=True, type=Path)
    parser.add_argument("--existing-sites", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-wf-centres", type=int, default=300)
    parser.add_argument("--n-wf-overlap", type=int, default=150)
    parser.add_argument("--n-morpho", type=int, default=150)
    parser.add_argument("--n-timestamps", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load wind farms
    with open(args.wind_farms) as f:
        farms = list(csv.DictReader(f))
    log.info("Loaded %d wind farms", len(farms))

    # ── Group B1: Wind farm centres ──
    wf_centres = select_wind_farm_centres(farms, args.n_wf_centres)

    # ── Group B2: Overlap points ──
    wf_overlaps = generate_overlap_points(wf_centres, args.n_wf_overlap)

    # ── Group C: Morpho sites ──
    if args.n_morpho > 0:
        morpho_sites = select_morpho_sites(args.srtm, args.existing_sites, args.n_morpho)
    else:
        morpho_sites = []
        log.info("C: skipped (n_morpho=0)")

    # ── Select timestamps ──
    timestamps = select_timestamps(args.era5_zarr, args.n_timestamps)

    # ── Build sites.csv ──
    sites_path = args.output_dir / "sites.csv"
    all_sites = []
    site_id = 0

    for c in wf_centres:
        all_sites.append({
            "site_id": f"site_{site_id:05d}",
            "lat": round(c["lat"], 5),
            "lon": round(c["lon"], 5),
            "group": "B1_windfarm",
            "name": c.get("name", ""),
            "capacity_mw": c.get("capacity_mw", 0),
            "country": c.get("country", ""),
        })
        site_id += 1

    for c in wf_overlaps:
        all_sites.append({
            "site_id": f"site_{site_id:05d}",
            "lat": round(c["lat"], 5),
            "lon": round(c["lon"], 5),
            "group": "B2_overlap",
            "name": c.get("name", ""),
            "overlap_of": c.get("overlap_of", ""),
            "country": c.get("country", ""),
        })
        site_id += 1

    for s in morpho_sites:
        all_sites.append({
            "site_id": f"site_{site_id:05d}",
            "lat": round(s["lat"], 5),
            "lon": round(s["lon"], 5),
            "group": "C_morpho",
            "std_elev": round(s["std_elev"], 1),
        })
        site_id += 1

    fieldnames = ["site_id", "lat", "lon", "group", "name", "capacity_mw",
                  "country", "overlap_of", "std_elev"]
    with open(sites_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_sites)
    log.info("Wrote %d sites → %s", len(all_sites), sites_path)

    # ── Build run_matrix.csv ──
    matrix_path = args.output_dir / "run_matrix.csv"
    run_id = 0
    with open(matrix_path, "w", newline="") as f:
        fieldnames_rm = ["run_id", "site_id", "timestamp", "lat", "lon",
                         "group", "priority", "status"]
        w = csv.DictWriter(f, fieldnames=fieldnames_rm)
        w.writeheader()
        for site in all_sites:
            priority = "high" if site["group"] in ("B1_windfarm", "B2_overlap") else "medium"
            for ts in timestamps:
                w.writerow({
                    "run_id": f"run_{run_id:06d}",
                    "site_id": site["site_id"],
                    "timestamp": ts,
                    "lat": site["lat"],
                    "lon": site["lon"],
                    "group": site["group"],
                    "priority": priority,
                    "status": "pending",
                })
                run_id += 1

    log.info("Wrote %d runs → %s", run_id, matrix_path)
    log.info("Summary: %d B1 + %d B2 + %d C = %d sites × %d ts = %d cases",
             len(wf_centres), len(wf_overlaps), len(morpho_sites),
             len(all_sites), len(timestamps), run_id)


if __name__ == "__main__":
    main()
