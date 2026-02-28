"""
baseline_era5.py — Baseline ERA5 interpolé aux positions des mâts Perdigão.

Phase 0 du pipeline de validation (Module 2A) : quantifie l'erreur brute
d'ERA5 avant tout downscaling CFD. C'est le zéro de référence.

Pipeline :
    1. Charger ERA5 Zarr (niveaux de pression + surface, 6h)
    2. Charger observations mâts Perdigão (Zarr 30 min)
    3. Pour chaque timestamp ERA5 et chaque mât :
       - Interpoler u, v ERA5 bilinéairement en lat/lon au point du mât
       - Interpoler verticalement (géopotentiel → hauteurs m ASL) aux hauteurs
         d'instruments (elevation_m + height_agl = hauteur physique ASL)
    4. Aligner temporellement ERA5 6h → obs 30min (moyenne fenêtre ±3h)
    5. Calculer RMSE, biais, R² par mât, hauteur, secteur, stabilité
    6. Exporter CSV résumé + figures

Résultat attendu (littérature) :
    RMSE(|u|) ≈ 2–4 m/s sur les crêtes (ERA5 résolution ~25 km)
    Biais systématique : sous-estimation des vitesses de crête, sur-estimation
    en vallée (ERA5 ne résout pas l'effet de canalisation orographique).

Usage :
    python baseline_era5.py \\
        --era5  data/raw/era5_perdigao.zarr \\
        --obs   data/raw/perdigao_obs.zarr \\
        --output data/processed/baseline_era5_vs_obs.csv

Références :
    Fernando et al. (2019), BAMS — site Perdigão
    Neunaber et al. (2023), WES — RANS Perdigão (base de comparaison)
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import RegularGridInterpolator, interp1d

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

log = get_logger("baseline_era5")

# ── Constantes physiques ──────────────────────────────────────────────────────

G_MS2       = 9.80665    # accélération gravitationnelle [m s⁻²]
KAPPA       = 0.4        # constante de von Kármán
DEG2RAD     = np.pi / 180.0

# Hauteurs standards des instruments sur les mâts (m AGL)
STANDARD_HEIGHTS_AGL = np.array([10, 20, 40, 60, 80, 100], dtype=np.float32)

# Largeur de fenêtre temporelle pour aligner ERA5 6h → obs 30min (en minutes)
ALIGN_WINDOW_MIN = 180  # ±3h → moyennage sur la fenêtre ERA5

# Secteurs de vent (8 × 45°), bords inférieurs
SECTOR_EDGES = np.arange(0, 361, 45, dtype=float)
SECTOR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ── Fonctions d'interpolation ─────────────────────────────────────────────────

def bilinear_interp_2d(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    """
    Interpolation bilinéaire d'un champ [*, lat, lon] à un point cible.

    Args:
        field:       Array (..., n_lat, n_lon) — dernières deux dimensions = lat/lon
        lats:        Latitudes de la grille (croissantes, °N)
        lons:        Longitudes de la grille (croissantes, °E)
        target_lat:  Latitude cible (°N)
        target_lon:  Longitude cible (°E)

    Returns:
        Array (...) — interpolé à (target_lat, target_lon)
    """
    target_lat = np.clip(target_lat, lats.min(), lats.max())
    target_lon = np.clip(target_lon, lons.min(), lons.max())

    orig_shape = field.shape[:-2]
    n_lat, n_lon = field.shape[-2], field.shape[-1]
    flat = field.reshape(-1, n_lat, n_lon)

    result = np.empty(flat.shape[0], dtype=np.float64)
    for i in range(flat.shape[0]):
        interp = RegularGridInterpolator(
            (lats, lons), flat[i].astype(np.float64),
            method="linear", bounds_error=False, fill_value=None,
        )
        result[i] = interp([[target_lat, target_lon]])[0]

    return result.reshape(orig_shape) if orig_shape else float(result[0])


def bilinear_weights(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int, int, int, float, float]:
    """
    Calcule les indices et poids bilinéaires pour une interpolation rapide
    à appliquer sur plusieurs champs au même point (lat, lon).

    Returns:
        (i0, i1, j0, j1, wlat, wlon) tels que :
        val ≈ (1-wlat)*(1-wlon)*f[i0,j0] + (1-wlat)*wlon*f[i0,j1]
             + wlat*(1-wlon)*f[i1,j0]   + wlat*wlon*f[i1,j1]
    """
    target_lat = float(np.clip(target_lat, lats.min(), lats.max()))
    target_lon = float(np.clip(target_lon, lons.min(), lons.max()))

    # Indices encadrants
    i0 = int(np.searchsorted(lats, target_lat) - 1)
    i0 = max(0, min(i0, len(lats) - 2))
    i1 = i0 + 1

    j0 = int(np.searchsorted(lons, target_lon) - 1)
    j0 = max(0, min(j0, len(lons) - 2))
    j1 = j0 + 1

    wlat = (target_lat - lats[i0]) / (lats[i1] - lats[i0])
    wlon = (target_lon - lons[j0]) / (lons[j1] - lons[j0])

    return i0, i1, j0, j1, float(wlat), float(wlon)


def apply_bilinear(
    field: np.ndarray,
    i0: int, i1: int, j0: int, j1: int,
    wlat: float, wlon: float,
) -> np.ndarray:
    """
    Applique des poids bilinéaires pré-calculés sur un champ [..., lat, lon].

    Args:
        field: Array (..., n_lat, n_lon)

    Returns:
        Array (...) — champ interpolé au point cible
    """
    return (
        (1 - wlat) * (1 - wlon) * field[..., i0, j0]
        + (1 - wlat) * wlon       * field[..., i0, j1]
        + wlat       * (1 - wlon) * field[..., i1, j0]
        + wlat       * wlon       * field[..., i1, j1]
    )


def geopot_to_height_asl(z_geopot: np.ndarray) -> np.ndarray:
    """Convertit le géopotentiel [m² s⁻²] en hauteur géométrique ASL [m]."""
    return z_geopot / G_MS2


def interp_vertical_profile(
    z_profile: np.ndarray,
    u_profile: np.ndarray,
    v_profile: np.ndarray,
    target_heights_asl: np.ndarray,
    log_extrapolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolation verticale d'un profil ERA5 à des hauteurs cibles.

    Les profils ERA5 sont ordonnés par pression décroissante (altitude croissante).
    On s'attend à ce que z_profile soit croissant (plus haut = plus grand z ASL).

    Args:
        z_profile:          Hauteurs ASL des niveaux ERA5 [m], shape (n_levels,)
        u_profile:          Composante U du vent [m/s], shape (n_levels,)
        v_profile:          Composante V du vent [m/s], shape (n_levels,)
        target_heights_asl: Hauteurs cibles [m ASL], shape (n_targets,)
        log_extrapolate:    Si True, extrapolation log pour les hauteurs < min(z_profile)

    Returns:
        (u_interp, v_interp) arrays shape (n_targets,)
    """
    # Trier par altitude croissante (ERA5 pressure niveaux : 1000 → 200 hPa)
    sort_idx = np.argsort(z_profile)
    z_sorted = z_profile[sort_idx]
    u_sorted = u_profile[sort_idx]
    v_sorted = v_profile[sort_idx]

    # Supprimer les doublons éventuels
    _, unique_idx = np.unique(z_sorted, return_index=True)
    z_sorted = z_sorted[unique_idx]
    u_sorted = u_sorted[unique_idx]
    v_sorted = v_sorted[unique_idx]

    if len(z_sorted) < 2:
        # Pas assez de niveaux : retourner NaN
        return np.full(len(target_heights_asl), np.nan), np.full(len(target_heights_asl), np.nan)

    # Interpolateur linéaire (bornes : extrapolation constante hors du domaine)
    fu = interp1d(z_sorted, u_sorted, kind="linear", bounds_error=False,
                  fill_value=(u_sorted[0], u_sorted[-1]))
    fv = interp1d(z_sorted, v_sorted, kind="linear", bounds_error=False,
                  fill_value=(v_sorted[0], v_sorted[-1]))

    u_interp = fu(target_heights_asl).astype(np.float32)
    v_interp = fv(target_heights_asl).astype(np.float32)

    return u_interp, v_interp


# ── Métriques ─────────────────────────────────────────────────────────────────

def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u**2 + v**2)


def wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Direction (convention météo : vent VENANT DE, 0–360°, 0° = Nord)."""
    # arctan2(u, v) donne l'angle par rapport au Nord pour la convention "from"
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def compute_sector(wind_dir_deg: float) -> str:
    """Retourne le secteur (N, NE, E, …) pour une direction de vent."""
    # Décaler de 22.5° pour centrer les secteurs
    shifted = (wind_dir_deg + 22.5) % 360.0
    idx = int(shifted / 45.0) % 8
    return SECTOR_NAMES[idx]


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(pred) | np.isnan(obs))
    if mask.sum() < 2:
        return np.nan
    return float(np.sqrt(np.nanmean((pred[mask] - obs[mask])**2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(pred) | np.isnan(obs))
    if mask.sum() < 2:
        return np.nan
    return float(np.nanmean(pred[mask] - obs[mask]))


def r_squared(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = ~(np.isnan(pred) | np.isnan(obs))
    if mask.sum() < 2:
        return np.nan
    p = pred[mask]
    o = obs[mask]
    ss_res = np.sum((o - p)**2)
    ss_tot = np.sum((o - np.mean(o))**2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# ── Chargement des données ─────────────────────────────────────────────────────

def load_era5_store(era5_path: Path) -> dict:
    """
    Charge un store ERA5 Zarr et retourne un dictionnaire d'arrays numpy.

    Returns:
        dict avec clés: 'times', 'levels_hpa', 'lats', 'lons',
                        'u'[time,level,lat,lon], 'v', 'z', 't',
                        'u10'[time,lat,lon], 'v10', 't2m'
    """
    try:
        import zarr
    except ImportError:
        log.error("zarr non disponible — pip install zarr")
        sys.exit(1)

    if not era5_path.exists():
        log.error("Store ERA5 introuvable", extra={"path": str(era5_path)})
        sys.exit(1)

    log.info("Chargement ERA5", extra={"path": str(era5_path)})
    root = zarr.open_group(str(era5_path), mode="r")

    # Coordonnées
    times_ns  = np.array(root["coords/time"])
    times     = times_ns.astype("datetime64[ns]")
    levels    = np.array(root["coords/level"])
    lats      = np.array(root["coords/lat"])
    lons      = np.array(root["coords/lon"])

    # Variables
    data = {
        "times":      times,
        "levels_hpa": levels,
        "lats":       lats,
        "lons":       lons,
        "u":   np.array(root["pressure/u"]),   # [time, level, lat, lon]
        "v":   np.array(root["pressure/v"]),
        "z":   np.array(root["pressure/z"]),   # géopotentiel [m² s⁻²]
        "t":   np.array(root["pressure/t"]),
        "u10": np.array(root["surface/u10"]),  # [time, lat, lon]
        "v10": np.array(root["surface/v10"]),
        "t2m": np.array(root["surface/t2m"]),
    }

    log.info("ERA5 chargé", extra={
        "n_times": len(times),
        "n_levels": len(levels),
        "grid": f"{len(lats)}×{len(lons)}",
        "t_start": str(times[0])[:10],
        "t_end":   str(times[-1])[:10],
    })
    return data


def load_obs_store(obs_path: Path, towers_cfg: dict) -> dict | None:
    """
    Charge un store d'observations Perdigão (Zarr) et retourne les données
    alignées avec les tours définies dans towers_cfg.

    Returns:
        dict avec clés: 'times', 'towers', 'u', 'v', 'u_std', 'T'
        où 'towers' est une liste de dicts {name, lat, lon, elevation_m, heights_m, tower_idx}
        et u/v/u_std/T sont [n_times, n_towers, n_heights]
        Retourne None si le store n'existe pas.
    """
    try:
        import zarr
    except ImportError:
        log.error("zarr non disponible")
        sys.exit(1)

    if not obs_path.exists():
        log.warning("Store observations introuvable — figures ERA5 seul seront générées",
                    extra={"path": str(obs_path)})
        return None

    log.info("Chargement observations", extra={"path": str(obs_path)})
    root = zarr.open_group(str(obs_path), mode="r")

    # Coordonnées obs — nouveau schéma ISFS (sites/ au lieu de masts/)
    times_ns      = np.array(root["coords/time"]).astype("datetime64[ns]")
    obs_lats      = np.array(root["coords/lat"])        # [n_sites]
    obs_lons      = np.array(root["coords/lon"])        # [n_sites]
    obs_elevs     = np.array(root["coords/altitude_m"]) # [n_sites] ASL
    obs_heights   = np.array(root["coords/height_m"])   # [n_heights] AGL
    obs_site_ids  = np.array(root["coords/site_id"])    # bytes → str
    obs_site_ids  = [s.decode("ascii") if isinstance(s, bytes) else s
                     for s in obs_site_ids]

    u_obs = np.array(root["sites/u"])     # [time, site, height]
    v_obs = np.array(root["sites/v"])
    u_std = np.array(root["sites/u_std"])
    T_obs = np.full_like(u_obs, np.nan)   # T absent dans ISFS 5-min → NaN

    # Associer tours de la config → indices dans le store
    # Nouveau format perdigao_towers.yaml : dict plat `towers: { name: {...} }`
    all_towers_cfg = towers_cfg.get("towers", {})

    towers_out = []
    for tname, tdef in all_towers_cfg.items():
        tlat = tdef["lat"]
        tlon = tdef["lon"]
        # Chercher le site ISFS le plus proche par lat/lon
        dist = np.sqrt((obs_lats - tlat)**2 + (obs_lons - tlon)**2)
        idx  = int(np.argmin(dist))
        if dist[idx] > 0.02:   # tolérance ~2 km
            log.warning("Tour absente du store", extra={"tower": tname, "dist_deg": float(dist[idx])})
            continue
        towers_out.append({
            "name":        tname,
            "lat":         tlat,
            "lon":         tlon,
            "elevation_m": tdef.get("altitude_m", tdef.get("elevation_m", 0.0)),
            "heights_m":   np.array(tdef.get("heights_m", [10.0]), dtype=np.float32),
            "tower_idx":   idx,
        })
        log.info("Site ISFS associé", extra={
            "tower": tname, "isfs_site": obs_site_ids[idx],
            "store_idx": idx, "dist_deg": round(float(dist[idx]), 4),
        })

    if not towers_out:
        log.warning("Aucune tour trouvée dans le store observations")
        return None

    # Mapper les hauteurs obs → STANDARD_HEIGHTS_AGL
    # Le store peut avoir un sous-ensemble des hauteurs standards
    height_map = {}
    for h in STANDARD_HEIGHTS_AGL:
        hdiff = np.abs(obs_heights - h)
        if hdiff.min() <= 5.0:
            height_map[int(h)] = int(np.argmin(hdiff))

    log.info("Hauteurs disponibles dans le store", extra={
        "obs_heights": obs_heights.tolist(),
        "mapped": list(height_map.keys()),
    })

    return {
        "times":      times_ns,
        "towers":     towers_out,
        "u":          u_obs,
        "v":          v_obs,
        "u_std":      u_std,
        "T":          T_obs,   # NaN placeholder (ISFS 5-min n'a pas T pour tous les sites)
        "heights_m":  obs_heights,
        "height_map": height_map,
    }


# ── Interpolation ERA5 aux positions des mâts ─────────────────────────────────

def era5_at_mast(
    era5: dict,
    tower: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait le profil ERA5 interpolé (u, v, |u|) aux hauteurs du mât pour
    tous les pas de temps (vectorisé sur les niveaux).

    Stratégie :
      1. Pré-calculer les poids bilinéaires une seule fois (grille fixe).
      2. Appliquer apply_bilinear sur les 3D slices [time, level, lat, lon]
         → résultat [time, level] pour u, v, z.
      3. Convertir z (géopotentiel) → hauteur ASL.
      4. Interpoler verticalement chaque profil [level] aux hauteurs cibles.

    Args:
        era5:   dict renvoyé par load_era5_store()
        tower:  dict avec lat, lon, elevation_m, heights_m

    Returns:
        (u_era5, v_era5, spd_era5) : arrays [n_times, n_heights]
        Les hauteurs correspondent aux heights_m du mât (AGL) + elevation (ASL).
    """
    tlat = tower["lat"]
    tlon = tower["lon"]
    elev = tower.get("elevation_m", tower.get("altitude_m", 0.0))  # m ASL
    heights_agl = tower["heights_m"]   # m AGL
    heights_asl = heights_agl + elev  # m ASL

    n_times   = len(era5["times"])
    n_heights = len(heights_asl)

    lats = era5["lats"]
    lons = era5["lons"]

    # ── Poids bilinéaires (calculés une seule fois) ────────────────────────
    i0, i1, j0, j1, wlat, wlon = bilinear_weights(lats, lons, tlat, tlon)

    # ── Interpolation spatiale vectorisée [time, level] ────────────────────
    # apply_bilinear sur [time, level, lat, lon] → [time, level]
    u_lev = apply_bilinear(era5["u"], i0, i1, j0, j1, wlat, wlon)  # [time, level]
    v_lev = apply_bilinear(era5["v"], i0, i1, j0, j1, wlat, wlon)
    z_lev = apply_bilinear(era5["z"], i0, i1, j0, j1, wlat, wlon)

    # Surface [time]
    u10 = apply_bilinear(era5["u10"], i0, i1, j0, j1, wlat, wlon)  # [time]
    v10 = apply_bilinear(era5["v10"], i0, i1, j0, j1, wlat, wlon)

    # Hauteurs ASL des niveaux de pression [time, level]
    h_lev = geopot_to_height_asl(z_lev)

    # ── Interpolation verticale (boucle sur les pas de temps) ─────────────
    u_out   = np.full((n_times, n_heights), np.nan, dtype=np.float32)
    v_out   = np.full((n_times, n_heights), np.nan, dtype=np.float32)
    spd_out = np.full((n_times, n_heights), np.nan, dtype=np.float32)

    # Point de surface ajouté à ~10m ASL (ERA5 plat, orographie non résolue)
    H_SURF = 10.0

    for ti in range(n_times):
        h_all = np.concatenate([[H_SURF], h_lev[ti]])
        u_all = np.concatenate([[u10[ti]], u_lev[ti]])
        v_all = np.concatenate([[v10[ti]], v_lev[ti]])

        u_i, v_i = interp_vertical_profile(h_all, u_all, v_all, heights_asl)
        u_out[ti]   = u_i
        v_out[ti]   = v_i
        spd_out[ti] = wind_speed(u_i, v_i)

    return u_out, v_out, spd_out


# ── Alignement temporel ERA5 → obs ───────────────────────────────────────────

def align_obs_to_era5(
    era5_times: np.ndarray,
    obs: dict,
    tower: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligne les observations au pas de temps ERA5 (6h) par moyennage
    sur une fenêtre de ±3h centrée sur chaque timestamp ERA5.

    Args:
        era5_times: Timestamps ERA5 [n_times] (datetime64[ns])
        obs:        dict renvoyé par load_obs_store()
        tower:      dict tour (avec tower_idx)

    Returns:
        (u_aligned, v_aligned, u_std_aligned, spd_aligned) : [n_era5_times, n_heights]
        NaN si aucune observation dans la fenêtre.
    """
    tidx = tower["tower_idx"]
    tower_heights_agl = tower["heights_m"]
    obs_heights_m     = obs["heights_m"]
    height_map        = obs["height_map"]

    # Indices de hauteur à extraire : correspondance entre tower_heights_agl et height_map
    target_height_idxs = []
    for h in tower_heights_agl:
        h_int = int(round(float(h)))
        if h_int in height_map:
            target_height_idxs.append(height_map[h_int])
        else:
            target_height_idxs.append(-1)  # -1 = absent

    n_era5  = len(era5_times)
    n_htgt  = len(tower_heights_agl)
    window_ns = np.timedelta64(ALIGN_WINDOW_MIN, 'm')

    u_out    = np.full((n_era5, n_htgt), np.nan, dtype=np.float32)
    v_out    = np.full((n_era5, n_htgt), np.nan, dtype=np.float32)
    ustd_out = np.full((n_era5, n_htgt), np.nan, dtype=np.float32)
    spd_out  = np.full((n_era5, n_htgt), np.nan, dtype=np.float32)

    obs_times = obs["times"]

    for ti, t_era5 in enumerate(era5_times):
        # Fenêtre [t_era5 - 3h, t_era5 + 3h]
        mask = (obs_times >= t_era5 - window_ns) & (obs_times <= t_era5 + window_ns)
        if mask.sum() == 0:
            continue

        for hi, hidx in enumerate(target_height_idxs):
            if hidx < 0:
                continue
            u_w    = obs["u"][mask, tidx, hidx]
            v_w    = obs["v"][mask, tidx, hidx]
            ustd_w = obs["u_std"][mask, tidx, hidx]

            # Masquer les NaN
            valid = ~(np.isnan(u_w) | np.isnan(v_w))
            if valid.sum() == 0:
                continue

            u_out[ti, hi]    = np.nanmean(u_w[valid])
            v_out[ti, hi]    = np.nanmean(v_w[valid])
            ustd_out[ti, hi] = np.nanmean(ustd_w[valid]) if not np.all(np.isnan(ustd_w)) else np.nan
            spd_out[ti, hi]  = wind_speed(u_out[ti, hi], v_out[ti, hi])

    return u_out, v_out, ustd_out, spd_out


# ── Construction du DataFrame résultat ───────────────────────────────────────

def build_result_dataframe(
    era5: dict,
    obs: dict | None,
    towers: list[dict],
    era5_profiles: dict,
    obs_profiles: dict | None,
    iop_start: str = "2017-05-01",
    iop_end:   str = "2017-06-15",
) -> pd.DataFrame:
    """
    Construit le DataFrame complet des comparaisons ERA5 vs obs.

    era5 doit déjà être filtré sur la période IOP (via le filtre dans main).

    Returns:
        DataFrame avec colonnes :
        time, tower, height_m, era5_u, era5_v, era5_spd,
        obs_u, obs_v, obs_spd, obs_u_std,
        err_u, err_v, err_spd,
        wind_dir_era5_deg, sector, hour_utc, is_day, stability
    """
    rows = []
    times_iop = era5["times"]   # déjà filtré IOP

    # Indice 850 hPa pour la direction synoptique
    levels  = era5["levels_hpa"]
    idx_850 = int(np.argmin(np.abs(levels - 850.0)))
    lats    = era5["lats"]
    lons    = era5["lons"]

    for tower in towers:
        tname       = tower["name"]
        heights_agl = tower["heights_m"]

        eu   = era5_profiles[tname]["u"]    # [n_times, n_heights]
        ev   = era5_profiles[tname]["v"]
        espd = era5_profiles[tname]["spd"]

        ou     = obs_profiles[tname]["u"]     if (obs_profiles and tname in obs_profiles) else None
        ov     = obs_profiles[tname]["v"]     if (obs_profiles and tname in obs_profiles) else None
        ospd   = obs_profiles[tname]["spd"]   if (obs_profiles and tname in obs_profiles) else None
        ou_std = obs_profiles[tname]["u_std"] if (obs_profiles and tname in obs_profiles) else None

        # Poids bilinéaires pour 850 hPa (direction synoptique) — calculés une fois
        i0, i1, j0, j1, wlat, wlon = bilinear_weights(lats, lons, tower["lat"], tower["lon"])
        u850_ts = apply_bilinear(era5["u"][:, idx_850, :, :], i0, i1, j0, j1, wlat, wlon)
        v850_ts = apply_bilinear(era5["v"][:, idx_850, :, :], i0, i1, j0, j1, wlat, wlon)

        for ti, t in enumerate(times_iop):
            wdir850 = float(wind_direction_deg(
                np.array([u850_ts[ti]]), np.array([v850_ts[ti]])
            )[0])
            sector  = compute_sector(wdir850)

            # Heure UTC + proxy stabilité (jour/nuit — à Perdigão ≈ +1h UTC)
            t_pd    = pd.Timestamp(t)
            hour_utc = t_pd.hour
            # Lever/coucher soleil approximatif à Perdigão : 5h–20h UTC (heure locale = UTC+1)
            is_day   = 5 <= hour_utc <= 19
            stability = "day_neutral" if is_day else "night_stable"

            for hi, h in enumerate(heights_agl):
                row = {
                    "time":          t_pd,
                    "tower":         tname,
                    "height_m":      float(h),
                    "era5_u":        float(eu[ti, hi])   if not np.isnan(eu[ti, hi])   else np.nan,
                    "era5_v":        float(ev[ti, hi])   if not np.isnan(ev[ti, hi])   else np.nan,
                    "era5_spd":      float(espd[ti, hi]) if not np.isnan(espd[ti, hi]) else np.nan,
                    "wind_dir_era5": wdir850,
                    "sector":        sector,
                    "hour_utc":      hour_utc,
                    "is_day":        is_day,
                    "stability":     stability,
                }
                if ou is not None:
                    row["obs_u"]    = float(ou[ti, hi])    if not np.isnan(ou[ti, hi])    else np.nan
                    row["obs_v"]    = float(ov[ti, hi])    if not np.isnan(ov[ti, hi])    else np.nan
                    row["obs_spd"]  = float(ospd[ti, hi])  if not np.isnan(ospd[ti, hi])  else np.nan
                    row["obs_u_std"]= float(ou_std[ti, hi]) if not np.isnan(ou_std[ti, hi]) else np.nan
                    row["err_u"]    = row["era5_u"]   - row["obs_u"]
                    row["err_v"]    = row["era5_v"]   - row["obs_v"]
                    row["err_spd"]  = row["era5_spd"] - row["obs_spd"]
                else:
                    for col in ("obs_u", "obs_v", "obs_spd", "obs_u_std", "err_u", "err_v", "err_spd"):
                        row[col] = np.nan

                rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ── Métriques récapitulatives ─────────────────────────────────────────────────

def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule RMSE, biais, R² par (tower, height_m, sector) et (tower, height_m, stability).

    Returns:
        DataFrame de métriques avec colonnes :
        groupby_type, tower, height_m, category, n_pairs,
        rmse_spd, bias_spd, r2_spd, rmse_u, rmse_v, bias_u, bias_v
    """
    if "obs_spd" not in df.columns or df["obs_spd"].isna().all():
        log.warning("Pas d'observations disponibles — métriques non calculées")
        return pd.DataFrame()

    records = []

    for group_col, gtype in [("sector", "sector"), ("stability", "stability"), ("height_m", "height")]:
        for (tower, height, cat), sub in df.groupby(["tower", "height_m", group_col]):
            valid = sub.dropna(subset=["era5_spd", "obs_spd"])
            n = len(valid)
            if n < 5:
                continue
            records.append({
                "groupby_type": gtype,
                "tower":        tower,
                "height_m":     height,
                "category":     str(cat),
                "n_pairs":      n,
                "rmse_spd":     rmse(valid["era5_spd"].values, valid["obs_spd"].values),
                "bias_spd":     bias(valid["era5_spd"].values, valid["obs_spd"].values),
                "r2_spd":       r_squared(valid["era5_spd"].values, valid["obs_spd"].values),
                "rmse_u":       rmse(valid["era5_u"].values, valid["obs_u"].values),
                "rmse_v":       rmse(valid["era5_v"].values, valid["obs_v"].values),
                "bias_u":       bias(valid["era5_u"].values, valid["obs_u"].values),
                "bias_v":       bias(valid["era5_v"].values, valid["obs_v"].values),
            })

    return pd.DataFrame(records)


# ── Figures ───────────────────────────────────────────────────────────────────

def make_figures(df: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    """
    Génère les figures de validation ERA5 baseline.

    Fig 1 : RMSE(|u|) vs hauteur par tour
    Fig 2 : Scatter ERA5 vs obs (|u|) pour T20, T25, T13
    Fig 3 : Biais par secteur à 80m (rose des vents)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib non disponible — figures non générées")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Charger plot_style si disponible
    _try_load_plot_style()

    # ── Fig 1 : RMSE vs hauteur par tour ──────────────────────────────────────
    if not summary.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        primary_towers = ["T20", "T25", "T13"]

        for ax, tname in zip(axes, primary_towers):
            sub = summary[
                (summary["tower"] == tname) &
                (summary["groupby_type"] == "height")
            ].sort_values("height_m")
            if sub.empty:
                ax.set_title(f"{tname} (no data)")
                continue

            ax.plot(sub["rmse_spd"], sub["height_m"], "o-", color="#1f77b4",
                    linewidth=1.5, markersize=6, label="RMSE |u|")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("RMSE |u| [m s⁻¹]")
            ax.set_title(tname)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 120)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Hauteur AGL [m]")
        fig.suptitle("ERA5 baseline — RMSE vitesse de vent vs observations Perdigão", fontsize=12)
        plt.tight_layout()
        out = figures_dir / "fig1_rmse_vs_height.png"
        fig.savefig(str(out), dpi=300)
        plt.close(fig)
        log.info("Fig 1 sauvegardée", extra={"path": str(out)})

    # ── Fig 2 : Scatter ERA5 vs obs ───────────────────────────────────────────
    primary = ["T20", "T25", "T13"]
    heights_plot = [80.0]   # hauteur de moyeu typique
    fig, axes = plt.subplots(1, len(primary), figsize=(13, 5))
    for ax, tname in zip(axes, primary):
        sub = df[(df["tower"] == tname) & (df["height_m"].isin(heights_plot))]
        sub = sub.dropna(subset=["era5_spd", "obs_spd"])
        if sub.empty:
            ax.set_title(f"{tname} 80m (no data)")
            continue

        vmax = max(sub["era5_spd"].max(), sub["obs_spd"].max()) + 1
        ax.scatter(sub["obs_spd"], sub["era5_spd"], s=4, alpha=0.4,
                   c="#2ca02c", edgecolors="none")
        ax.plot([0, vmax], [0, vmax], "k--", linewidth=0.8, label="1:1")

        rmse_v = rmse(sub["era5_spd"].values, sub["obs_spd"].values)
        bias_v = bias(sub["era5_spd"].values, sub["obs_spd"].values)
        r2_v   = r_squared(sub["era5_spd"].values, sub["obs_spd"].values)
        ax.text(0.05, 0.95,
                f"RMSE = {rmse_v:.2f} m/s\nBias = {bias_v:+.2f} m/s\nR² = {r2_v:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_xlabel("Obs |u| [m s⁻¹]")
        ax.set_ylabel("ERA5 |u| [m s⁻¹]")
        ax.set_title(f"{tname} — z=80m")
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ERA5 baseline — Scatter |u| à 80m", fontsize=12)
    plt.tight_layout()
    out = figures_dir / "fig2_scatter_spd_80m.png"
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    log.info("Fig 2 sauvegardée", extra={"path": str(out)})

    # ── Fig 3 : Biais par secteur ─────────────────────────────────────────────
    if not summary.empty:
        sub_sector = summary[
            (summary["groupby_type"] == "sector") &
            (summary["height_m"] == 80.0) &
            (summary["tower"] == "T20")
        ]
        if not sub_sector.empty:
            fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
            angles  = np.deg2rad([
                SECTOR_EDGES[SECTOR_NAMES.index(s)] + 22.5
                for s in sub_sector["category"].values
            ])
            biases  = sub_sector["bias_spd"].values
            colors  = ["#d62728" if b > 0 else "#1f77b4" for b in biases]

            ax.bar(angles, np.abs(biases), width=np.deg2rad(40),
                   color=colors, alpha=0.7, edgecolor="white")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)   # sens horaire
            ax.set_title("T20 z=80m — Biais ERA5 par secteur\n"
                         "rouge = surestimation, bleu = sous-estimation", size=10)

            out = figures_dir / "fig3_bias_by_sector_T20_80m.png"
            fig.savefig(str(out), dpi=300)
            plt.close(fig)
            log.info("Fig 3 sauvegardée", extra={"path": str(out)})


def _try_load_plot_style() -> None:
    """Charge le style de publication si plot_style.py est disponible."""
    try:
        style_path = Path(__file__).parent / "plot_style.py"
        if style_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("plot_style", str(style_path))
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "apply"):
                mod.apply()
    except Exception:
        pass   # style non critique


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--era5",          required=True, help="Store Zarr ERA5 (data/raw/era5_perdigao.zarr)")
@click.option("--obs",           default=None,  help="Store Zarr observations (data/raw/perdigao_obs.zarr)")
@click.option("--output",        default="data/processed/baseline_era5_vs_obs.csv",
              help="CSV de sortie (comparaisons par pas de temps)")
@click.option("--summary",       default="data/processed/baseline_era5_summary.csv",
              help="CSV métriques récapitulatives (RMSE, biais, R² par strate)")
@click.option("--towers-config", default="configs/sites/perdigao_towers.yaml",
              help="Configuration des tours de mesure")
@click.option("--figures-dir",   default="figures/baseline",
              help="Répertoire des figures de sortie")
@click.option("--iop-start",     default="2017-05-01", show_default=True)
@click.option("--iop-end",       default="2017-06-15", show_default=True)
@click.option("--towers",        default="tnw01,tse01,tnw08,tse06",
              help="Sites ISFS à traiter (séparés par virgule, ex: tnw01,tse01)")
def main(era5, obs, output, summary, towers_config, figures_dir, iop_start, iop_end, towers):
    """
    Quantifie l'erreur ERA5 brute interpolé aux positions des mâts Perdigão.

    C'est la Phase 0 (baseline) du pipeline de validation Module 2A.
    Produit le CSV de référence avant tout downscaling CFD.

    \\b
    Résultats attendus (littérature) :
      T20 (crête NW) : RMSE ≈ 3–5 m/s — sous-estimation forte vent fort
      T25 (vallée)   : RMSE ≈ 2–4 m/s — sur-estimation modérée
      T13 (flanc)    : RMSE ≈ 2–3 m/s

    Ces chiffres servent de référence pour mesurer le gain du downscaling CFD.
    """
    log.info("Démarrage baseline ERA5", extra={"iop": f"{iop_start} → {iop_end}"})

    # ── Chargement des données ─────────────────────────────────────────────────
    era5_data  = load_era5_store(Path(era5))
    towers_cfg = yaml.safe_load(Path(towers_config).read_text())

    # Filtrer les tours demandées
    # Nouveau format perdigao_towers.yaml : dict plat `towers: { name: {...} }`
    requested_towers = [t.strip() for t in towers.split(",")]
    if "towers" in towers_cfg:
        towers_cfg["towers"] = {
            k: v for k, v in towers_cfg["towers"].items()
            if k in requested_towers
        }
    # Compat ancien format primary_towers/secondary_towers
    for group in ("primary_towers", "secondary_towers"):
        if group in towers_cfg:
            towers_cfg[group] = {
                k: v for k, v in towers_cfg[group].items()
                if k in requested_towers
            }

    obs_data = load_obs_store(Path(obs) if obs else Path("__nonexistent__"), towers_cfg)

    # ── ERA5 aux positions des mâts ───────────────────────────────────────────
    all_towers: list[dict] = []
    # Support nouveau format plat (towers:) et ancien (primary_towers/secondary_towers)
    flat_towers = towers_cfg.get("towers", {})
    for tname, tdef in flat_towers.items():
        if tname in requested_towers:
            all_towers.append({
                "name":        tname,
                "lat":         tdef["lat"],
                "lon":         tdef["lon"],
                "elevation_m": tdef.get("altitude_m", tdef.get("elevation_m", 0.0)),
                "heights_m":   np.array(tdef.get("heights_m", [10.0]), dtype=np.float32),
                "tower_idx":   -1,
            })
    for group in ("primary_towers", "secondary_towers"):
        for tname, tdef in (towers_cfg.get(group) or {}).items():
            if tname in requested_towers:
                all_towers.append({
                    "name":        tname,
                    "lat":         tdef["lat"],
                    "lon":         tdef["lon"],
                    "elevation_m": tdef.get("altitude_m", tdef.get("elevation_m", 0.0)),
                    "heights_m":   np.array(tdef.get("heights_m", [10.0]), dtype=np.float32),
                    "tower_idx":   -1,
                })

    log.info("Tours à traiter", extra={"towers": [t["name"] for t in all_towers]})

    # ── Filtrer ERA5 à la période IOP (avant la boucle coûteuse) ──────────────
    iop_s = np.datetime64(iop_start, "ns")
    iop_e = np.datetime64(iop_end,   "ns")
    iop_mask = (era5_data["times"] >= iop_s) & (era5_data["times"] <= iop_e)
    era5_iop = {
        "times":      era5_data["times"][iop_mask],
        "levels_hpa": era5_data["levels_hpa"],
        "lats":       era5_data["lats"],
        "lons":       era5_data["lons"],
        "u":   era5_data["u"][iop_mask],
        "v":   era5_data["v"][iop_mask],
        "z":   era5_data["z"][iop_mask],
        "t":   era5_data["t"][iop_mask],
        "u10": era5_data["u10"][iop_mask],
        "v10": era5_data["v10"][iop_mask],
        "t2m": era5_data["t2m"][iop_mask],
    }
    log.info("Période IOP ERA5 filtrée", extra={
        "n_times": int(iop_mask.sum()),
        "start": str(era5_iop["times"][0])[:10] if iop_mask.sum() > 0 else "—",
        "end":   str(era5_iop["times"][-1])[:10] if iop_mask.sum() > 0 else "—",
    })
    if iop_mask.sum() == 0:
        log.error("Aucun timestamp ERA5 dans la période IOP — vérifier les dates --iop-start/--iop-end")
        sys.exit(1)

    era5_profiles: dict[str, dict] = {}
    for tower in all_towers:
        tname = tower["name"]
        log.info("Interpolation ERA5 → mât", extra={"tower": tname})
        u_era5, v_era5, spd_era5 = era5_at_mast(era5_iop, tower)
        era5_profiles[tname] = {"u": u_era5, "v": v_era5, "spd": spd_era5}

    # ── Observations alignées sur ERA5 ────────────────────────────────────────
    obs_profiles: dict[str, dict] | None = None
    if obs_data is not None:
        obs_profiles = {}
        # Retrouver tower_idx dans obs_data
        for tower in all_towers:
            tname = tower["name"]
            obs_tower = next(
                (t for t in obs_data["towers"] if t["name"] == tname), None
            )
            if obs_tower is None:
                log.warning("Tour absente des observations", extra={"tower": tname})
                continue
            tower["tower_idx"] = obs_tower["tower_idx"]
            u_al, v_al, ustd_al, spd_al = align_obs_to_era5(
                era5_iop["times"], obs_data, obs_tower,
            )
            obs_profiles[tname] = {"u": u_al, "v": v_al, "u_std": ustd_al, "spd": spd_al}

    # ── Construction du DataFrame ─────────────────────────────────────────────
    log.info("Construction du DataFrame résultat")
    df = build_result_dataframe(
        era5_iop, obs_data, all_towers,
        era5_profiles, obs_profiles,
        iop_start=iop_start, iop_end=iop_end,
    )

    # ── Export CSV ────────────────────────────────────────────────────────────
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    log.info("CSV exporté", extra={"path": str(output_path), "n_rows": len(df)})

    # ── Métriques récapitulatives ─────────────────────────────────────────────
    summary_df = compute_summary_metrics(df)
    if not summary_df.empty:
        summary_path = Path(summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(str(summary_path), index=False)
        log.info("Métriques exportées", extra={"path": str(summary_path), "n_rows": len(summary_df)})

        # Log résumé à l'écran
        log.info("── Résumé RMSE ERA5 baseline ──")
        for tname in requested_towers:
            sub = summary_df[
                (summary_df["tower"] == tname) &
                (summary_df["groupby_type"] == "height") &
                (summary_df["height_m"] == 80.0)
            ]
            if not sub.empty:
                r = sub.iloc[0]
                log.info(
                    f"  {tname} z=80m : RMSE={r['rmse_spd']:.2f} m/s  "
                    f"bias={r['bias_spd']:+.2f} m/s  R²={r['r2_spd']:.2f}  "
                    f"N={r['n_pairs']}"
                )
    else:
        log.info("Pas d'observations disponibles — seuls les profils ERA5 ont été calculés")
        log.info("Pour une comparaison complète, lancez d'abord :")
        log.info("  python services/data-ingestion/ingest_perdigao_obs.py --site perdigao")

    # ── Figures ───────────────────────────────────────────────────────────────
    make_figures(df, summary_df, Path(figures_dir))

    log.info("Baseline ERA5 terminé", extra={
        "output_csv": str(output_path),
        "figures_dir": str(figures_dir),
        "n_towers": len(all_towers),
        "iop_start": iop_start,
        "iop_end": iop_end,
    })


if __name__ == "__main__":
    main()
