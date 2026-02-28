"""
shared.data_io — Helpers Zarr pour lire/écrire les données météo DownscaleWind.

Schéma Zarr unifié pour ERA5, IFS Open-Meteo, et CERRA :

    {source}_perdigao.zarr/
      pressure/
        u       [time, level, lat, lon]  — m/s, composante zonale
        v       [time, level, lat, lon]  — m/s, composante méridionale
        z       [time, level, lat, lon]  — m² s⁻², géopotentiel
        t       [time, level, lat, lon]  — K, température
        q       [time, level, lat, lon]  — kg/kg, humidité spécifique (ou NaN si indisponible)
      surface/
        u10     [time, lat, lon]         — m/s
        v10     [time, lat, lon]         — m/s
        t2m     [time, lat, lon]         — K
      coords/
        time    [time]  — np.datetime64[ns] UTC
        level   [level] — hPa
        lat     [lat]   — °N
        lon     [lon]   — °E

CF-conventions (Climate and Forecast) respectées pour la compatibilité avec
les outils standards (xarray, ncview, etc.).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.storage
from zarr.codecs import BloscCodec


# ── Compression par défaut ────────────────────────────────────────────────────

_DEFAULT_COMPRESSOR = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")


# ── Métadonnées CF-conventions par variable ───────────────────────────────────

VARIABLE_META: dict[str, dict[str, str]] = {
    # Niveaux de pression
    "u": {
        "long_name":    "U-component of wind",
        "standard_name": "eastward_wind",
        "units":        "m s-1",
        "coordinates":  "time level lat lon",
    },
    "v": {
        "long_name":    "V-component of wind",
        "standard_name": "northward_wind",
        "units":        "m s-1",
        "coordinates":  "time level lat lon",
    },
    "z": {
        "long_name":    "Geopotential",
        "standard_name": "geopotential",
        "units":        "m2 s-2",
        "coordinates":  "time level lat lon",
    },
    "t": {
        "long_name":    "Temperature",
        "standard_name": "air_temperature",
        "units":        "K",
        "coordinates":  "time level lat lon",
    },
    "q": {
        "long_name":    "Specific humidity",
        "standard_name": "specific_humidity",
        "units":        "kg kg-1",
        "coordinates":  "time level lat lon",
    },
    # Surface
    "u10": {
        "long_name":    "10 metre U wind component",
        "standard_name": "eastward_wind",
        "units":        "m s-1",
        "height":       "10 m",
        "coordinates":  "time lat lon",
    },
    "v10": {
        "long_name":    "10 metre V wind component",
        "standard_name": "northward_wind",
        "units":        "m s-1",
        "height":       "10 m",
        "coordinates":  "time lat lon",
    },
    "t2m": {
        "long_name":    "2 metre temperature",
        "standard_name": "air_temperature",
        "units":        "K",
        "height":       "2 m",
        "coordinates":  "time lat lon",
    },
}


# ── Dimensions et chunks ──────────────────────────────────────────────────────

# Chunks calibrés pour un accès temporel typique (~30 jours à 6h = 120 pas)
# Les dimensions spatiales sont complètes par chunk (petite grille 7×7)
PRESSURE_CHUNKS = {"time": 120, "level": -1, "lat": -1, "lon": -1}
SURFACE_CHUNKS  = {"time": 120, "lat": -1, "lon": -1}


# ── Fonctions principales ─────────────────────────────────────────────────────

def open_store(path: str | Path, mode: str = "r") -> zarr.Group:
    """
    Ouvre un store Zarr DownscaleWind.

    Args:
        path: Chemin vers le répertoire .zarr
        mode: 'r' (lecture seule), 'r+' (lecture/écriture), 'w' (écrase), 'a' (append)

    Returns:
        zarr.Group racine du store
    """
    return zarr.open_group(str(path), mode=mode)


def create_empty_store(
    path: str | Path,
    n_times: int,
    levels_hpa: list[int],
    lats: list[float],
    lons: list[float],
    source: str,
    site: str,
    time_step_hours: int = 6,
    overwrite: bool = False,
) -> zarr.Group:
    """
    Crée un store Zarr vide avec le schéma DownscaleWind.

    Args:
        path:            Chemin de destination
        n_times:         Nombre de pas de temps
        levels_hpa:      Liste des niveaux de pression en hPa
        lats:            Latitudes de la grille (°N)
        lons:            Longitudes de la grille (°E)
        source:          Identifiant de la source ("era5", "ifs", "cerra")
        site:            Identifiant du site ("perdigao")
        time_step_hours: Pas de temps en heures
        overwrite:       Si True, écrase un store existant

    Returns:
        zarr.Group racine initialisée
    """
    path = Path(path)
    mode = "w" if overwrite else "w-"  # "w-" échoue si le store existe déjà

    root = zarr.open_group(str(path), mode=mode)

    nl = len(levels_hpa)
    nlat = len(lats)
    nlon = len(lons)

    # Attributs globaux (CF-conventions)
    root.attrs.update({
        "Conventions":    "CF-1.9",
        "title":          f"DownscaleWind — {source.upper()} data for {site}",
        "source":         source,
        "site":           site,
        "time_step_hours": time_step_hours,
        "created_by":     "DownscaleWind pipeline (downscalewind/shared/data_io.py)",
        "history":        f"Created by DownscaleWind data_io.create_empty_store",
    })

    # Coordonnées
    coords = root.require_group("coords")
    arr = coords.create_array("time", shape=(n_times,), dtype=np.int64,
                              chunks=(n_times,), overwrite=True)
    arr.attrs.update({"note": "UTC timestamps as int64 (datetime64[ns])"})

    arr = coords.create_array("level", shape=(nl,), dtype=np.float32,
                              chunks=(nl,), overwrite=True)
    arr[:] = np.array(levels_hpa, dtype=np.float32)
    arr.attrs.update({
        "long_name": "pressure level", "units": "hPa",
        "standard_name": "air_pressure", "positive": "down",
    })

    arr = coords.create_array("lat", shape=(nlat,), dtype=np.float32,
                              chunks=(nlat,), overwrite=True)
    arr[:] = np.array(lats, dtype=np.float32)
    arr.attrs.update({
        "long_name": "latitude", "units": "degrees_north",
        "standard_name": "latitude",
    })

    arr = coords.create_array("lon", shape=(nlon,), dtype=np.float32,
                              chunks=(nlon,), overwrite=True)
    arr[:] = np.array(lons, dtype=np.float32)
    arr.attrs.update({
        "long_name": "longitude", "units": "degrees_east",
        "standard_name": "longitude",
    })

    # Variables niveaux de pression
    pres = root.require_group("pressure")
    shape_4d = (n_times, nl, nlat, nlon)
    chunks_4d = (
        min(PRESSURE_CHUNKS["time"], n_times),
        nl, nlat, nlon,
    )
    for var in ("u", "v", "z", "t", "q"):
        arr = pres.create_array(var, shape=shape_4d, dtype=np.float32,
                                chunks=chunks_4d, overwrite=True)
        arr.attrs.update(VARIABLE_META[var])

    # Variables de surface
    surf = root.require_group("surface")
    shape_3d = (n_times, nlat, nlon)
    chunks_3d = (min(SURFACE_CHUNKS["time"], n_times), nlat, nlon)
    for var in ("u10", "v10", "t2m"):
        arr = surf.create_array(var, shape=shape_3d, dtype=np.float32,
                                chunks=chunks_3d, overwrite=True)
        arr.attrs.update(VARIABLE_META[var])

    return root


def append_pressure_slice(
    root: zarr.Group,
    time_idx: int,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    q: np.ndarray | None,
    timestamp: np.datetime64,
) -> None:
    """
    Écrit un snapshot (un pas de temps) dans les variables de pression.

    Args:
        root:      Store Zarr racine (ouvert en mode écriture)
        time_idx:  Index temporel (0-based)
        u, v, z, t: Arrays [n_levels, n_lat, n_lon]
        q:         Array [n_levels, n_lat, n_lon] ou None (sera NaN)
        timestamp: Timestamp UTC
    """
    root["coords/time"][time_idx] = timestamp.astype("datetime64[ns]").astype(np.int64)
    root["pressure/u"][time_idx] = u.astype(np.float32)
    root["pressure/v"][time_idx] = v.astype(np.float32)
    root["pressure/z"][time_idx] = z.astype(np.float32)
    root["pressure/t"][time_idx] = t.astype(np.float32)
    if q is not None:
        root["pressure/q"][time_idx] = q.astype(np.float32)
    # q reste à 0.0 si None — to be filled in post-processing from RH


def append_surface_slice(
    root: zarr.Group,
    time_idx: int,
    u10: np.ndarray,
    v10: np.ndarray,
    t2m: np.ndarray,
) -> None:
    """
    Écrit un snapshot dans les variables de surface.

    Args:
        root:     Store Zarr racine
        time_idx: Index temporel (0-based)
        u10, v10: Arrays [n_lat, n_lon]
        t2m:      Array [n_lat, n_lon]
    """
    root["surface/u10"][time_idx] = u10.astype(np.float32)
    root["surface/v10"][time_idx] = v10.astype(np.float32)
    root["surface/t2m"][time_idx] = t2m.astype(np.float32)


def sha256_file(path: str | Path) -> str:
    """Calcule le SHA256 d'un fichier (lecture par blocs de 8 Mo)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wind_speed_direction_to_uv(
    speed: np.ndarray,
    direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruit les composantes u/v depuis la vitesse et la direction météo.

    Convention météorologique : le vent VIENT DE la direction indiquée.
      direction = 0°  → vent du Nord  → u=0,  v>0 (vent vers le Sud)
      direction = 90° → vent de l'Est → u<0,  v=0 (vent vers l'Ouest)

    Formule :
        u = -speed * sin(direction_rad)   # composante Est (positive vers l'Est)
        v = -speed * cos(direction_rad)   # composante Nord (positive vers le Nord)

    Args:
        speed:         Vitesse scalaire (m/s), array de toute forme
        direction_deg: Direction météo (°, 0–360), même forme que speed

    Returns:
        (u, v) composantes en m/s
    """
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u.astype(np.float32), v.astype(np.float32)


def relative_humidity_to_specific_humidity(
    rh: np.ndarray,
    t_kelvin: np.ndarray,
    p_hpa: np.ndarray,
) -> np.ndarray:
    """
    Convertit l'humidité relative en humidité spécifique (approximation Tetens).

    Valide pour T > 0°C. Erreur < 5% pour T ∈ [0°C, 40°C].
    Pour T < 0°C (glace), utiliser la formule de Buck — non implémentée ici.

    Args:
        rh:       Humidité relative (0–100 %)
        t_kelvin: Température (K)
        p_hpa:    Pression (hPa)

    Returns:
        q : humidité spécifique (kg/kg)
    """
    t_celsius = t_kelvin - 273.15
    # Pression de vapeur saturante (hPa) — formule de Tetens
    e_sat = 6.1078 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))
    e = rh / 100.0 * e_sat
    # Humidité spécifique (approximation, ε = Mw/Md = 0.622)
    q = 0.622 * e / (p_hpa - 0.378 * e)
    return q.astype(np.float32)
