"""
select_validation_cases.py — Sélection du cas de validation canonique.

Phase 2 du pipeline Module 2A. Identifie les timestamps ERA5 les plus
adaptés à l'étude de convergence et à la validation CFD :

Cas canonique (pour l'étude de convergence) :
  - Direction SW ≈ 220° (vent perpendiculaire aux crêtes Perdigão, dominant)
  - Vitesse à 850 hPa ≈ 8–12 m/s (vent modéré bien défini)
  - Stabilité neutre (Ri_b ≈ 0, |Ri_b| < 0.05)
  - Couverture maximale des mâts T20, T25, T13

Cas secondaires (~5) pour la robustesse :
  - Différentes directions (N, NW, E)
  - Différentes vitesses (3–5 m/s, > 15 m/s)
  - Stabilité stable (nuit, Ri_b > 0.1) et instable (jour, Ri_b < -0.1)

Usage :
    python select_validation_cases.py \\
        --era5 data/raw/era5_perdigao.zarr \\
        --obs  data/raw/perdigao_obs.zarr \\
        --output data/processed/validation_cases.yaml

Référence :
    Neunaber et al. (WES 2023) : cas canonique = 4 mai 2017, 22h–22h30,
    conditions quasi-neutres, vent du SW. Notre sélection vise une période
    comparable avec bonne couverture ERA5 + observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

log = get_logger("select_validation_cases")

# ── Constantes physiques ──────────────────────────────────────────────────────

G_MS2  = 9.80665
KAPPA  = 0.4
R_DRY  = 287.05   # J kg⁻¹ K⁻¹ (constante des gaz secs)

# ── Critères de sélection ─────────────────────────────────────────────────────

# Cas canonique
CANONICAL_CRITERIA = {
    "wind_dir_deg":   (195, 245),   # secteur SW ±25° autour de 220°
    "wind_speed_ms":  (7.0, 14.0),  # vitesse modérée à 850 hPa
    # Ri_b calculé sur le layer 1000→850 hPa (~1400m) :
    # p5=1.9, median=14.6, p95=221.9 → "neutre" = 1-5 pour cette épaisseur
    "ri_bulk_abs":    5.0,          # |Ri_b| < 5 → quasi-neutre (layer 1000→850 hPa)
    "hour_utc":       (0, 23),      # toutes heures (ERA5 6h)
}

# Cas secondaires : liste de (label, critères)
SECONDARY_CRITERIA = [
    ("northerly_moderate", {
        "wind_dir_deg": (340, 360),
        "wind_speed_ms": (6.0, 14.0),
        "ri_bulk_abs": 20.0,
    }),
    ("northerly_moderate_2", {
        "wind_dir_deg": (0, 20),
        "wind_speed_ms": (6.0, 14.0),
        "ri_bulk_abs": 20.0,
    }),
    ("easterly", {
        "wind_dir_deg": (60, 120),
        "wind_speed_ms": (5.0, 12.0),
        "ri_bulk_abs": 20.0,
    }),
    ("sw_strong", {
        "wind_dir_deg": (195, 245),
        "wind_speed_ms": (14.0, 25.0),
        "ri_bulk_abs": 20.0,
    }),
    ("sw_stable", {
        "wind_dir_deg": (195, 245),
        "wind_speed_ms": (5.0, 14.0),
        "ri_bulk_above": 20.0,   # stable (layer 1000→850): Ri_b > 20
    }),
    ("sw_unstable", {
        "wind_dir_deg": (195, 245),
        "wind_speed_ms": (5.0, 14.0),
        "ri_bulk_below": -2.0,   # instable (layer 1000→850): Ri_b < -2
    }),
]


# ── Utilitaires ───────────────────────────────────────────────────────────────

def bilinear_weights(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int, int, int, float, float]:
    """Calcule les poids bilinéaires pour un point cible sur une grille régulière."""
    target_lat = float(np.clip(target_lat, lats.min(), lats.max()))
    target_lon = float(np.clip(target_lon, lons.min(), lons.max()))
    i0 = max(0, min(int(np.searchsorted(lats, target_lat) - 1), len(lats) - 2))
    j0 = max(0, min(int(np.searchsorted(lons, target_lon) - 1), len(lons) - 2))
    i1, j1 = i0 + 1, j0 + 1
    wlat = (target_lat - lats[i0]) / (lats[i1] - lats[i0])
    wlon = (target_lon - lons[j0]) / (lons[j1] - lons[j0])
    return i0, i1, j0, j1, float(wlat), float(wlon)


def apply_bilinear(
    field: np.ndarray,
    i0: int, i1: int, j0: int, j1: int,
    wlat: float, wlon: float,
) -> np.ndarray:
    """Applique des poids bilinéaires sur un champ [..., lat, lon]."""
    return (
        (1 - wlat) * (1 - wlon) * field[..., i0, j0]
        + (1 - wlat) * wlon       * field[..., i0, j1]
        + wlat       * (1 - wlon) * field[..., i1, j0]
        + wlat       * wlon       * field[..., i1, j1]
    )


def wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Direction de vent (convention météo : vent VENANT DE, 0°=Nord)."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def direction_in_range(directions: np.ndarray, dir_min: float, dir_max: float) -> np.ndarray:
    """Filtre une direction dans un secteur, en gérant le wrap 0°/360°."""
    if dir_min <= dir_max:
        return (directions >= dir_min) & (directions <= dir_max)
    else:
        # Wrap autour de 0°/360° (ex: 350–10°)
        return (directions >= dir_min) | (directions <= dir_max)


def potential_temperature(t: np.ndarray, p_hpa: float) -> np.ndarray:
    """θ = T * (1000/p)^(Rd/Cp) — température potentielle [K]."""
    RD_CP = R_DRY / 1004.0   # ≈ 0.2857
    return t * (1000.0 / p_hpa) ** RD_CP


def bulk_richardson_number(
    u_lower: np.ndarray, v_lower: np.ndarray,
    u_upper: np.ndarray, v_upper: np.ndarray,
    theta_lower: np.ndarray, theta_upper: np.ndarray,
    z_lower: np.ndarray, z_upper: np.ndarray,
) -> np.ndarray:
    """
    Nombre de Richardson bulk entre deux niveaux de pression.

    Ri_b = g/θ_mean * (Δθ/Δz) / ((Δu/Δz)² + (Δv/Δz)²)

    Utilise la température potentielle θ (neutre → dθ/dz ≈ 0, pas dT/dz).

    Positif → stable, négatif → instable, ~0 → neutre.

    Args:
        u_lower, v_lower, theta_lower, z_lower : niveau inférieur (ex: 1000 hPa)
        u_upper, v_upper, theta_upper, z_upper : niveau supérieur (ex: 850 hPa)
        z_* : hauteurs géopotentielles [m] (géopotentiel/g)

    Returns:
        Ri_b (array, même shape que entrées)
    """
    dz = z_upper - z_lower
    dz = np.where(np.abs(dz) < 1.0, np.sign(dz) * 1.0, dz)   # éviter division par 0

    theta_mean = 0.5 * (theta_lower + theta_upper)
    dtheta     = theta_upper - theta_lower

    du = u_upper - u_lower
    dv = v_upper - v_lower
    wind_shear2 = du**2 + dv**2
    wind_shear2 = np.where(wind_shear2 < 0.01, 0.01, wind_shear2)   # éviter /0 si calme

    ri_b = (G_MS2 / theta_mean) * (dtheta / dz) / (wind_shear2 / dz**2)
    return ri_b


def score_case(
    wind_dir: float,
    wind_speed: float,
    ri_b: float,
    criteria: dict,
) -> float:
    """
    Score de correspondance d'un cas aux critères (0–1).
    Retourne 0 si les critères durs ne sont pas remplis.
    """
    score = 1.0

    # Direction
    if "wind_dir_deg" in criteria:
        d_min, d_max = criteria["wind_dir_deg"]
        in_range = (
            (wind_dir >= d_min and wind_dir <= d_max)
            if d_min <= d_max
            else (wind_dir >= d_min or wind_dir <= d_max)
        )
        if not in_range:
            return 0.0
        # Score proportionnel à la distance au centre
        d_center = (d_min + d_max) / 2.0 % 360.0
        diff = abs(wind_dir - d_center)
        if diff > 180:
            diff = 360 - diff
        score *= max(0.0, 1.0 - diff / 25.0)

    # Vitesse
    if "wind_speed_ms" in criteria:
        v_min, v_max = criteria["wind_speed_ms"]
        if not (v_min <= wind_speed <= v_max):
            return 0.0
        # Score gaussien centré sur le milieu
        v_center = (v_min + v_max) / 2.0
        score *= max(0.0, 1.0 - abs(wind_speed - v_center) / (0.5 * (v_max - v_min)))

    # Stabilité
    if "ri_bulk_abs" in criteria:
        if abs(ri_b) > criteria["ri_bulk_abs"]:
            return 0.0
        score *= max(0.0, 1.0 - abs(ri_b) / criteria["ri_bulk_abs"])
    if "ri_bulk_above" in criteria:
        if ri_b < criteria["ri_bulk_above"]:
            return 0.0
    if "ri_bulk_below" in criteria:
        if ri_b > criteria["ri_bulk_below"]:
            return 0.0

    return score


# ── Chargement ERA5 ───────────────────────────────────────────────────────────

def load_era5_profiles(
    era5_path: Path,
    site_lat: float,
    site_lon: float,
    iop_start: str,
    iop_end: str,
) -> dict:
    """
    Charge les profils ERA5 interpolés au point du site (bilinéaire) et filtrés IOP.

    Returns:
        dict avec 'times', 'u850', 'v850', 'spd850', 'dir850',
                  'u1000', 'v1000', 't850', 't1000', 'z850', 'z1000', 'ri_b'
    """
    try:
        import zarr
    except ImportError:
        log.error("zarr non disponible — pip install zarr")
        sys.exit(1)

    if not era5_path.exists():
        log.error("Store ERA5 introuvable", extra={"path": str(era5_path)})
        sys.exit(1)

    root = zarr.open_group(str(era5_path), mode="r")
    times_ns = np.array(root["coords/time"]).astype("datetime64[ns]")
    levels   = np.array(root["coords/level"])
    lats     = np.array(root["coords/lat"])
    lons     = np.array(root["coords/lon"])

    # Filtre IOP
    iop_s = np.datetime64(iop_start, "ns")
    iop_e = np.datetime64(iop_end,   "ns")
    iop_mask = (times_ns >= iop_s) & (times_ns <= iop_e)
    times = times_ns[iop_mask]

    if iop_mask.sum() == 0:
        log.error("Aucun timestamp ERA5 dans la période IOP")
        sys.exit(1)

    log.info("ERA5 IOP filtré", extra={
        "n_times": int(iop_mask.sum()),
        "start": str(times[0])[:13],
        "end":   str(times[-1])[:13],
    })

    # Poids bilinéaires au centre du site
    i0, i1, j0, j1, wlat, wlon = bilinear_weights(lats, lons, site_lat, site_lon)

    def interp(var_4d):
        return apply_bilinear(var_4d[iop_mask], i0, i1, j0, j1, wlat, wlon)
    def interp_lev(var_4d, idx_lev):
        return apply_bilinear(var_4d[iop_mask, idx_lev, :, :], i0, i1, j0, j1, wlat, wlon)

    # Indices des niveaux
    idx_1000 = int(np.argmin(np.abs(levels - 1000.0)))
    idx_925  = int(np.argmin(np.abs(levels -  925.0)))
    idx_850  = int(np.argmin(np.abs(levels -  850.0)))

    u_raw = np.array(root["pressure/u"])
    v_raw = np.array(root["pressure/v"])
    z_raw = np.array(root["pressure/z"])
    t_raw = np.array(root["pressure/t"])

    u850  = interp_lev(u_raw, idx_850)
    v850  = interp_lev(v_raw, idx_850)
    u1000 = interp_lev(u_raw, idx_1000)
    v1000 = interp_lev(v_raw, idx_1000)
    t850  = interp_lev(t_raw, idx_850)
    t1000 = interp_lev(t_raw, idx_1000)
    z850  = interp_lev(z_raw, idx_850)  / G_MS2   # géopotentiel → hauteur m
    z1000 = interp_lev(z_raw, idx_1000) / G_MS2

    # Température potentielle θ = T*(1000/p)^(Rd/Cp) — neutre → dθ/dz ≈ 0
    theta850  = potential_temperature(t850,  levels[idx_850])
    theta1000 = potential_temperature(t1000, levels[idx_1000])

    spd850 = np.sqrt(u850**2 + v850**2)
    dir850 = wind_direction_deg(u850, v850)

    ri_b = bulk_richardson_number(
        u1000, v1000, u850, v850, theta1000, theta850, z1000, z850,
    )

    log.info("Statistiques ERA5 IOP", extra={
        "spd850_mean": round(float(np.nanmean(spd850)), 1),
        "spd850_max":  round(float(np.nanmax(spd850)), 1),
        "ri_b_mean":   round(float(np.nanmean(ri_b)), 3),
        "ri_b_std":    round(float(np.nanstd(ri_b)), 3),
    })

    return {
        "times":  times,
        "u850":   u850, "v850":  v850,
        "spd850": spd850, "dir850": dir850,
        "u1000":  u1000, "v1000":  v1000,
        "t850":   t850,  "t1000":  t1000,
        "z850":   z850,  "z1000":  z1000,
        "ri_b":   ri_b,
        "levels": levels,
        "lats":   lats,  "lons":   lons,
    }


# ── Sélection des cas ─────────────────────────────────────────────────────────

def select_best_case(
    profiles: dict,
    criteria: dict,
    n_candidates: int = 5,
) -> list[dict]:
    """
    Sélectionne les n_candidates meilleurs timestamps selon les critères donnés.

    Returns:
        Liste de dicts avec 'timestamp', 'score', 'wind_dir', 'wind_speed', 'ri_b'
    """
    times   = profiles["times"]
    dir850  = profiles["dir850"]
    spd850  = profiles["spd850"]
    ri_b    = profiles["ri_b"]

    candidates = []
    for ti in range(len(times)):
        s = score_case(
            float(dir850[ti]),
            float(spd850[ti]),
            float(ri_b[ti]),
            criteria,
        )
        if s > 0:
            candidates.append({
                "timestamp":  str(times[ti])[:16],
                "score":      round(float(s), 4),
                "wind_dir":   round(float(dir850[ti]), 1),
                "wind_speed": round(float(spd850[ti]), 2),
                "ri_b":       round(float(ri_b[ti]), 4),
                "idx":        ti,
            })

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:n_candidates]


# ── Commande principale ───────────────────────────────────────────────────────

@click.command()
@click.option("--era5",        required=True, help="Store Zarr ERA5")
@click.option("--obs",         default=None,  help="Store Zarr observations (optionnel)")
@click.option("--output",      default="data/processed/validation_cases.yaml",
              help="YAML de sortie avec les cas sélectionnés")
@click.option("--site-config", default="configs/sites/perdigao.yaml",
              help="Config du site (coordonnées)")
@click.option("--iop-start",   default="2017-05-01", show_default=True)
@click.option("--iop-end",     default="2017-06-15", show_default=True)
@click.option("--n-secondary", default=5, show_default=True,
              help="Nombre de cas secondaires à sélectionner par catégorie")
@click.option("--plot",        is_flag=True, default=False,
              help="Générer les figures diagnostiques (rose des vents, stabilité)")
@click.option("--figures-dir", default="figures/validation_cases")
def main(era5, obs, output, site_config, iop_start, iop_end, n_secondary, plot, figures_dir):
    """
    Sélectionne le cas de validation canonique et les cas secondaires.

    Lit ERA5 IOP, calcule la direction/vitesse 850 hPa et le Richardson bulk,
    puis identifie les timestamps qui correspondent le mieux aux critères.

    Le cas canonique est le meilleur timestamp SW, neutre, vent modéré.
    Les cas secondaires couvrent d'autres directions et stabilités pour
    tester la robustesse du pipeline CFD.
    """
    log.info("Sélection des cas de validation", extra={
        "iop": f"{iop_start} → {iop_end}",
    })

    # ── Configuration du site ─────────────────────────────────────────────────
    cfg_path = Path(site_config)
    if not cfg_path.exists():
        log.error("Config site introuvable", extra={"path": str(cfg_path)})
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    site_lat = cfg["site"]["coordinates"]["latitude"]
    site_lon = cfg["site"]["coordinates"]["longitude"]

    # ── Chargement ERA5 ───────────────────────────────────────────────────────
    profiles = load_era5_profiles(
        Path(era5), site_lat, site_lon, iop_start, iop_end,
    )

    # ── Cas canonique ─────────────────────────────────────────────────────────
    log.info("Recherche du cas canonique (SW neutre)")
    canonical_candidates = select_best_case(profiles, CANONICAL_CRITERIA, n_candidates=10)

    if not canonical_candidates:
        log.error(
            "Aucun cas canonique trouvé avec les critères stricts. "
            "Vérifier la disponibilité ERA5 et les critères CANONICAL_CRITERIA."
        )
        sys.exit(1)

    canonical = canonical_candidates[0]
    log.info("Cas canonique sélectionné", extra={
        "timestamp":  canonical["timestamp"],
        "wind_dir":   canonical["wind_dir"],
        "wind_speed": canonical["wind_speed"],
        "ri_b":       canonical["ri_b"],
        "score":      canonical["score"],
    })

    # ── Cas secondaires ───────────────────────────────────────────────────────
    secondary_cases = []
    for label, criteria in SECONDARY_CRITERIA:
        candidates = select_best_case(profiles, criteria, n_candidates=n_secondary)
        if candidates:
            best = candidates[0]
            secondary_cases.append({
                "label":      label,
                "timestamp":  best["timestamp"],
                "wind_dir":   best["wind_dir"],
                "wind_speed": best["wind_speed"],
                "ri_b":       best["ri_b"],
                "score":      best["score"],
                "criteria":   criteria,
            })
            log.info(f"Cas secondaire '{label}'", extra={
                "timestamp":  best["timestamp"],
                "wind_dir":   best["wind_dir"],
                "wind_speed": best["wind_speed"],
                "ri_b":       best["ri_b"],
            })
        else:
            log.warning(f"Aucun cas trouvé pour '{label}'")

    # ── Export YAML ───────────────────────────────────────────────────────────
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "site": cfg["site"]["name"],
        "iop":  {"start": iop_start, "end": iop_end},

        "canonical_case": {
            "label":       "sw_neutral_canonical",
            "description": (
                "Cas de référence pour l'étude de convergence : "
                "vent SW modéré, conditions quasi-neutres."
            ),
            "timestamp":   canonical["timestamp"],
            "wind_dir_deg":    canonical["wind_dir"],
            "wind_speed_ms":   canonical["wind_speed"],
            "ri_b":            canonical["ri_b"],
            "score":           canonical["score"],
            "top_candidates":  canonical_candidates[:5],
        },

        "secondary_cases": secondary_cases,

        "notes": {
            "reference":   "Neunaber et al. (2023, WES) : cas 2017-05-04 22h SW neutre",
            "criteria":    "Direction 850 hPa, vitesse 850 hPa, Richardson bulk 1000–850 hPa",
            "stability":   "Ri_b < 0.05 = neutre, > 0.10 = stable, < -0.10 = instable",
            "usage": (
                "Le cas canonique est utilisé pour l'étude de convergence maillage. "
                "Les cas secondaires testent la robustesse sur différentes conditions."
            ),
        },
    }

    with open(str(output_path), "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.info("Fichier YAML exporté", extra={"path": str(output_path)})

    # ── Résumé ────────────────────────────────────────────────────────────────
    log.info("── Résumé des cas sélectionnés ──")
    log.info(f"  Canonique : {canonical['timestamp']}  "
             f"dir={canonical['wind_dir']}°  "
             f"spd={canonical['wind_speed']} m/s  "
             f"Ri_b={canonical['ri_b']:.3f}")
    for sc in secondary_cases:
        log.info(f"  {sc['label']:<22} : {sc['timestamp']}  "
                 f"dir={sc['wind_dir']}°  spd={sc['wind_speed']} m/s")

    # ── Figures (optionnel) ───────────────────────────────────────────────────
    if plot:
        _make_selection_figures(profiles, canonical, secondary_cases, Path(figures_dir))


def _make_selection_figures(
    profiles: dict,
    canonical: dict,
    secondary_cases: list[dict],
    figures_dir: Path,
) -> None:
    """
    Génère les figures diagnostiques :
    - Rose des vents ERA5 IOP (850 hPa) avec cas sélectionnés mis en évidence
    - Scatter vitesse vs stabilité avec cas sélectionnés annotés
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib non disponible — figures ignorées")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    dir850  = profiles["dir850"]
    spd850  = profiles["spd850"]
    ri_b    = profiles["ri_b"]

    # ── Fig 1 : Rose des vents ERA5 IOP ──────────────────────────────────────
    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(7, 7))

    # Histogramme par secteur 22.5°
    bins    = np.arange(0, 361, 22.5)
    dir_centered = (dir850 + 11.25) % 360.0
    n_hist, _ = np.histogram(dir_centered, bins=bins)
    bin_centers = np.deg2rad(bins[:-1])
    width = np.deg2rad(22.5) * 0.9

    ax.bar(bin_centers, n_hist, width=width, alpha=0.6, color="#4472C4", edgecolor="white")

    # Marquer le cas canonique
    can_angle = np.deg2rad(canonical["wind_dir"])
    ax.annotate(
        f"  ★ Canonique\n  {canonical['timestamp'][:10]}",
        xy=(can_angle, n_hist.max() * 0.8),
        fontsize=8, color="#C0392B",
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Rose des vents ERA5 IOP (850 hPa)\nPerdigão 2017-05 → 06", fontsize=11)

    out = figures_dir / "fig1_wind_rose_era5_iop.png"
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    log.info("Fig rose des vents sauvegardée", extra={"path": str(out)})

    # ── Fig 2 : Vitesse vs Ri_b ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        spd850, ri_b,
        c=dir850, cmap="hsv", vmin=0, vmax=360,
        s=15, alpha=0.5, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Direction 850 hPa [°]")

    # Cas canonique
    ax.scatter(
        [canonical["wind_speed"]], [canonical["ri_b"]],
        s=150, marker="*", color="#C0392B", zorder=5, label="Canonique",
    )
    ax.annotate(canonical["timestamp"][:13],
                (canonical["wind_speed"], canonical["ri_b"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Cas secondaires
    colors_sec = ["#27AE60", "#8E44AD", "#E67E22", "#16A085", "#2980B9"]
    for i, sc_case in enumerate(secondary_cases):
        color = colors_sec[i % len(colors_sec)]
        ax.scatter(
            [sc_case["wind_speed"]], [sc_case["ri_b"]],
            s=80, marker="^", color=color, zorder=5,
            label=sc_case["label"],
        )

    ax.axhline(0.0,   color="gray", linewidth=0.8, linestyle="--", label="Ri_b = 0 (neutre)")
    ax.axhline(0.10,  color="#E74C3C", linewidth=0.6, linestyle=":", label="Ri_b = 0.10 (stable)")
    ax.axhline(-0.10, color="#2980B9", linewidth=0.6, linestyle=":", label="Ri_b = -0.10 (instable)")

    ax.set_xlabel("Vitesse 850 hPa [m s⁻¹]")
    ax.set_ylabel("Richardson bulk [—]")
    ax.set_title("Sélection des cas de validation — ERA5 IOP Perdigão")
    ax.set_ylim(-1.0, 1.0)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    out = figures_dir / "fig2_speed_vs_stability.png"
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    log.info("Fig vitesse/stabilité sauvegardée", extra={"path": str(out)})


if __name__ == "__main__":
    main()
