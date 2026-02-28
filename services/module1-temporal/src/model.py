"""
model.py — AdvectionResidualInterpolator

Architecture :
  1. Advection physique per-niveau en 2D : utilise le vent (u,v) moyen des deux
     snapshots pour transporter S0 vers τ et S1 vers τ depuis chaque direction.
  2. CNN 3D résiduel : corrige les processus non-advectifs (chauffage diurne,
     dissipation, erreur d'ordre 2 de l'advection).
  3. Combinaison bridge : S(τ) = S_adv(τ) + τ·(1-τ)·correction(τ)

Le terme τ·(1-τ) garantit que les bornes sont exactement respectées :
  - correction est zéro-initialisée → le modèle démarre comme advection pure.
  - S(τ=0) = S0, S(τ=1) = S1 par construction.

Conventions tenseurs :
  - Shape entrée/sortie : (B, V, L, H, W) — Batch, Variables, Levels, Lat, Lon
  - Variables : u=0, v=1, z=2, t=3, q=4 (correspond au schéma Zarr DownscaleWind)
  - Niveaux : [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200] hPa (N→S pour lat)

Convention ERA5 lat :
  - Stocké N→S : lat[0] = Nord (plus grande latitude), lat[-1] = Sud
  - v > 0 (vent vers le Nord) → dans l'index lat, mouvement vers des indices PLUS PETITS
  - Ce qui implique que la grille de sampling pour un vent nordique AUGMENTE l'index y
    (pour trouver d'où venait la particule : du Sud = index plus grand)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Indices de variables (schéma Zarr DownscaleWind) ─────────────────────────

IDX_U = 0  # composante zonale du vent (m/s)
IDX_V = 1  # composante méridionale du vent (m/s)


class AdvectionResidualInterpolator(nn.Module):
    """
    Interpolateur temporel par advection physique + correction CNN résiduelle.

    Args:
        n_vars:        Nombre de variables par niveau (default: 5 = u,v,z,t,q)
        n_levels:      Nombre de niveaux de pression (default: 10)
        n_hidden:      Canaux cachés du CNN (default: 48)
        n_intermediate: Nombre de pas intermédiaires à produire (default: 5 pour 6h→1h)
        dx_m:          Espacement spatial en longitude en mètres (default: ~21480m à 40°N, 0.25°)
        dy_m:          Espacement spatial en latitude en mètres (default: ~27830m, 0.25°)
        dt_s:          Intervalle de temps total en secondes (default: 6h = 21600s)
    """

    def __init__(
        self,
        n_vars: int = 5,
        n_levels: int = 10,
        n_hidden: int = 48,
        n_intermediate: int = 5,
        dx_m: float = 21_480.0,   # 0.25° × 111320 × cos(40°)
        dy_m: float = 27_830.0,   # 0.25° × 111320
        dt_s: float = 6 * 3600.0,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_levels = n_levels
        self.n_intermediate = n_intermediate
        self.dx_m = dx_m
        self.dy_m = dy_m
        self.dt_s = dt_s

        # τ pour chaque pas intermédiaire — buffers non-apprenables
        taus = torch.tensor(
            [(k + 1) / (n_intermediate + 1) for k in range(n_intermediate)],
            dtype=torch.float32,
        )
        self.register_buffer("taus", taus)  # (n_intermediate,)

        # ── CNN résiduel ──────────────────────────────────────────────────────
        # Input : cat([S0, S1]) → (B, 2*n_vars, n_levels, H, W)
        # Conv3d avec kernel 3×3×3 et padding=1 : conserve toutes les dimensions
        # Output : n_intermediate * n_vars corrections
        self.cnn = nn.Sequential(
            nn.Conv3d(2 * n_vars, n_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(n_hidden, n_intermediate * n_vars, kernel_size=1),
        )
        # Zéro-init de la couche de sortie :
        # → au début de l'entraînement, corrections = 0 → modèle = advection pure
        nn.init.zeros_(self.cnn[-1].weight)
        nn.init.zeros_(self.cnn[-1].bias)

    # ── Utilitaire : warp 2D per-niveau ──────────────────────────────────────

    def _warp_per_level(
        self,
        field: torch.Tensor,    # (B, V, L, H, W)
        disp_x: torch.Tensor,   # (B, L, H, W) — déplacement normalisé en x (longitude)
        disp_y: torch.Tensor,   # (B, L, H, W) — déplacement normalisé en y (latitude)
    ) -> torch.Tensor:          # (B, V, L, H, W)
        """
        Applique un warp 2D indépendant par niveau via grid_sample.

        disp_x et disp_y sont des déplacements en coordonnées normalisées [-1, 1].
        Un déplacement positif en x signifie "échantillonner à l'est".
        Un déplacement positif en y signifie "échantillonner plus au sud" (convention N→S).
        """
        B, V, L, H, W = field.shape

        # Fusionner les dimensions batch et niveau : (B*L, V, H, W)
        field_flat = field.permute(0, 2, 1, 3, 4).reshape(B * L, V, H, W)
        disp_x_flat = disp_x.reshape(B * L, H, W)
        disp_y_flat = disp_y.reshape(B * L, H, W)

        # Grille identité en coordonnées normalisées [-1, 1]
        # grid[h, w, 0] = x ∈ [-1,1] sur l'axe W (longitude)
        # grid[h, w, 1] = y ∈ [-1,1] sur l'axe H (latitude)
        xs = torch.linspace(-1.0, 1.0, W, device=field.device, dtype=field.dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=field.device, dtype=field.dtype)
        grid_x = xs.unsqueeze(0).expand(H, W)           # (H, W)
        grid_y = ys.unsqueeze(1).expand(H, W)           # (H, W)
        identity = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        identity = identity.unsqueeze(0).expand(B * L, -1, -1, -1)  # (B*L, H, W, 2)

        # Ajouter le déplacement
        disp = torch.stack([disp_x_flat, disp_y_flat], dim=-1)  # (B*L, H, W, 2)
        grid = identity + disp

        # grid_sample : (B*L, V, H, W) avec grid (B*L, H, W, 2)
        # padding_mode='border' : réplique les valeurs aux bords (évite les zéros artificiels)
        warped_flat = F.grid_sample(
            field_flat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B*L, V, H, W)

        # Restaurer la forme (B, V, L, H, W)
        return warped_flat.reshape(B, L, V, H, W).permute(0, 2, 1, 3, 4)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        S0: torch.Tensor,  # (B, V, L, H, W) — snapshot à t₀ (normalisé)
        S1: torch.Tensor,  # (B, V, L, H, W) — snapshot à t₀+Δt (normalisé)
    ) -> torch.Tensor:     # (B, n_intermediate, V, L, H, W)
        """
        Produit les n_intermediate états intermédiaires entre S0 et S1.

        Returns:
            Tensor de forme (B, n_intermediate, V, L, H, W).
            outputs[:, k] est l'état à τ = (k+1)/(n_intermediate+1).
        """
        B, V, L, H, W = S0.shape

        # ── Vent moyen pour l'advection ───────────────────────────────────────
        # On utilise le vent connu (u,v) des deux snapshots
        # Vent moyen : meilleure approximation du vent sur l'intervalle [t0, t1]
        u_mean = (S0[:, IDX_U] + S1[:, IDX_U]) / 2.0  # (B, L, H, W) m/s (normalisé)
        v_mean = (S0[:, IDX_V] + S1[:, IDX_V]) / 2.0  # (B, L, H, W) m/s (normalisé)

        # Déplacement normalisé pour l'intervalle complet Δt :
        # disp_x = u * Δt / Δx * 2/(W-1)  [adimensionnel, espace normalisé [-1,1]]
        # disp_y = v * Δt / Δy * 2/(H-1)
        # Note : les données sont normalisées (z-score) — on dénormalise pour calculer
        # le déplacement physique. En pratique, la normalisation du vent est absorbée
        # dans dx_m_eff et dy_m_eff ci-dessous.
        # IMPORTANT : si u_mean est normalisé par std_u, le déplacement réel est
        # u_mean * std_u * Δt / Δx. Cela est géré par l'attribut eff_dx/dy dans
        # NormStats.denorm_for_advection(). Pour le PoC, les données sont supposées
        # déjà dénormalisées OU le caller passe les valeurs physiques.
        # → Le dataset renvoie les champs normalisés ; l'advection utilise
        #   les indices de vent bruts après dénormalisation partielle.
        #   Voir dataset.py pour la convention adoptée.
        disp_x_total = u_mean * (self.dt_s / self.dx_m) * (2.0 / max(W - 1, 1))
        disp_y_total = v_mean * (self.dt_s / self.dy_m) * (2.0 / max(H - 1, 1))

        # ── CNN résiduel (une seule passe pour tous les τ) ────────────────────
        cnn_input = torch.cat([S0, S1], dim=1)         # (B, 2V, L, H, W)
        corrections = self.cnn(cnn_input)              # (B, n_intermediate*V, L, H, W)

        # ── Combiner advection + résidu pour chaque τ ─────────────────────────
        outputs = []
        for k in range(self.n_intermediate):
            tau = self.taus[k]  # scalaire

            # Advection S0 → τ : la particule à (x,y) au temps τ venait de
            # (x - τ·disp_x, y - ...) pour le vent zonal, avec la convention N→S.
            # - Vent zonal u>0 (vers l'Est) : la particule vient de l'Ouest → x - disp
            # - Vent méridional v>0 (vers le Nord) + convention ERA5 N→S (index↑ = lat↓) :
            #   la particule vient du Sud → index y plus grand → disp_y positif
            S0_warped = self._warp_per_level(
                S0,
                disp_x=-tau * disp_x_total,   # vient de l'Ouest si u>0
                disp_y= tau * disp_y_total,    # vient du Sud si v>0 (N→S convention)
            )

            # Advection S1 → τ (depuis t1 en arrière) :
            # La particule à (x,y) au temps τ sera à (x + (1-τ)·disp) au temps t1
            # → On échantillonne S1 à (x + (1-τ)·disp_x, y - (1-τ)·disp_y)
            S1_warped = self._warp_per_level(
                S1,
                disp_x= (1.0 - tau) * disp_x_total,   # plus à l'Est dans S1
                disp_y=-(1.0 - tau) * disp_y_total,    # plus au Nord dans S1
            )

            # Blending pondéré par la distance aux bornes
            S_adv = (1.0 - tau) * S0_warped + tau * S1_warped  # (B, V, L, H, W)

            # Correction résiduelle avec facteur bridge τ·(1-τ) ∈ [0, 0.25]
            corr = corrections[:, k * V:(k + 1) * V]           # (B, V, L, H, W)
            S_out = S_adv + tau * (1.0 - tau) * corr

            outputs.append(S_out)

        return torch.stack(outputs, dim=1)  # (B, n_intermediate, V, L, H, W)

    def count_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
