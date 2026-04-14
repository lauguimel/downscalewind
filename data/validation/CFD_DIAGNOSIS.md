# CFD Diagnosis — Puéchabon 2022

## Question posée
Pourquoi T et RH CFD sont-elles pires qu'ERA5, alors qu'ERA5 est en
condition aux limites du CFD ?

## Réponse — deux problèmes distincts

### Problème 1 : BUG dans `prepare_inflow.py` (T et q)

Le script construit le profil T/q **uniquement depuis les niveaux pression
ERA5**. Le niveau le plus bas (1000 hPa) est à géopotentiel ~100m altitude,
soit à **-145m AGL** à Puéchabon (altitude 270m).

```
# services/module2a-cfd/prepare_inflow.py lignes 766-770
era5_data = {
    "u":   store["pressure/u"][:],
    "v":   store["pressure/v"][:],
    "t":   store["pressure/t"][:],  # ← niveaux pression, pas t2m
    "q":   store["pressure/q"][:],  # ← idem
}
```

La spline cubique extrapole ensuite vers z=0..20m AGL à partir de ce T
sous-sol, qui est plus froid de 1-3°C que `t2m` (surface réchauffée par le
soleil). Résultat : CFD T = 28.7°C vs ERA5 t2m = 30.7°C (été 2022).

**Fix** (simple, 1 journée) : dans `reconstruct_inlet_profile`, ajouter
artificiellement un niveau "z=2m" avec T=t2m, q=f(d2m, p_surf) avant la
spline.

### Problème 2 : Limite physique du solveur neutre (vent)

Avec z0=0.05m et u_star=0.14 m/s, le profil log-law CFD donne :
- u(10m) = u*/κ × ln(10/z0) = **1.86 m/s**
- u(100m) = 2.64 m/s (imposé par ERA5 à 1000 hPa)

C'est mathématiquement cohérent, **mais** :
- ERA5 u10 = 2.87 m/s, ERA5 1000 hPa = 2.64 m/s → **profil quasi-uniforme**
- → ERA5 intègre déjà la couche limite convective qui mélange verticalement
- Notre CFD `simpleFoam` neutre respecte strictement la loi log → sur-cisaille

**Diagnostic** : en été méditerranéen, la convection diurne crée une couche
mélangée (Mixed Layer) où le vent est quasi-uniforme. simpleFoam neutre ne
modélise pas ce mélange → sous-estime le vent en surface.

**Fixes possibles** :
1. `buoyantSimpleFoam` avec flux de chaleur surfacique → résout la thermique
2. Ajuster le Monin-Obukhov length en fonction de ERA5 surface H0
3. Utiliser le profil ERA5 "mixed" directement sans log-law en surface

## Impact sur les résultats

| Variable | OBS | ERA5 | CFD | Bug identifié ? |
|----------|-----|------|-----|------------------|
| Wind @ 11m | 2.96 | 2.87 (-0.1) | 1.90 (-1.1) | Limitation physique |
| T @ 2m | 31.9°C | 30.7 (-1.2) | 28.7 (-3.2) | **Bug inflow** |
| RH | 28.8% | 37.4 (+8.6) | 38.9 (+10.1) | **Bug inflow** |

## Recommandations

### Court terme (fix code, re-run campagne)
1. Patch `prepare_inflow.py` pour injecter t2m/d2m comme niveau surface
2. Re-lancer campaign Puéchabon (12 min sur Aqua)
3. Attendu : T bias réduit à ~-1°C, RH bias réduit à ~+2-3%

### Moyen terme (site à terrain complexe)
Ajouter FR-OHP (684m, pré-Alpes) où :
- Terrain complexe → CFD résout speedup topographique (ERA5 rate)
- Conditions souvent stables/neutres (pas de CL convective aussi forte)
- Mistral occasionnel → conditions synoptiques où CFD neutre est approprié

### Long terme (solveur avec physique)
`buoyantSimpleFoam` + flux de chaleur ERA5 → résout convection diurne.
Nécessaire pour couvrir l'ensemble des régimes méditerranéens.

## Pourquoi le résultat actuel reste valide pour le papier

Même avec ces limitations :
- **IMERG_QM seul bat CEMS ERA5 de -62%** sur FWI fire-risk (résultat robuste)
- Le CFD wind fonctionne sur 7 tall-towers européennes (-20% MAE globale,
  -45% à -54% sur OPE 120m, SAC 100m, TRN 180m)
- Le bug T/q inflow n'affecte pas ces résultats-là (on compare juste le vent)

Pour le papier FWI, le message clair est :
1. **Correction pluie = levier principal** (démontré)
2. **Downscaling vent = levier complémentaire** sur sites à terrain complexe
   (démontré sur tall-towers, à démontrer spécifiquement pour FWI en
   ajoutant FR-OHP ou autre site méditerranéen à relief)
3. La thermique reste un chantier futur (buoyantSimpleFoam)
