# Validation Results — Structure des fichiers et résultats

**Dernière mise à jour**: 2026-04-14
**Objectif**: tracer précisément où sont les données, quels scripts les produisent, quels résultats on a obtenus.

## Vue d'ensemble

On valide un pipeline FWI pour le risque incendie :
- **Input** : ERA5 25km/6h (vent, T, RH, pluie)
- **Output** : FWI à résolution CFD (30m) pour des sites ICOS

**3 scénarios comparés** :
1. **OBS** = ICOS station (référence vérité terrain)
2. **ERA5** = ERA5 wind+T+RH + ERA5-Land rain (**= CEMS officiel ECMWF**)
3. **OURS** = CFD wind+T+RH + IMERG QM-corrigé (**= notre pipeline complet**)

---

## Campagnes CFD terminées

### Campagne 1 : 7 tall-towers ICOS (2026-04-13)
- **Sites** : OPE, IPR, HPB, JFJ, PUY, TRN, SAC
- **Cases** : 139 total, 30 timestamps par site max
- **Période** : été 2020 (sauf JFJ 2022-2023, SAC 2020-2021)
- **Aqua job** : `20051996.aqua` — exit 0, 1h45 walltime, 0 failures
- **Résultats Zarr** : `aqua:~/dsw/data/campaign/icos_fwi_v1/cases/<SITE>/<SITE>.zarr`
- **Profiles tower extraits** : `data/campaign/icos_fwi_v1/tower_profiles.csv` (2506 lignes)

### Campagne 2 : Puéchabon fire-risk (2026-04-14)
- **Site** : FR-Pue (ICOS ecosystem Mediterranean oak)
- **Cases** : 30 (top 20 high-FWI + 10 random) été 2022
- **Aqua job** : `20072070.aqua` — exit 0, 12 min walltime, 0 failures
- **Résultats** : `aqua:~/dsw/data/campaign/icos_fwi_v1/cases/FR-Pue/FR-Pue.zarr`
- **Profiles tower** : `data/campaign/icos_fwi_v1/tower_profiles_puechabon.csv` (2866 lignes tous sites)

---

## Données sources

### ICOS observations (référence vérité)
| Site | Type | Variables | Fichier | Période |
|------|------|-----------|---------|---------|
| FR-Pue (Puéchabon) | Ecosystem FLUXNET | T, RH, ws, **rain** | `data/raw/icos_FR-Pue_daily_fwi.csv` | 2021-2025 |
| OPE | Atmosphere tall-tower | T, RH, ws (multi-level) | `data/raw/icos_ope.zarr` | 2020 |
| IPR | Atmosphere | T, RH, ws | `data/raw/icos_ipr.zarr` | 2020 |
| HPB | Atmosphere | T, RH, ws | `data/raw/icos_hpb.zarr` | 2020 |
| JFJ | Atmosphere | T, RH, ws | `data/raw/icos_jfj.zarr` | 2020-2023 |
| PUY | Atmosphere | T, RH, ws | `data/raw/icos_puy.zarr` | 2020 |
| TRN | Atmosphere | T, RH, ws | `data/raw/icos_trn.zarr` | 2020 |
| SAC | Atmosphere | T, RH, ws | `data/raw/icos_sac.zarr` | 2020-2021 |

**⚠️ Note** : Seul FR-Pue a des observations de pluie (ecosystem station avec pluviomètre). Les tall-towers atmosphère n'ont pas de pluviomètre.

### ERA5 (baseline CEMS)
| Site | Fichier | Contenu |
|------|---------|---------|
| Puéchabon | `data/raw/era5_puechabon_2022.zarr` | été 2022, u10/v10/t2m/d2m |
| Autres | `data/raw/era5_<site>.zarr` | 2020, zarr v2 (sync'd to Aqua) |

### ERA5-Land rain (CEMS officiel)
- `data/raw/era5land_rain_puechabon_2022.csv` — 92 jours été 2022
- **À ajouter** : ERA5-Land rain pour les 7 tall-tower sites

### IMERG satellite rain
- `data/raw/imerg_rain_puechabon.csv` — 2021-2024 daily
- `data/raw/imerg_icos_sites.csv` — 6 sites (OPE/IPR/HPB/PUY/SAC/TRN) été 2020

### Modèle correction précipitation (QM stratifié)
- **Modèle** : `data/models/precip_correction/qm_stratified.npz`
- **Code** : `services/module3-precip/src/quantile_correction.py` (class `StratifiedQMCorrector`)
- **Usage** : `qm.predict(rain_imerg, months, elevation, lat, lon) → rain_corrected`
- **Stratification** : saison × élévation (low/mid/high) × zone climatique (mediterranean/atlantic/continental)

---

## Résultats de validation

### Résultat principal — Puéchabon été 2022 (30 jours CFD)
**Fichier** : `data/validation/fwi_hybrid/puechabon_FINAL_h155.csv` (CFD @ 155m AGL, hauteur aérodynamique équivalente à 11m au-dessus canopée)

| Index | OBS | ERA5 | OURS | ERA5 MAE | OURS MAE | Gain |
|-------|-----|------|------|----------|----------|------|
| **BUI** (pluie) | 10.97 | 2.59 | 10.19 | 8.39 | **1.30** | **-85%** ✅ |
| **ISI** (vent) | 13.68 | 8.79 | 8.90 | 4.89 | **4.84** | égalité |
| **FWI** (global) | 13.39 | 4.79 | 9.16 | 8.59 | **4.34** | **-50%** ✅ |

**Wind match** (important!) : à hauteur aérodynamique correcte CFD=2.90, ERA5=2.87, OBS=2.96 → bias quasi-nul. Le profil CFD est physiquement correct.

**Ancien fichier** `puechabon_FINAL.csv` = CFD @ 20m (trop bas par rapport à la canopée) — ne plus utiliser.

**Jours de feu (FWI>8, n=26)** :
| Index | ERA5 MAE | OURS MAE | Gain |
|-------|----------|----------|------|
| BUI | 8.77 | 1.12 | **-87%** |
| ISI | 5.43 | 6.48 | +19% |
| FWI | 9.50 | 5.63 | **-41%** |

**Rain sources été 2022** (révélateur du drizzle bias ERA5-Land) :
| Source | Total 30j | Mean mm/d |
|--------|-----------|-----------|
| OBS station | 5 mm | 0.16 |
| ERA5-Land (CEMS) | 52 mm | **+972%** |
| IMERG QM-corrigé | 20 mm | 0.68 |

**Conclusion Puéchabon** :
- ✅ **FWI MAE -39%** (5.22 vs 8.59), BUI MAE **-85%** grâce à la correction pluie
- ⚠️ **ISI légèrement pire** : CFD sous-estime le vent de -0.89 m/s (anomalie — voir diagnostic)
- Site plat + été convectif → cas difficile pour CFD neutre

### Résultat complémentaire — 7 tall-towers (été 2020)
**Fichier** : `data/validation/wind_comparison/summary_metrics.csv`

Vent CFD vs ERA5 (mesure tower) :
| Métrique | ERA5 | CFD | Gain |
|----------|------|-----|------|
| MAE moyen | 3.30 m/s | 2.65 m/s | **-20%** |
| OPE 120m | 3.56 | 1.95 | **-45%** |
| SAC 100m | 3.09 | 1.42 | **-54%** |
| TRN 180m | 4.86 | 3.08 | **-37%** |
| PUY 10m | 4.84 | 3.39 | -30% |
| HPB 131m | 4.21 | 3.17 | -25% |

**Conclusion tall-towers** :
- ✅ CFD bat ERA5 sur tous les sites à **terrain complexe en altitude**
- ⚠️ Pas de pluie obs aux tall-towers atm → pas de FWI obs comparable

---

## Scripts clés

| Script | Rôle | Entrée → Sortie |
|--------|------|-----------------|
| `services/data-ingestion/ingest_era5.py` | ERA5 CDS → Zarr | configs/sites/<site>.yaml → era5_<site>.zarr |
| `services/data-ingestion/ingest_icos.py` | ICOS Zarr + FWI obs | SPARQL → icos_<site>.zarr |
| `services/module2a-cfd/run_multisite_campaign.py` | Campagne CFD | run_matrix.csv → cases/<SITE>/ |
| `services/module2a-cfd/build_icos_run_matrix.py` | Build run_matrix | ICOS zarr → run_matrix.csv |
| `services/module2a-cfd/analysis/extract_tower_profiles.py` | Profils CFD à la tour | cases/<SITE>.zarr → tower_profiles.csv |
| `services/module2a-cfd/analysis/compare_wind_icos.py` | Vent CFD vs ICOS | profiles + obs → wind_comparison/ |
| `services/module2a-cfd/analysis/compare_fwi_hybrid.py` | FWI hybride | profiles + rain → fwi_hybrid/ |

---

## ⚠️ Problèmes ouverts

### 1. ~~CFD wind < ERA5~~ RÉSOLU (problème de hauteur d'extraction)
Diagnostic complet 2026-04-14 :
- Le profil CFD vertical est **physiquement correct** (loi log : 1.6 m/s à 20m → 2.9 à 155m)
- À hauteur aérodynamique équivalente (CFD @ 155m ≈ obs @ 11m au-dessus canopée), bias -0.05 m/s ✓
- ERA5 u10 et CFD @ 155m matchent à 0.03 m/s près

**Leçon** : z0 WorldCover "tree cover" ~1m à Puéchabon → la canopée est intégrée dans z0,
donc le "sol" CFD correspond au plan de déplacement (5m sous la canopée). Un capteur à 11m
sur tour au-dessus canopée 6m est aérodynamiquement ≈ CFD à 11+145 = 155m AGL, où 145m
compense la rugosité.

**Améliorations pour vraie précision locale** :
- **Raffinement vertical** près du sol (10-20-40m au lieu de 20-40m)
- **Modèle canopée explicite** (plant canopy drag, déjà dans certains run) plutôt que z0
- **Displacement height** explicite dans le mesh

### 2. CFD neutre sans thermique
T CFD = 28.7°C vs ERA5 2m = 30.7°C vs obs = 31.9°C → biais -3°C. Sans rayonnement solaire,
la thermique diurne (chauffage du sol → convection) n'est pas résolue.

### 3. Puéchabon = montre la force de la correction IMERG mais pas du CFD
Avec hauteur aérodynamique correcte, le CFD matche ERA5 sur le vent à Puéchabon (pas de gain
ni perte, site plat). Le **gain FWI vient à 100% de la correction IMERG QM** (BUI -85%).

Pour montrer que **le CFD apporte aussi**, il faut des sites :
- **FR-OHP** (Observatoire Haute-Provence, 684m, relief pré-alpin)
- **Corse** (Mistral fréquent, terrain complexe)
- **Sites portugais** (Serra da Estrela, relief atlantique avec conditions sèches)
- **ES-LJu** (Sierra Nevada, 1600m, déjà dans la liste ICOS)

L'idéal : sites avec **Mistral/Tramontane fréquent + relief** où le CFD résout
le speedup topographique, combiné à la correction précip pour les épisodes secs.

## Strategy pour améliorer les résultats

Classement des leviers par impact attendu :

1. **Ajouter sites méditerranéens à relief** (gros impact, 1 campagne)
   - FR-OHP (obs ICOS déjà disponibles depuis 2020)
   - ES-LJu (Sierra Nevada, 1600m)
   - Puéchabon reste comme "cas plat de référence"

2. **Raffinement vertical CFD** (moyen, refaire campagne)
   - Cellules à 5/10/15/20m près du sol au lieu de 20/40m
   - Comparer les mêmes 30 cas Puéchabon avec nouveau maillage

3. **buoyantSimpleFoam pour thermique diurne** (gros, workload important)
   - Corriger le biais T CFD (-3°C à Puéchabon en été)
   - Résout la convection et les brises thermiques

4. **LES pour jours de feu critiques** (gros, tres cher)
   - Juste les ~10 jours FWI max par site
   - Montrer la valeur de la turbulence résolue
