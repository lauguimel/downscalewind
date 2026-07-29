# MANDATE — Phase H' : Correction DEVINE-style (NN pre-correction + surrogate v2 frozen)

Boss source of truth. Departments read only the relevant mission slice.

## 0. Contexte (résumé pour reprise rapide)

**État après échec Phase H E.2 (option canal OBS in-model) 2026-05-27** :

- **M_H0_smoke** YELLOW (2026-05-26) : pipeline OK, toggle_delta plat à 1 epoch (attendu)
- **M_H1 full** YELLOW (2026-05-27) : 30 epochs, val_mse plateau 0.119, **toggle_delta = 3.4e-6 à epoch 29 — canal OBS jamais activé**. Strategy A (sparse additive embedding) a échoué.
- **5 causes diagnostiquées** :
  1. Init `obs_init_std=1e-3` trop faible
  2. Pixel (i,j) random chaque batch → pas de structure spatiale apprenable
  3. Signal dilué 1/225 patches dans la loss
  4. Loss n'a pas de term forçant `pred[pixel_obs] = value_obs`
  5. Cosine schedule descend lr à 2.5e-6, plus assez pour amplifier

**Référence académique DEVINE 2024 (Le Toumelin et al., NPG)** : exactement notre setup résolu différemment :
- CNN surrogate downscaler frozen (DEVINE U-Net entraîné sur 7279 topo Gaussiennes synthétiques + ARPS)
- Petit ANN de correction **AVANT** le downscaler frozen, pas canal in-model
- OBS comme target seul, pas input
- Patch toujours centré sur station
- Loss custom asymétrique τ au pixel central seul
- Backprop end-to-end à travers downscaler frozen
- Résultats : MAE 1.42 m/s, **-7.8% vs DEVINE seul, -15.7% vs raw AROME** sur 55 stations Alpes held-out

**Décision Boss + user 2026-05-27** : abandon option E.2, pivot Phase H' DEVINE-style. Verrou propagation aux pixels non-station = présumée par cohérence du surrogate frozen ; **validation explicite via Perdigão IOP 41 stations dans 6×6 km**.

## 1. High-level objective

**Construire un module de correction NN simple en amont du surrogate v2 frozen**, calibré sur observations station, validé sur Perdigão IOP pour la propagation spatiale.

Pipeline cible :
```
ERA5 vector (408) + features topo locales → ANN correction (MLP small, ~10k params)
                                                          ↓ skip connection
                                                  ERA5 corrigé (408)
                                                          ↓
                                        surrogate v2 frozen (terrain, geo, corrected ERA5)
                                                          ↓
                                              champ pred 5 vars × 180×180×24 AGL
                                                          ↓
                            extract pixel central (90,90, k_10m) → speed_pred
                                                          ↓
                            loss = speed_obs · τ · MSE(speed_obs, speed_pred)
                            τ = 0.6 si speed_obs ≤ speed_pred (underest) | 0.4 sinon
                                                          ↓
                            backprop à travers surrogate v2 frozen → update only ANN
```

**Cible empirique M_H'2** :
- MAE global ≤ 1.0 m/s (vs surrogate raw 1.48, ERA5 raw 1.36)
- MAE wind_class=high ≤ 1.5 m/s (vs raw 2.34)
- bias wind_class=high ≥ -0.5 m/s (vs raw -2.14)
- **Perdigão IOP** : amélioration mesurable de la propagation spatiale (MAE pixel-station improved AND propagation to neighbor pixels validated)

**Done** = (a) M_H'0 smoke pipeline DEVINE-style fonctionnel ; (b) M_H'1 full training winter2223 95 stations converge avec amélioration mesurable ; (c) M_H'2 sur dataset étendu 300-400k pairings avec cible empirique atteinte ou rapport explicite des limites ; (d) M_H'3 décision déploiement / Phase I.

## 2. Out of scope

- **Canal OBS in-model** : abandonné par M_H1 fail.
- **Re-entraînement du surrogate v2** : reste frozen (DEVINE protocol).
- **Phase I summit** : conditional, décidée à la fin de Phase H'.
- **Correction du champ 3D complet** : on cible surface 10 m AGL (où sont les stations et les applications).
- **Classification a priori des régimes physiques** : capturée implicitement par features auxiliaires.

## 3. Constraints

- **Langue** : FR communication, EN code/commits.
- **File budget** : soft ≤500, hard ≤700 LOC.
- **HPC Aqua** : env `fuxicfd`. Jamais mpirun login.
- **Surrogate v2 base** : `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`. **FROZEN** dans Phase H'. `requires_grad=False` sur tous ses params.
- **Dataset OBS** : `obs_unified_noaa_isd_prod.zarr` (362 stations × 6 ans) + Perdigão IOP `data/raw/perdigao_obs.zarr` (41 stations × IOP 2017) + ICOS 7 stations.
- **Dataset pairings** : winter2223 actuel (57k) + après re-inference v2 (300-400k attendus). Re-inference v2 lancée 2026-05-27 (jobs 22066177/78/79 Q).
- **Patch centré sur station** : nouveau dataset loader nécessaire (existing `extract_v2_input_at_coords.py` déjà construit pour ça en M_G6).
- **Backprop à travers frozen** : `surrogate.eval()` + `torch.no_grad` PAS appliqué (sinon gradient bloqué) ; à la place `for p in surrogate.parameters(): p.requires_grad_(False)`.
- **Confirmation user** : avant qsub massif >2h, scp >1 GB, push.
- **Pièges connus** (cf. memory) :
  - PRESSURE_LEVELS canoniques `[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]`
  - ERA5 d2m required
  - Vectorize Zarr writes (1000× speedup)
  - PBS walltime kill → checkpointing tous N=1000 rows
  - qstat -x "Time Use" = CPU time, pas wall
  - GPU usage faible signal possible bottleneck I/O (batch_size + num_workers à tuner)

## 4. Architecture decisions (ADRs)

| Date       | Decision                                                                  | Rationale                                                                              |
|------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| 2026-05-27 | Phase H' = correction DEVINE-style (NN pre-correction + surrogate frozen) | M_H1 fail + DEVINE 2024 paper validé empiriquement                                     |
| 2026-05-27 | ANN architecture = MLP small (~10k params) avec skip connection résiduelle | Apprend Δ vs absolute, stabilise training, généralise mieux                            |
| 2026-05-27 | OBS = target seul, pas input feature                                       | DEVINE protocol. Évite les 5 causes d'échec de Strategy A                               |
| 2026-05-27 | Patch centré sur station (pas random pixel)                               | Structure spatiale apprenable, station au pixel (90,90) toujours                       |
| 2026-05-27 | Loss τ asymétrique (0.6 underest / 0.4 overest)                           | Surrogate v2 slope 0.59 → underestime vent fort, asymétrie cible cette correction      |
| 2026-05-27 | Backprop end-to-end à travers surrogate v2 frozen                         | ANN apprend dans le contexte du downscaler, anticipe la compression                    |
| 2026-05-27 | Validation propagation = Perdigão IOP 41 stations dans 6×6 km             | Seul dataset OBS qui permet de tester propagation pixel-by-pixel                       |
| 2026-05-27 | Skip M_H0 comparaison alternative                                          | DEVINE est la référence empirique. Pivot direct sans re-explorer A/B/C/D               |

## 5. Missions

### M_H'0 — Design + smoke pipeline DEVINE-style

- **Status**: planned (à lancer maintenant en background)
- **Goal**: Engineer (Claude subagent runner) :
  - Étudie le repo DEVINE github (`louisletoumelin/wind_downscaling_cnn`) pour comprendre l'architecture ANN exacte (ANNdirection + ANNspeed)
  - Conçoit le nouveau module `services/module2b-surrogate/src/ann_correction.py` :
    - MLP 2-3 layers, ~10k params
    - Input : ERA5 vector 408 + features topo locales (z0_eff, slope_mean, elevation, distance_to_coast, hour_utc, season)
    - Skip connection : `out = era5_in + delta` (apprend Δ)
    - Init standard Glorot
  - Conçoit `services/module2b-surrogate/src/dataset_v2_obs_centered.py` :
    - Loader OBS pairings (parquet) + grid.zarr inputs déjà préparés
    - Patch centré sur station (pixel 90,90 = station via `extract_v2_input_at_coords.py` déjà construit)
    - Target = OBS speed au pixel central, 10 m AGL
  - Conçoit `services/module2b-surrogate/train_v2_devine_style.py` :
    - Pipeline : (ANN → ERA5_corrigé → surrogate v2 frozen → pred → loss au pixel central)
    - `for p in surrogate.parameters(): p.requires_grad_(False)` (mais surrogate reste en mode eval/train)
    - Loss custom τ asymétrique
    - Optimizer sur ANN params seul (Adam, lr=1e-3 initial)
    - MLflow logging
  - Smoke training : 50 stations × winter2223 = ~5k pairings, 5 epochs sur Aqua H100, walltime 1h
  - Validation smoke :
    - Loss baisse visiblement
    - `pred[pixel_central]` se rapproche de `speed_obs` au pixel central
    - **Comparaison vs baseline raw surrogate** sur même 5k pairings : MAE should improve
- **Inputs** :
  - DEVINE github : `https://github.com/louisletoumelin/wind_downscaling_cnn`
  - DEVINE paper : https://npg.copernicus.org/articles/31/75/2024/
  - `services/module2b-surrogate/src/{dataset_v2_vit,model_vit_v2}.py` (architecture surrogate v2)
  - `services/module2b-surrogate/utils/inference_input.py` (M_G6 deliverable, patch centré coord)
  - `data/inference/noaa_winter2223.parquet` (pairings actuels)
  - `data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
  - `.orchestrator/memory/{engineer,department,boss}.md`
- **Allowed edit zones** :
  - `services/module2b-surrogate/src/ann_correction.py` (nouveau)
  - `services/module2b-surrogate/src/dataset_v2_obs_centered.py` (nouveau)
  - `services/module2b-surrogate/train_v2_devine_style.py` (nouveau)
  - `services/module2b-surrogate/eval_devine_style.py` (nouveau, validation smoke)
  - `configs/training/devine_style_smoke.yaml` (nouveau)
  - `configs/hpc/devine_style_smoke.pbs` (nouveau)
  - `data/models/surrogate_v2_devine_smoke/` (output)
  - `test/scratch/`, `scratch/`, `tmp/`
- **Forbidden actions** :
  - Pas de modification surrogate v2 base (`model_vit_v2.py`, `dataset_v2_vit.py`, `best.pt`)
  - Pas de modification M_H0/M_H1 deliverables (`model_vit_v2_e2.py` etc — restent là pour archive)
  - Pas de full training (smoke only ≤50 stations)
  - Pas de commit/push sans Boss approval
- **Exit criterion** :
  - Smoke training tourne sans crash sur Aqua H100
  - Loss baisse au cours des 5 epochs (factor ≥ 2× idéalement)
  - Comparison sur 5k pairings : `MAE(speed_pred_corrigé) < MAE(speed_pred_raw)` mesurable
  - Rapport Department ≤300 mots : architecture détails, smoke metrics, comparaison avant/après, recommandation proceed M_H'1

### M_H'1a — JJA 2023 training DONE GREEN (2026-06-01)

- **Status**: ✅ DONE GREEN
- **Job**: 22135584.aqua, walltime 11h15, exit 0, 8 epochs
- **Config V2 winner** : lr=5e-4 + grad_clip=1.0 + τ=0.6/0.4 DEVINE default + batch_size=4
- **Dataset**: JJA 2023 = 12061 train + 3015 val pairings × 214 stations FR/ES/PT
- **Results**:
  - val_mae = 1.222 (vs raw 1.812) = **-32.5% MAE** ✅
  - val_bias = +0.008 (vs raw -1.453) = **bias quasi-nul** ✅
  - Best epoch 5, stable plateau ep 5-7
  - No epoch 1 overshoot (V2 config kills both H1 and H2)
- **Output**: `/home/maitreje/dsw/data/models/surrogate_v2_devine_M_H1_jja/best.pt`

### M_H'1c — Validation Perdigão IOP propagation spatiale (planned)

- **Status**: planned (priorité 1 après M_H'1a)
- **Goal**: tester si la correction M_H'1a best.pt se propage aux pixels voisins (pas juste pixel station). Perdigão IOP a 41 stations dans 6×6 km = test cross-pixel within same domain.
- **Strategy**:
  - Charger M_H'1a best.pt + surrogate v2 frozen
  - Inférer sur Perdigão IOP 2017 (41 stations × IOP timestamps, déjà ingéré dans `data/raw/perdigao_obs.zarr`)
  - Pour chaque station Perdigão : eval ANN + surrogate au pixel station vs ANN + surrogate au pixel voisin (10-50m around)
  - Métrique : MAE à station, MAE aux voisins, gradient de correction (smooth/discontinue)
- **Exit criterion**: si MAE pixels voisins ≤ MAE pixel station × 1.5 → propagation OK ; sinon → correction overfite le pixel station
- **Allowed edit zones**: `services/validation/audit_devine_perdigao.py` (new), `data/validation/phase_H_prime_perdigao/`
- **ETA**: ~2-3h compute

### M_H'1d — LOSO honest spatial split (planned)

- **Status**: planned (priorité 2)
- **Goal**: re-évaluer best.pt M_H'1a avec un split watertight 80/20 stations différentes (pas val random 20% des pairings actuels). Confirmer rigueur du -32% gain.
- **Strategy**:
  - Read existing best.pt M_H'1a
  - Re-split 214 stations JJA → 170 train / 44 test (DIFFÉRENTES)
  - Re-eval best.pt sur les 44 test stations
  - Comparer MAE test vs MAE train+val mixe
- **Exit criterion**: MAE test ≤ MAE val M_H'1a × 1.3 (acceptable degradation 30%). Si MAE test >> MAE val → overfitting station-id.
- **Allowed edit zones**: `services/validation/audit_devine_loso.py` (new)
- **ETA**: ~1h compute

### M_H'1b — Multi-season training enrichi (features physiques + pré-matérialisation 264k)

- **Status**: blocked on M_H'1a verdict (DONE 2026-06-01) + design features enrichies
- **Goal**: étendre M_H'1a JJA à 264k pairings 4 saisons (winter2223 + mam2023 + jja2023 + son2023) en **enrichissant les features auxiliaires** avec contexte physique saisonnier/diurne. Un seul modèle apprend des corrections différenciées par régime (canicule vs Mistral vs catabatique vs synoptique).

- **Stratégie 2 temps** :

  **M_H'1b' (priorité)** — features dérivées de l'ERA5 existant (pas de re-download CDS) :
  - Ajouter dans `dataset_v2_obs_centered.py` les features suivantes au `topo_features` (vector aux passé à l'ANN) :
    - `gradient_T_850_surf` = T(850 hPa) - t2m → instability thermique, signal vent thermique
    - `gradient_T_500_850` = T(500 hPa) - T(850 hPa) → mid-trop warming, signal canicule
    - `RH_surface` = f(t2m, d2m) via formule Magnus-Tetens
    - `q_surface` = humidité spécifique surface (proxy sécheresse)
  - `topo_dim` augmente de 8 → 12
  - Adapt `ann_correction.py` ANNCorrection.__init__ pour accepter topo_dim=12

  **M_H'1b'' (si gain insuffisant)** — features ERA5 nouvelles (re-download CDS) :
  - Re-ingérer ERA5 hourly 4 saisons avec : `tcc` (cloud cover), `ssrd` (solar radiation), `cape` (instability), `blh` (boundary layer height)
  - Étend topo_dim à 16
  - Coût : ~1-2 jours CDS + storage

- **Pré-matérialisation 264k grid.zarr** (prérequis training) :
  - Le loader actuel route via era5_store unique → multi-saison impossible sans refactor OU sans cache pré-matérialisé
  - **Solution choisie** : pré-matérialiser tous les 264k grid.zarr dans `M_H1b_cache/` en 4 jobs PBS séparés (un par saison), chacun pointant vers son era5_store
  - Une fois le cache complet, le training loader skip era5_store lookup (overwrite_cache=false)
  - ~14h compute total (parallèle si Aqua a 4 nodes CPU dispo)

- **Strategy training** :
  - Config V2 winner inchangée : lr=5e-4, grad_clip=1.0, τ=0.6/0.4 DEVINE default, batch_size=4
  - topo_dim 12 (M_H'1b') ou 16 (M_H'1b'')
  - 8-10 epochs (early stop patience=3)
  - Walltime ~50h (264k pairings × 5h/epoch × ~10 epochs)
  - LOSO honest split (cf. M_H'1d) — réutiliser le split watertight 80/20 stations

- **Allowed edit zones** :
  - `services/module2b-surrogate/src/dataset_v2_obs_centered.py` (modify: add 4 physical features)
  - `services/module2b-surrogate/src/ann_correction.py` (modify: topo_dim 8→12)
  - `services/module2b-surrogate/train_v2_devine_style.py` (modify only if multi-cache support needed)
  - `services/data-ingestion/prematerialise_grid_zarrs.py` (new helper, pour pré-mat 264k)
  - `configs/training/devine_style_M_H1b.yaml`
  - `configs/hpc/devine_style_M_H1b.pbs`
  - `configs/hpc/prematerialise_grid_zarrs_*.pbs` (4 fichiers, un par saison)
  - `data/models/surrogate_v2_devine_M_H1b/`

- **Forbidden actions** :
  - Pas de modification de model_vit_v2.py, dataset_v2_vit.py, surrogate v2 best.pt (frozen)
  - Pas de commit/push sans Boss approval
  - Pas de qsub walltime > 60h sans Boss approval

- **Exit criterion** :
  - 264k grid.zarrs pré-matérialisés
  - Training M_H'1b complet, val_mae held-out spatial ≤ M_H'1a JJA val_mae × 1.2 (acceptable degradation cross-saison)
  - Comparaison features ablation : MAE avec/sans `gradient_T_850_surf`, `RH_surface`, etc. — identifie quel feature contribue le plus
  - Report ≤500 mots Boss avec décision M_H'1b'' nécessaire ou pas

- **Total ETA** : 6-8 jours wall (½j eng features + ~14h pré-mat + ~50h training + ~½j analyse)

### M_H'1 — Full training Phase H' winter2223 + validation Perdigão IOP (archived)

- **Status**: replaced by M_H'1a/b/c/d sub-missions
- **Goal**: training full sur 95 stations winter2223 (57k pairings), 30 epochs. LOSO spatial split 80/15/5 stations. **Validation propagation sur Perdigão IOP 41 stations** comme test set additionnel.
- **Allowed edit zones** :
  - `configs/training/devine_style_full_winter2223.yaml`
  - `configs/hpc/devine_style_full_winter2223.pbs`
  - `data/models/surrogate_v2_devine_winter2223/`
  - `services/validation/audit_devine_perdigao.py` (nouveau, validation propagation)
- **Exit criterion** :
  - val_mse train/val converge (no overfitting visible)
  - test_mae < val_mae × 1.3 (signe que LOSO honest)
  - **Perdigão IOP test** : MAE at pixel station ≤ raw surrogate MAE AND propagation aux pixels voisins ne dégrade pas (vérifier MAE pixels 90±5)
  - Si Perdigão dit "correction overfit pixel" → escalate Boss pour rescope

### M_H'2 — Full training Phase H' dataset étendu (après re-inference v2)

- **Status**: blocked on M_H'1 GREEN + re-inference v2 jobs 22066177/78/79 complete
- **Goal**: training full sur dataset 300-400k pairings (4 saisons × 362 stations), même architecture, LOSO spatial honest.
- **Cible empirique** : MAE global < 1.0 + MAE wind_class=high < 1.5 + bias wind_class=high > -0.5

### M_H'3 — Audit final + décision Phase I

- **Status**: blocked on M_H'2
- **Goal**: Boss décide GO/NO-GO Phase I summit (re-sim alpine ~50 cases) selon performance par strate.

## 6. Mission graph

```
                  M_H'0 — smoke DEVINE-style (5k pairings, 5 epochs, 1h)
                                  │ GREEN
                                  ▼
              ┌─── re-inference v2 jobs 22066177/78/79 (24-36h, parallel) ────┐
              ▼                                                              ▼
   M_H'1 — full winter2223 (95 stations, 57k pairings)                   parquet étendu
              │                  300-400k pairings après merge
              │ GREEN + Perdigão validation OK
              ▼
   M_H'2 — full dataset étendu (300-400k pairings, 30 epochs)
              │ GREEN cible empirique
              ▼
   M_H'3 — audit final + décision Phase I
```

## 7. Decisions taken after M_H'0 / M_H'1

### Verdict M_H'1c/1f/1g (2026-06-02/03) — voie ANN plafonne sur Perdigão, 3 leviers convergents
- M_H'1a descripteurs 8 → Perdigão centre 2.27 ; M_H'1f phys 12 → 2.15 ; M_H'1g encodeur 180×180 → 2.32 (raw cible 1.37).
- NOAA val préservé partout (1.21–1.23, −32% MAE, bat ERA5). Propagation 100% lisse.
- M_H'1a déployable pour son domaine (été plaine EU, fire weather). Perdigão = OOD terrain raide.

### RE-SCOPE user 2026-06-03 — M_H'1h (verdict 1g jugé PRÉMATURÉ : l'ANN n'a jamais eu d'obs crête)
- L'encodeur M_H'1g a la capacité d'une correction position-dépendante mais a été entraîné sur NOAA JJA
  plaine → aucun exemple de crête → backprop sans signal. Conclure « mur = surrogate » = prématuré.
- Décisions user : (1) obs montagne **raide** via NOAA ISD +CH/AT/IT (SYNOP-MF/OGIMET morts ; AEMET écarté,
  pas de clé) ; (2) ablation **M_H'1f scalaire VS M_H'1g encodeur** sur dataset combiné 4 saisons + obs crête.
- Données 4 saisons DÉJÀ faites : `noaa_seasons_all_v2.parquet` 264 773 rows (Aqua). radcloud absent (pas requis).
- Conditionnel : si M_H'1h répare Perdigão → voie ANN réhabilitée, étage B PAS le mur. Si plafonne AVEC obs
  crête → renforce (sans prouver) « mur = surrogate » → Phase I (surrogate v3 terrain raide) justifiée.

### Programme M_H'1h
- **M_I1** Ingestion obs raide (NOAA +SZ/AU/IT) + helper pente + DEM Alpes → `obs_unified_steep.zarr`.
  **DONE GREEN 2026-06-03** : 567 stations (362 prod + 205 Alps AT/IT/CH), 55 Alps >15° pente (max 38°,
  couvre Perdigão ~17°). NCEI down → bascule miroir AWS (`--source aws`). Store/code LOCAUX, pas commité.
- **M_I2** Pairings 4 saisons obs montagne (inférence v2). Smoke GREEN 2026-06-03 (surrogate sous-prédit
  la crête = signal confirmé). **ÉLARGI EU-Med dry-summer 2026-06-04** (user) : Ibérie (déjà dans 264k pairings)
  + Apennins IT-sud (nouveau) + Alpes (205) + S-France. DEVINE = Alps-only + dégrade hors Alpes → valide multi-région.
  - **M_I2b** Ré-ingest ERA5 mam/jja/son grille étendue lon −10→20 lat 35→49 (CDS Aqua). EN COURS 2026-06-04.
  - **M_I2c** Ingest Apennins IT-sud (NOAA AWS, slope-filtré) + calcul pente stations prod FR/ES/PT (flag dry-Med raides).
  - **M_I2d** Pairings Alpes + Apennins × 4 saisons (après M_I2b). winter2223 déjà OK (lon→26.8). Grèce inexploitable (ISD).
- **M_I3** Training ablation M_H'1f (scalaire + pente) ET M_H'1g (encodeur), LOSO hold-out montagne + Perdigão.
- **M_I4** Verdict Boss : voie ANN réhabilitée OU Phase I confirmée.

## 8. Pointers

- **Verdict M_H1 fail** : `.orchestrator/memory/boss.md` §M_H1 + history.yaml `data/models/surrogate_v2_e2_stage1/`
- **DEVINE paper** : https://npg.copernicus.org/articles/31/75/2024/
- **DEVINE github** : https://github.com/louisletoumelin/wind_downscaling_cnn
- **Surrogate v2 base path** : `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
- **OBS unified** : `/home/maitreje/dsw/data/raw/obs_unified_noaa_isd_prod.zarr`
- **Perdigão IOP** : `data/raw/perdigao_obs.zarr` (41 tours dans 6×6 km, IOP 2017)
- **Audit Phase G baseline** : `data/validation/phase_G_obs_audit/prod_winter2223/REPORT.md`
- **Re-inference v2 jobs** : 22066177 (MAM) / 22066178 (JJA) / 22066179 (SON) — Q on Aqua, ~24-36h
- **Memory** : `.orchestrator/memory/{boss,department,engineer}.md`

## 9. Status historique (anciennes missions clôturées)

### Phase H — E.2 abandonnée 2026-05-27 (Strategy A canal OBS in-model)
- **M_H0_smoke** : YELLOW. Pipeline OK, toggle plat 1 epoch (attendu)
- **M_H1 preflight** : GREEN. val_mse 0.097 sur 32 cases × 1 epoch
- **M_H1 full** : YELLOW. 30 epochs, val_mse plateau 0.119 (~baseline), **toggle_delta 3.4e-6 à epoch 29 — canal OBS jamais activé**
- **M_H+ Axe 1 DEM** : GREEN. 86 → 186 tiles Copernicus
- **M_H+ Axe 2 ERA5 multi-season** : GREEN. MAM/JJA/SON 2023 hourly stores 1.2GB chacun
- **M_H+ Axe 3 multiproc** : GREEN. 5× speedup smoke validé
- **Re-inference v2 production** : jobs 22066177/78/79 lancés 2026-05-27, ETA 24-36h

### Phase G — DONE 2026-05-26
- M_G0-M_G9 : pipeline OBS unifié + surrogate inference + audit stratifié livrés
- Détails : voir backup `.orchestrator/mandate.md` git history pre-2026-05-27

### Phase F / step-back / V0 statu quo — DONE 2026-05-20
- M1-M17 : ablation OFAT, multi-hill, V0 statu quo retenu
- Détails : `data/validation/ablation_multi_hill/REPORT.md` §10
