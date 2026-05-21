# MANDATE — Phase G : Extension du dataset OBS (1000+ stations) + inférence surrogate v2 aux stations

Boss source of truth. Departments read only the relevant mission slice.

## 0. Contexte (résumé pour reprise rapide)

**État après step-back session 2026-05-18/20** (voir
`data/validation/ablation_multi_hill/REPORT.md` §4 + §9 + §10) :

- **Stack CFD = V0 statu quo** (= Venkatraman Perdigão WES 2023 adapté). Pas
  de regen 9k. 9252 grid.zarr déjà sur Aqua sous
  `/scratch/maitreje/dsw/training_v2/`.
- **Surrogate v2 entraîné** : multiples variantes ViT/FNO sous
  `~/dsw/data/models/surrogate_v2_*/best.pt`. À identifier en M_G0 lequel
  est le best officiel (val_loss, type d'architecture).
- **Verdict M16/M17** : la voie d'amélioration n'est PAS dans le stack CFD
  ni dans un nouveau patch BC, mais dans (G) extension du dataset OBS
  pour permettre (H) ML correction stratifié physiquement honnête.
- **Dataset OBS actuel** : 7 sites ICOS tall-tower 2020 + Perdigão IOP 2017
  (48 tours, `data/raw/perdigao_obs.zarr` déjà ingéré). M17 a montré que
  N=7 sites = climatologie pas physique : top features XGBoost = lat/lon
  /elev, CFD feature importance 0.04.
- **Seuil empirique** pour ML correction utile : ~30-50 sites min pour
  LOSO honest, cible ~1000+ stations pour vraie stratification
  (terrain × height × wind class × season).

## 1. High-level objective

Constituer un **dataset OBS multi-sources couvrant la péninsule ibérique +
France continentale** (1000+ stations), aligné temporellement avec un
système d'**inférence surrogate v2 aux coordonnées arbitraires**, pour
produire un large set de pairings `(station, timestamp) → (U_obs,
U_surrogate)` permettant en sortie de Phase G :

1. Mesurer le biais surrogate v2 vs OBS **sur 1000+ stations stratifiées**
   (vs 7 ICOS actuels).
2. Confirmer ou infirmer le verdict M16 (`U_cfd = 0.54·U_obs + 1.88` est
   physique RANS + wall functions, pas BC tuning) sur statistiques larges.
3. **Débloquer Phase H** : DNN bias correction stratifié sur dataset
   suffisant pour LOSO honest.

**Done** = (a) >1000 stations FR/ES/PT ingérées au format Zarr unifié ;
(b) ≥50k pairings `(station, timestamp)` extraits ; (c) pipeline
`infer_surrogate_at_coords.py` opérationnel ; (d) audit comparison
stratifié `OBS vs surrogate` produisant les tables/figures qui
conditionnent la décision Phase H.

## 2. Out of scope

- Phase H (DNN bias correction stratifié) : décision portée par G, mais
  exécution = mission suivante.
- Phase I (re-simulation alpine summit avec domaine 10×10×5 km).
- Modification du stack CFD ou regen 9k.
- Re-entraînement du surrogate v2 (utiliser le best.pt existant).
- Sources OBS au-delà de FR/ES/PT (extension à Europe entière déferrée).
- Stations à pas de temps subhoraire (filtrer hourly comme commun
  dénominateur).
- Calibration et correction des biais ERA5 amont (hors scope, mais à
  noter si découvert).

## 3. Constraints

- **Langue** : communication FR, code/commits EN, conventional commits.
- **Données OBS** : Zarr 3.x avec schema unifié multi-sources (cf. M_G5).
  Hourly cadence, vent 10 m AGL minimum. Garder source, station_id, lat,
  lon, elev, et toutes les hauteurs disponibles.
- **APIs** : respecter rate-limits SYNOP/AEMET/IPMA. Préférer downloads
  bulk historiques (CSV/NetCDF) aux API live quand disponible.
- **Stockage** : OBS Zarr sous `data/raw/obs_{source}_{country}.zarr/`
  ou Zarr unifié `data/raw/obs_unified.zarr/`. Choix M_G5.
- **Apptainer + Aqua** : surrogate v2 inférence sur Aqua H100 (env
  `fuxicfd`). Pas de mpirun login node.
- **Confirmation** : confirmer avec user avant qsub massif, scp gros
  volumes (>1 GB).
- **Pièges connus** : (1) ne pas faire confiance à un audit OBS sur <50
  stations (climatologie cachée) ; (2) toujours stratifier (terrain ×
  height × wind class × season) ; (3) WC tif coastal bug pour stations
  côtières — vérifier WC AVANT d'utiliser un site.
- **File budget** : soft ≤500 LOC, hard ≤700 LOC. Tout nouveau script
  ingestion en module séparé.

## 4. Architecture decisions (ADRs)

| Date       | Decision                                                          | Rationale                                              |
|------------|-------------------------------------------------------------------|--------------------------------------------------------|
| 2026-05-20 | Phase G priorisée plutôt que regen 9k                             | Verdict M17 : V0 statu quo défendable, levier = OBS+ML |
| 2026-05-21 | Sources prioritaires : Perdigão IOP + SYNOP FR + AEMET ES + IPMA PT | Choix user : couvre 1000+ stations péninsule ibérique  |
| 2026-05-21 | Stratégie = inférence surrogate aux stations, pas re-simulation OF | Choix user : zéro nouveau OF, exploite surrogate v2 existant |
| 2026-05-21 | Zarr unifié multi-sources OBS (schema M_G5)                       | Permet stratification cross-source en aval             |
| 2026-05-21 | Surrogate v2 utilisé = best.pt à identifier M_G0                  | Plusieurs variantes ViT/FNO, choisir celle avec meilleur val_loss |

## 5. Missions

### M_G0 — Audit pipelines OBS + design alignement station↔surrogate (read-only)

- **Status**: planned
- **Goal**: Department lit le repo pour :
  (a) lister les pipelines ingestion OBS déjà présents
  (`ingest_perdigao_obs.py`, `ingest_icos.py`, autres) et évaluer leur
  réutilisabilité ; (b) identifier l'API/URL/format de SYNOP Météo France,
  AEMET Espagne, IPMA Portugal (formats, rate-limits, historicité) ;
  (c) identifier le best surrogate v2 sur Aqua (lister
  `~/dsw/data/models/surrogate_v2_*/best.pt`, lire les métriques
  d'évaluation si dispo) ; (d) proposer un schema Zarr unifié multi-sources
  pour M_G5 ; (e) proposer la définition d'un "pairing" station ↔
  timestamp v2 (quel timestamp surrogate utiliser pour quelle obs).
- **Allowed edit zones**: aucun (audit + design)
- **Exit criterion**: rapport ≤300 mots Department avec :
  (i) tableau sources OBS (nom, URL/API, format, n_stations estimées,
  période, vent dispo h_AGL, T_air dispo, complexité ingestion) ;
  (ii) chemin du best surrogate v2 + ses métadonnées principales (arch,
  val_loss, val sites, features attendues en input) ;
  (iii) schema Zarr unifié proposé ; (iv) algo de pairing station ↔
  timestamp ; (v) découpe en sous-missions ingestion.

### M_G1 — Pipeline ingestion Perdigão IOP 2017 unifié

- **Status**: planned
- **Goal**: Codex (via Department) audit `data/raw/perdigao_obs.zarr`
  (déjà ingéré) et l'adapte au schema unifié M_G5. Exposer 48 tours,
  multi-heights (10/20/40/60/80/100 m), hourly aggregation depuis 30-min
  raw. Filtrage qualité (despiking, nan_ratio < 0.3).
- **Allowed edit zones**:
  - `services/data-ingestion/ingest_perdigao_obs.py` (refactor)
  - `data/raw/obs_unified_perdigao.zarr/` (nouveau ou existant à
    convertir)
  - `test/scratch/`, `scratch/`, `tmp/`
- **Exit criterion**: Zarr Perdigão au schema M_G5 + smoke read OK
  (read 5 stations × 10 timestamps × multi-heights).

### M_G2bis — Pipeline ingestion NOAA ISD (remplace M_G2 SYNOP MF RED + M_G4 OGIMET RED)

- **Status**: planned (lancée 2026-05-21 après pivot)
- **Goal**: Codex implémente `services/data-ingestion/ingest_noaa_isd.py` qui télécharge l'archive NOAA Integrated Surface Database (ISH format), filtre les stations EU (FR/ES/PT/IT/DE/etc), et écrit `data/raw/obs_unified_noaa_isd.zarr` au schema unifié défini dans `.orchestrator/mandate.md` §7.
- Source : ~12 000 stations EU, hourly, vent 10 m AGL, T2m, RH, période 1973→présent
- Endpoint : `ftp://ftp.ncdc.noaa.gov/pub/data/noaa/<YYYY>/<USAF>-<WBAN>-<YYYY>.gz` (ISH format)
- Stations metadata : `ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-history.csv`
- **Allowed edit zones**:
  - `services/data-ingestion/ingest_noaa_isd.py` (nouveau, ≤500 LOC)
  - `services/data-ingestion/utils/isd_parser.py` (helper ISH parser, ≤300 LOC)
  - `data/raw/obs_unified_noaa_isd.zarr/`
  - `tmp/noaa_cache/`
- **Exit criterion**: smoke = ingestion 3 mois 2023 Q1 sur ≥30 stations FR/ES/PT, zarr lisible via `shared.obs_io.read_obs`.

### M_G2 — Pipeline ingestion SYNOP Météo France (RED 2026-05-21, blocker externe)

- **Status**: planned
- **Goal**: Codex implémente
  `services/data-ingestion/ingest_synop_meteofr.py`. Source =
  Données Publiques Météo France (téléchargement bulk historique
  recommandé). ~150 stations, hourly, vent 10 m AGL, T2m. Format de
  sortie = schema M_G5.
- **Allowed edit zones**:
  - `services/data-ingestion/ingest_synop_meteofr.py` (nouveau)
  - `services/data-ingestion/utils/` (helpers communs si besoin)
  - `data/raw/obs_unified_synop_fr.zarr/`
  - `test/scratch/`, `scratch/`, `tmp/`
- **Exit criterion**: ≥120 stations FR ingérées, période 2016-2023 min,
  Zarr lisible, ≥10⁶ pairings disponibles.

### M_G3 — Pipeline ingestion AEMET Espagne

- **Status**: planned
- **Goal**: Codex implémente
  `services/data-ingestion/ingest_aemet_es.py`. API AEMET OpenData (clé
  API requise — vérifier env). ~600 stations, hourly, vent 10 m AGL.
- **Allowed edit zones**:
  - `services/data-ingestion/ingest_aemet_es.py` (nouveau)
  - `data/raw/obs_unified_aemet_es.zarr/`
  - `.env.example` (si nouvelles clés API)
  - `test/scratch/`, `scratch/`, `tmp/`
- **Exit criterion**: ≥400 stations ES ingérées, period min 2018-2023,
  Zarr lisible.

### M_G4 — Pipeline ingestion IPMA Portugal

- **Status**: planned
- **Goal**: Codex implémente
  `services/data-ingestion/ingest_ipma_pt.py`. IPMA API ou téléchargement
  bulk. ~200 stations, hourly, vent 10 m AGL.
- **Allowed edit zones**:
  - `services/data-ingestion/ingest_ipma_pt.py` (nouveau)
  - `data/raw/obs_unified_ipma_pt.zarr/`
  - `.env.example` (si nouvelles clés API)
  - `test/scratch/`, `scratch/`, `tmp/`
- **Exit criterion**: ≥150 stations PT ingérées, period min 2018-2023,
  Zarr lisible.

### M_G5 — Schema Zarr unifié multi-sources + helper read API

- **Status**: planned
- **Goal**: Codex (via Department) crée
  `shared/obs_io.py` avec :
  - schema Zarr unifié documenté (groupes : `stations/`, `data/`,
    `metadata/`).
  - Variables : `u10, v10, t2m, q2m (optionnel), wind_speed_10m,
    wind_dir_10m, timestamp` ;
  - Coords : `station_id` (bytes), `lat`, `lon`, `elev`, `height_obs`,
    `source` (string : perdigao/synop_fr/aemet_es/ipma_pt/icos).
  - Helpers : `merge_obs_sources(*zarrs) → unified_zarr`,
    `read_obs(zarr, station_ids=None, time_range=None) → DataFrame`.
- **Allowed edit zones**:
  - `shared/obs_io.py` (nouveau)
  - `data/raw/obs_unified.zarr/` (merged store)
  - `test/scratch/`
- **Exit criterion**: schema documenté ; merge des 4 sources opérationnel ;
  `read_obs` retourne DataFrame propre avec ≥1000 stations.

### M_G6 — Pipeline extract_v2_input_at_coords.py

- **Status**: planned
- **Goal**: Codex implémente
  `services/module2b-surrogate/extract_v2_input_at_coords.py`. Étant donné
  `(lat, lon, timestamp)`, produit le `grid.zarr/input` au format attendu
  par le surrogate v2 :
  - terrain DEM 180×180 (SRTM 30 m, recadré, lissé)
  - z0_eff 180×180 (ESA WorldCover 2021)
  - lat scalaire (Coriolis)
  - ERA5 3×3 × {u,v,T,q} × N_pressure_levels au timestamp donné
  - ERA5 surface 3×3 × {t2m, d2m, u10, v10}
  - inflow_meta (timestamp, wind direction)
  - Pas besoin du `target/` (on inférera, pas validera contre OF).
- **Allowed edit zones**:
  - `services/module2b-surrogate/extract_v2_input_at_coords.py` (nouveau)
  - `services/module2b-surrogate/utils/` (si helpers communs)
  - `test/scratch/`, `scratch/`, `tmp/`
- **Exit criterion**: smoke test : génère grid.zarr/input compatible
  pour 5 coords arbitraires de stations Perdigão IOP, dimensions OK,
  norm OK.

### M_G7 — Inférence surrogate v2 + extraction U/W au point station

- **Status**: planned
- **Goal**: Codex implémente
  `services/module2b-surrogate/infer_at_stations.py`. Pipeline :
  (1) lit le Zarr OBS unifié, sélectionne `(station_id, timestamp)`
  pairings de l'audit M_G8 ;
  (2) appelle M_G6 pour chaque pairing → grid.zarr/input ;
  (3) charge le best surrogate v2 (identifié M_G0) ;
  (4) inférence sur H100 (batched, multi-pairing par batch) ;
  (5) extraction U/V/W au voxel central (90, 90, k_obs) où k_obs
  correspond au z = elev_station + height_obs (z terrain-following) ;
  (6) écriture `data/inference/surrogate_at_stations.parquet` avec
  colonnes `(station_id, timestamp, source, u_pred, v_pred, w_pred,
  speed_pred, u_obs, v_obs, speed_obs)`.
- **Allowed edit zones**:
  - `services/module2b-surrogate/infer_at_stations.py` (nouveau)
  - `configs/hpc/infer_at_stations.pbs` (nouveau)
  - `data/inference/`
  - `test/scratch/`
- **Exit criterion**: parquet >10000 pairings, smoke OK sur ≥100
  pairings Perdigão.

### M_G8 — Audit comparison surrogate vs OBS stratifié

- **Status**: planned
- **Goal**: Department produit
  `services/validation/audit_surrogate_vs_obs.py` + figures + report
  markdown. Stratification :
  - `class_topo` : plain / foothill / mountain / summit / coastal
  - `height_bucket` : 10 / 20 / 50 / 100 m AGL
  - `wind_class` : low (<3) / mid (3-7) / high (>7) m/s
  - `season` : winter / spring / summer / autumn
  - `climate_zone` (Köppen : Cfb / Csa / BSk / etc.)
  Métriques par strate : MAE, RMSE, bias, p10/p90 ratio CFD/OBS.
  Comparer aussi vs ERA5_U10 baseline.
- **Allowed edit zones**:
  - `services/validation/audit_surrogate_vs_obs.py` (nouveau)
  - `data/validation/phase_G_obs_audit/` (CSVs, figures, report)
- **Exit criterion**: report ≤500 mots Boss + figures stratifiées +
  CSVs. Décision Phase H = GO / NO-GO / RESCOPE.

### M_G9 — Documentation + commit (Boss-only) + décision Phase H

- **Status**: planned
- **Goal**: Boss filtre les memory candidates, commit conventionnel,
  prend décision Phase H avec user.
- **Allowed edit zones**: Boss-only.

## 6. Mission graph

```
M_G0 ──► M_G5 ──► [M_G1, M_G2, M_G3, M_G4] (parallèles) ──► M_G5 merge
  │                                                                │
  └───► M_G6 ─────────► M_G7 ─────────► M_G8 ─────────► M_G9
```

Notes ordre :
- M_G0 est le go/no-go : valide la faisabilité APIs + schema avant tout.
- M_G6 peut se faire en parallèle de M_G1-G4 (indépendants).
- M_G7 attend M_G5 merge OK + M_G6 OK + best surrogate identifié.
- M_G8 attend M_G7 parquet.

## 7. Decisions taken after M_G0 (2026-05-21)

### Choix du best surrogate v2

- Path : `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
- Arch : ViT base + residual + S4 + geo + AGL 24 niveaux denses (0-100 m) + ERA5 surface branch
- val_loss=0.5843, **val_mse=0.121** (le plus bas des variantes)
- Code Aqua : `~/dsw/services/module2b-surrogate/src/{dataset_v2_vit, model_vit_v2}.py`
- Inputs attendus : `terrain[180,180]` + `z0_eff` scalaire + `lat` + ERA5 3×3 pressure (4 vars × N_p) + ERA5 surface 3×3 (t2m/d2m/u10/v10) + AGL levels.
- Output : `(5, 180, 180, N_agl)` = (u,v,w,T,q) normalisés.
- Alternative ViT-Large (`surrogate_v2_vit_large_resid_s4_geo_agl`, val_loss=0.4966) tenue en réserve si benchmark Phase G montre meilleure transférabilité.

### n_stations révisées (sources OBS)

| Source | n_stations effective | Période effective | Statut ingestion |
|---|---:|---|---|
| Perdigão IOP 2017 | 41 (vs 48 initial) | 2017-05→06 | déjà ingéré, refactor schema |
| SYNOP Météo France | ~62 | 1996→présent, 3h | bulk CSV.gz simple |
| AEMET Espagne | ~250 | 2008→présent, hourly | clé API + cache disque |
| IPMA Portugal (open live) | ~120 | 24-72h glissantes | open insuffisant |
| IPMA Portugal (OGIMET) | ~120 | 2018-2023 SYNOP décodés | fallback indispensable |
| ICOS (déjà ingéré) | 7 tall-tower | 2020 + 2022 | ré-aligner schema |
| **Total** | **~480** | 2018-2023 commun | — |

~480 stations est sub-1000 du brief initial, mais **>10× le N=7 actuel** et **>10× le seuil empirique 30-50 sites pour LOSO honest**. Acceptable pour débloquer Phase H. Si M_G8 montre un effet plafond climat, élargir à Europe entière en Phase G+ (mission ultérieure).

### Découpe missions revue

- **M_G1.5 ajoutée** : ré-aligner les 7 sites ICOS déjà ingérés sur le schema M_G5 (sinon baseline perdue). Trivial, à coupler avec M_G1.
- **M_G4 redéfinie** : IPMA-live + **OGIMET-archive fusion**. IPMA open ne donne que 24-72h, donc OGIMET (synops décodés gratuits) est indispensable pour l'archive 2018-2023.
- **M_G3 batchs régionaux** : AEMET rate-limit 60 req/min → ingestion en 5 batchs régionaux séquentiels (Norte, Centro, Sur, Baleares, Canarias).

### Stratification timestamps (pairing)

Plutôt que tous les hours OBS dispo, **stratification 4 saisons × 3 wind_class × 4 heures synoptiques (00/06/12/18 UTC) = 48 cellules** avec 30 timestamps tirés / cellule / station → ~1440 pairings/station × ~480 stations = **~690 000 pairings** au global. Coût compute raisonnable sur 1 H100 (~quelques heures inférence batchée).

### Schema Zarr unifié `data/raw/obs_unified.zarr/` (validé)

```
stations/
  station_id (S16), lat, lon, elev, source (S16), country (S2), z0_class_wc (int8)
heights/
  height_m (H,) : [10,20,40,60,80,100], NaN si non dispo
data/  chunks=(time=720, S=1, H=-1)
  u, v, wind_speed, wind_dir, t2m, rh  (T, S, H)  float32 NaN-padded
coords/
  time (T,) int64 ns UTC hourly
```

NaN-padding sur H : SYNOP/AEMET/IPMA → seul `H=10 m` rempli ; Perdigão → 6 hauteurs ; ICOS → tour-spécifique.

## 8. Pointers

- Verdict V0 : `data/validation/ablation_multi_hill/REPORT.md` §4 + §9
- Mémoire boss : `.orchestrator/memory/boss.md` (VERDICT FINAL session)
- Surrogate v2 path : `~/dsw/data/models/surrogate_v2_*/best.pt` (Aqua)
- Dataset v2 grid.zarr : `/scratch/maitreje/dsw/training_v2/<site>_case_tsNNN/`
- ERA5 europe (déjà ingéré) : `data/raw/era5_europe.zarr`
- Module 3 precip QM stratifié (pattern à reproduire pour Phase H) :
  `services/module3-precip/`, `data/models/precip_correction/qm_stratified.npz`
- Existing ingestion : `services/data-ingestion/ingest_perdigao_obs.py`,
  `ingest_icos.py`, `ingest_era5_europe.py`
- Memory : `.orchestrator/memory/{boss,department,engineer}.md`
- Auto-memory MEMORY.md: `~/.claude/projects/-Users-guillaume-Documents-Recherche-downscalewind/memory/`

## 9. Status historique (anciennes missions clôturées)

- **M1-M5** (z0_treatment canary, 2026-05-16) **done** → `wc_capped_0.05`
- **M6-M10** (ablation OFAT multi-hill, 2026-05-18) **done** →
  best-stack V1 sur 2D
- **M11-M13** (Phase D V10 + Phase E V1, 2026-05-18) **done** → V1
  Phase E
- **M14-M17** (audit direction + step-back + OBS + ML POC, 2026-05-19/20)
  **done** → **V0 statu quo** est l'état de l'art ; investir Phase G
  plutôt que stack BC.
- Détails : `data/validation/ablation_multi_hill/REPORT.md` §10.
