# Department memory — DownscaleWind

## Config M_H1 era5_store = JJA-2023 → override pour tout eval 2017 IOP (2026-06-02, M_H'1c)

`devine_style_full_M_H1.yaml` a `era5_store: era5_europe_hourly_jja2023.zarr`. Tout eval sur un
AUTRE période (Perdigão IOP 2017) DOIT créer une config override pointant le bon store —
ici `era5_europe_spring2017_v2.zarr` (couvre 2017-05→06, d2m présent, ⚠️ Δt=360min/6h →
predictions constantes par blocs 6h, OK pour test SPATIAL au même timestamp, PAS pour
variabilité temporelle). L'Engineer Codex ne peut PAS voir ça (pas d'accès Aqua) → c'est au
Department de repérer le mismatch période↔store et de repointer le YAML + PBS.

How to apply : avant tout eval cross-période, vérifier que `era5_store` du YAML couvre bien
les timestamps cibles (`zarr coords/time` min/max). Mismatch silencieux = audit sur mauvaises
données ERA5.

## perdigao_obs.zarr schema + scp (2026-06-02, M_H'1c)

`perdigao_obs.zarr` : arrays `(time,site,height)=(2208,48,14)`, 10m exact à un h_idx donné,
~2400 fichiers chunk minuscules. scp -r lent (plusieurs min) mais fiable — vérifier via
`find -type f | wc -l` remote vs local. Local 12 MB ; était ABSENT d'Aqua avant M_H'1c.

## Codex sandbox n'a PAS de SSH Aqua → le Department fait le qsub (2026-06-02, M_H'1d″/1e)

L'Engineer Codex (run-engineer.sh) tourne dans un sandbox sans résolution DNS Aqua
(`Could not resolve hostname aqua.qut.edu.au`). Donc tout brief impliquant scp/qsub/ssh Aqua
doit séparer : (1) Engineer = écrit le CODE + dry-run CPU local + py_compile ; (2) Department
(qui a SSH) = scp vers Aqua + qsub + capture job ID + vérif cache. Ne pas demander à
l'Engineer de soumettre — il échouera silencieusement et rapportera "BLOCKED on Aqua".

## Dériver TOUTES les métriques du même obs apparié (2026-06-02, bug direction M_H'1d″)

`eval_devine_loso.py` calculait la VITESSE depuis `speed_obs` (déjà apparié dans le parquet)
mais la DIRECTION via un 2e appariement séparé (`_attach_obs_direction`, nearest-time obs-store)
+ ERA5 depuis les arrays forward → **+33° d'erreur fantôme quasi-constant sur TOUTES les
directions** (ERA5 affiché 74° au lieu du vrai 34°). Fix : dériver pred ET obs ET era5 de
la MÊME table appariée (parquet), une seule fonction `_dir_from_uv` partout (toute constante
de convention s'annule).

How to apply : pour tout eval multi-métrique, un SEUL appariement obs→pairing partagé par
toutes les métriques. Un 2e join "nearest-time" introduit un désalignement temporel qui gonfle
le signal. Red flag diagnostique : si une quantité SANS modèle (ERA5 brut vs obs) change de
valeur entre deux méthodes de calcul, c'est l'appariement qui est faux, pas le modèle.

## obs_uv_dir_check ≈ 0 ne valide PAS l'appariement (2026-06-02)

Un check `wind_dir_store == _dir_from_uv(u_store, v_store)` (mae 3e-6) prouve seulement la
cohérence INTERNE du store, pas que les bonnes lignes obs ont été appariées aux pairings.
Ne jamais conclure "métrique correcte" depuis ce check. Valider l'appariement = comparer une
quantité sans modèle (ERA5 vs obs) à une référence indépendante.

## SE ≠ std value : calcul rigoureux pour trancher noise vs signal (2026-05-29, Dept B M_H'1 audit)

Lors d'un epoch-to-epoch comparison MAE entre deux iterations training, ne PAS confondre :
- `MAE_std` = std de la distribution des erreurs absolues per-pairing (typiquement ~0.7 m/s pour wind speed)
- `SE on MAE_mean` = `MAE_std / √N` (variance d'estimation de la MAE moyenne)

Pour smoke v3 : val_ds = 6000 pairings, MAE_std ≈ 0.7 → **SE = 0.009 m/s** (pas 0.025 comme l'estimation rapide).
Le jump epoch 0→1 de 0.081 m/s correspond à **z=9σ**, pas 3.2σ.

Boss avait estimé 3.2σ et conclu "peut-être noise". Real signal était **massif et indubitable**.

How to apply : pour tout debate noise vs signal sur metric MAE/RMSE inter-epoch, calculer SE correct (std/√N) avant de conclure. Si N grand (5k+) et metric noise std connu (~0.5-1.0 wind speed), un Δ de 0.02 m/s est déjà 3σ.

## DEVINE τ=0.6/0.4 est empirique, pas théorique (2026-05-29, Dept B M_H'1 audit)

Le Toumelin et al. 2024 NPG utilise `τ=0.6` quand `obs≤pred` et `τ=0.4` quand `obs>pred`. La valeur choisie est **calibrée empiriquement pour AROME × Alpes**, pas théorique.

Dujardin & Lehning 2022 NPG sur même type de problème (downscaling wind mountain) utilise `τ=0.3 + ε=4.3 m/s` après "numerous trials" pour atteindre une prédiction sans biais. **DEVINE τ n'est pas une référence universelle.**

How to apply : pour un nouveau projet (NWP × climate × terrain) qui adopte DEVINE-style asymmetric loss, re-tuner τ via smoke court est légitime (et probablement nécessaire). 0.55/0.45 ou 0.5/0.5 sont des starting points valides.

## Dry-run 1-batch CPU avant qsub Aqua (2026-05-28, M_H'0 lesson)

L'Engineer M_H'0 a livré 5 fichiers train_v2_devine_style.py / dataset_v2_obs_centered.py
/ model wiring sans dry-run de validation. Résultat : **3 bugs en cascade**, chacun
détecté seulement par un qsub PBS Aqua, soit 3 cycles × 5-15 min wall :

1. `terrain_in_channels=2` hardcodé alors que checkpoint surrogate v2 base attend 4
2. Dataset `terrain_2d` ne produit que 2 channels (terrain + z0), pas les 4 attendus (slopes manquants)
3. `batch_size=16` → CUDA OOM sur A100 40GB (le M_H1 baseline tournait avec batch_size=4)

Chacun de ces 3 bugs aurait été détecté par un **dry-run 1-batch CPU** en ~30s :
```python
# pseudocode dry-run avant qsub :
ds = build_dataset(cfg)
batch = next(iter(ds))
print(batch['terrain'].shape)  # should match checkpoint contract
surrogate = build_frozen_surrogate(cfg)
pred = surrogate(*batch)  # catches channel mismatch + early OOM (if any)
print('dry-run OK, ready to qsub')
```

How to apply : tout futur Engineer brief pour mission ML training sur HPC doit imposer
un **dry-run CPU 1-batch avant qsub Aqua**, et l'inclure comme étape obligatoire dans
les validation commands. Économise des heures de cycle PBS sur des bugs détectables
en 30s.

## Diagnostic perf : profiler les patterns I/O, pas que les chunks (2026-05-25)

Quand un pipeline Zarr est anormalement lent (NOAA prod : 31 MB/h, ETA
80 h), le réflexe naturel est d'inspecter les chunks. C'est nécessaire
mais SOUVENT pas suffisant : le vrai bottleneck est typiquement dans le
**code append/write** qui peut faire des accès O(T) au lieu de O(1)
sur un slice.

Pattern de diagnostic :
1. Compter les chunks créés vs attendus → suspect si fragmenté
2. **Lire le code write_obs/append_obs** et chercher les loops sur
   `arr[i, j, :]` ou `arr[..., k]` qui itèrent sur une dimension
3. Vectoriser : `arr[slice, j, :]` (1 IO Zarr) au lieu de loop
   `for i: arr[i, j, :]` (T IO Zarr)

Sur NOAA prod : le chunks patch seul aurait donné un speedup ~10x ; le
vectorize patch a donné le 1000x manquant. Combiné : 60h → 30min.

How to apply : pour toute mission "Zarr ingestion is slow", spawner un
Engineer brief qui demande explicitement de (1) inspecter chunks ET
(2) profiler le code append pour patterns per-element. Ne pas se
contenter du premier suspect.

## M_G8 audit thresholds + strata axes (2026-05-23)

**Verdict thresholds (Phase H GO/NO-GO/RESCOPE)** :
- GO : MAE < 1.5 m/s ET |bias| < 1.0 m/s ET spread MAE class_topo > 0.5 m/s
  (= pattern stratifié exploitable par DNN)
- NO-GO : MAE > 3.0 m/s (= plafond saturé)
- YELLOW : N_pairings < 1000 (= dataset insuffisant pour LOSO honest)
- RESCOPE : autres cas, e.g. certaines strates exploitables seulement

**Strates minimales (6 axes + 2 croisements)** :
- class_topo (plain/foothill/mountain/summit), height_bucket
  (10/20/50/100), wind_class (low/mid/high), season (DJF/MAM/JJA/SON),
  era5_freshness (on_time/interpolated/far), source
- Croisements clés : topo×wind, season×height

How to apply : tout audit OBS vs surrogate doit utiliser cette grille
pour éviter explosion combinatoire et permettre verdict reproductible.

## M_G7 parquet upgrade requis pour audit ERA5 baseline (2026-05-23, M_G8)

Le parquet actuel de M_G7 n'inclut PAS `speed_era5_baseline` (vent ERA5
au point central du grid.zarr). M_G8 skip la comparaison vs ERA5
baseline sans cette colonne → décision GO/NO-GO perd un degré de
liberté.

Fix : dans `infer_at_stations.py`, après extraction du grid.zarr, lire
`era5_surface/u10, v10` au centre (i=1, j=1 du 3×3) et écrire
`u10_era5_baseline`, `v10_era5_baseline`, `speed_era5_baseline` dans
le parquet à côté de `speed_pred`.

How to apply : faire un PATCH ciblé sur M_G7 (mission M_G7+ ou M_G8.5)
avant le run production massif sur Aqua. ~30 LOC ajoutés.

## `append_obs_data` time-axis pattern (2026-05-21, M_G3)

Pattern d'appel optimal pour `shared.obs_io` quand on a un axe `time`
fixé à l'avance (cas batched ingestion API) :

1. Construire le full `time_array` (e.g. tous les hours de la période
   d'ingestion).
2. Appeler `create_obs_store(path, stations_df, heights_array,
   time_array=full_T)` une seule fois.
3. Pour chaque station, appeler `append_obs_data(path, source,
   station_id, time_array=same_T, data_dict=..., height_idx_map=...)`.
   L'helper match via `_positions_in_existing_times` (matches exacts,
   pas de resize), donc pas de réécriture du store.

How to apply : M_G2 (SYNOP), M_G4 (IPMA+OGIMET) et toutes les
ingestions Phase G+ doivent suivre ce pattern pour éviter des resize
en cascade qui multiplient le coût write.

## Rate-limit logging : timestamp début, pas fin (2026-05-21, M_G3)

Quand on log les timestamps des requêtes API pour audit rate-limit
(e.g. AEMET 60 req/min), logger le timestamp **de début de requête**,
pas de fin. Sinon la latence réseau fausse l'audit `Δt ≥ 1.0 s`. Calc :

```python
unix_ts_start = time.time() - (time.monotonic() - last_call_monotonic)
```

How to apply : tout pipeline ingestion avec rate-limit (M_G4 OGIMET 1
req/5s, futurs APIs externes) doit suivre ce pattern.

## OBS unified Zarr `H`-axis NaN-padding (2026-05-21, M_G5)

Le schema `data/raw/obs_unified.zarr/` axe `H` (heights) accommode des
sources hétérogènes (SYNOP/AEMET/IPMA = 10 m seul, Perdigão = 6 hauteurs,
ICOS = tour-spécifique) via NaN-padding. Le caller fournit un
`height_idx_map={height_m: idx_in_H}` à `append_obs_data()`, l'helper
écrit aux index correspondants et laisse NaN ailleurs. `read_obs` drop
les rows entièrement NaN sur les heights demandées.

How to apply : tout pipeline ingestion M_G1/G2/G3/G4 doit (1) initialiser
le store via `create_obs_store(path, stations_df, heights_array)` avec
le full set des heights couvertes par TOUTES les sources ([10, 20, 40,
60, 80, 100] est l'union sûre), puis (2) appeler `append_obs_data(...
height_idx_map={10.0: 0, 20.0: 1, ...})`. Helper validé via
test/scratch/test_obs_io.py.

## Surrogate v2 best per use-case (2026-05-21, M_G0)

Sur Aqua sous `~/dsw/data/models/surrogate_v2_*/best.pt` :

- `vit_large_resid_s4_geo_agl/best.pt` : val_loss=0.4966 (le plus bas),
  val_mse=0.2259. Bon pour volumes 3D.
- `vit_base_resid_s4_geo_agl100_k24_surface/best.pt` : val_loss=0.5843
  mais val_mse=**0.1212** (le plus bas !), 24 niveaux AGL FWI 0-100 m
  + ERA5 surface input. **Recommandé pour pairing stations à 10 m**.

How to apply : pour Phase G (inférence aux stations OBS), utiliser
le `_agl100_k24_surface`. Pour benchmarks volume entier 3D, prendre
`_vit_large_resid_s4_geo_agl`. Code Aqua :
`~/dsw/services/module2b-surrogate/src/{dataset_v2_vit,model_vit_v2}.py`.

## OBS multi-sources schema (2026-05-21, M_G0)

NaN-pad sur l'axe `heights/` car SYNOP/AEMET/IPMA n'ont que 10 m AGL
alors que Perdigão a 6 hauteurs et ICOS varie par tour. Ne pas créer
1 Zarr par source — merger sous `data/raw/obs_unified.zarr/`. Schema :

```
stations/ {station_id, lat, lon, elev, source, country, z0_class_wc}
heights/ {height_m (H,)}
data/ {u, v, wind_speed, wind_dir, t2m, rh} chunks=(time=720, S=1, H=-1)
coords/time (int64 ns UTC hourly)
```

How to apply : tout pipeline ingestion (M_G1/G2/G3/G4) doit écrire au
même schema → merge trivial en M_G5. Source obligatoire dans coord
`station/source` pour stratifier en M_G8.

## OBS sources caveats (2026-05-21, M_G0)

- **SYNOP Météo France** : dataset 90 "SYNOP essentielles OMM" sur
  donneespubliques.meteofrance.fr. CSV.gz mensuels, FTP bulk, no key,
  ~62 stations FR métro. Cadence 3h (pas hourly natif). Historique
  1996→présent.
- **AEMET Espagne** : opendata.aemet.es/opendata/api. Clé API gratuite
  (env). ~250 stations. Rate-limit 60 req/min, 2 GET en chaîne par
  requête (token + data). Ingestion en 5 batchs régionaux séquentiels.
- **IPMA Portugal** : api.ipma.pt n'expose qu'un open live 24-72h.
  Archive historique non-publique. **Fallback OGIMET obligatoire**
  pour archive 2018-2023 (synops décodés gratuits).

How to apply : tout pipeline ingestion d'une de ces sources doit
intégrer ses caveats spécifiques. Prévoir cache disque pour AEMET
(rate-limit) et OGIMET parser pour IPMA.

## WC audit (heterogeneous site selection)

- ESA WC 2021 class → z0 lookup (Wieringa/Davenport, meters):
  10 tree ~0.5–1.5 | 20 shrub ~0.05–0.1 | 30 grass ~0.03 | 40 crop ~0.05
  50 built ~0.5–1.5 | 60 bare ~0.005 | 70 snow ~0.002 | 80 water ~0.0002
  90 wetland | 95 mangrove | 100 moss/lichen ~0.005
- Discriminance metric: `log10(z0_max_class>5pct / z0_min_class>5pct)`.
  Useful target: ≥ 2 décades (= the tree↔bare contrast).
- Reject coastal sites: `water > 30%` is almost always a bbox-centering
  bug (cf. ct_d_fire_0170 Skiathos at 100% water).

## Source-of-truth status

`data/campaign/complex_terrain_v1/manifests/dataset_v2_status.csv` is
authoritative for what converged. Status values: `solved`, `diverged`,
`early_converged`, `mesh_failed`. Tiers: `gold`, `silver`, `rejected`.
Pre-filter on `case_status='solved'` + `tier in {gold, silver}`.

## Aqua read-only verification pattern

To check a case is usable on Aqua before selecting it:

```
ssh maitreje@aqua "
ls /scratch/maitreje/dsw/complex_terrain_v1/sites/<SITE>/case_ts014/constant/polyMesh/ &&
ls /scratch/maitreje/dsw/complex_terrain_v1/sites/<SITE>/case_ts014/constant/boundaryData/
"
```

Expected: `boundary faces neighbour owner points blockMeshDict` in
polyMesh, and `section_0` ... `section_7` + `terrain` in boundaryData.

## OFAT propre > silos cross-site, toujours (2026-05-18, M9)

Les décisions z0_wall=0.005 (canary wall_z0 sur ct_d_fire_0170) et
wc_capped_0.05 (canary z0_treatment sur ct_d_fire_0056) ont été
prises sur des silos cross-site. L'OFAT propre toutes-choses-égales
sur multi-hill (M9) montre que ces deux choix sont
**non-discriminants** (|Δ crop_mean| ≤ 0.023).

How to apply: pour toute future décision majeure (regen 9k, nouvelle
canary), exiger une ablation OFAT propre sur le même mesh / même
inflow avant adoption. Les silos cross-site sont OK pour
diagnostiquer un mode de fail (cf. M5 z0_treatment), mais pas pour
trancher la valeur de production.

## Cross-check rotation = consistency check (2026-05-18, M9)

V1↔V1n (best-stack à 270°W vs 0°N) ont des Δ ≤ 0.001 sur 8 stats.
Quand l'ablation est correctement isolée (même terrain, même mesh,
même seed numérique), la rotation pure n'introduit aucun bruit
mesurable. C'est un check de consistance peu cher (1 run de plus
par config).

How to apply: pour valider qu'un protocole d'ablation isole bien
le facteur d'intérêt (pas de fuite par direction implicite), refaire
1 variante à direction tournée. Si Δ > 0.05 sur une stat, il y a un
confondant directionnel à investiguer.

## multi_hill builder gap: boundaryData not copied (2026-05-18)

`build_terrain_canary.py --terrain-kind multi_hill` (M7 deliverable)
regenerates the STL + polyMesh from terrainBlockMesher, but does NOT
copy `constant/boundaryData/section_0..7 + terrain` from the
base-case. Without those, the mappedFile lateral BCs make
`simpleFoam` crash on startup. The fix lives in the PBS (`cp -r`
from base-case → variant case BEFORE running TBM). Do NOT patch the
builder until a future SPLIT mission — patches to a 1020 LOC file
are fragile.

## Audit multi_hill: per-case CLI, not --canary-dir (2026-05-18)

`audit_terrain_canary.py` has two modes:
- `--canary-dir <dir>`: generic audit_grid fallback (no PDF, no
  per-hill masks).
- `--grid-zarr <case>/grid.zarr --variant V* --terrain-kind multi_hill
  --output <csv>`: full multi-hill ablation audit (PDF + per-hill
  crest_k/lee_k masks).

For the ablation, ALWAYS use the second form, looped per variant,
then concatenate CSVs. The `--canary-dir` mode is for single-case
canaries (flat / ridge_cos2 / z0_treatment).

## Canary regeneration pattern on Aqua (2026-05-18)

For multi-variant canaries (e.g. multi_hill ablation, z0_treatment),
do NOT scp the locally-generated case directories. They have
placeholder STL/inflow. Instead:

1. Identify a known-good *base-case* already on Aqua (e.g.
   `/scratch/maitreje/dsw/complex_terrain_v1/sites/ct_d_fire_0056/case_ts014/`
   with best-stack BCs and real ERA5 inflow).
2. Run the builder CLI ON Aqua, passing `--base-case <real_path>`
   and `--variant V0..VN`. The builder COPIES polyMesh and
   boundaryData (never symlinks — Apptainer blocks).
3. PBS array `-J 0-N` submits one task per variant.
4. Local: only the audit CSV + figures are scp'd back.

The smoke output `scratch/multi_hill_smoke/` is local-only — never
uploaded as base-case.

## Phase B silos warning (2026-05-18)

When comparing canary results across the recovery plan, watch for
**silos cross-site / cross-mesh / cross-treatment**. Each canary
(top_BC, wall_z0, z0_treatment, terrain) changed BOTH the variable
under study AND the underlying site/mesh. The MANDATE §0 summary
table in `.orchestrator/mandate.md` is convenient but rows must NOT
be added or differenced as a clean ablation — they are not.

How to apply: any new Phase C / multi-hill ablation MUST fix
everything except the OFAT variable. Same mesh, same inflow, same
SFD. If the brief to the Engineer doesn't enforce this, the
Department brief is wrong.

## Apptainer caveat

Apptainer bind-mount only `case_dir → /home/ofuser/run`. Symlinks
to `/mnt/weka/...` DO NOT WORK inside the container — anything the
solver reads must be a real copy, not a symlink. The
`build_terrain_canary.py` builder enforces this for polyMesh +
section_*.
