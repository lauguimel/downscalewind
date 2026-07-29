# Boss memory — DownscaleWind orchestrator

## ✅ VERDICT diagnostic terrain-fix 2026-06-12 — bug RÉEL mais IMPACT MARGINAL, chiffres TIENNENT

Job 22309966 exit 0 (walltime 2h41, A100). Fix L83 `abs(floor(lon))` appliqué. Modèle M_I5 régime (best.pt ep2)
ré-évalué sur terrain CORRIGÉ ; 23 val lon<0 + Perdigão re-matérialisés.
- **Held-out val 106 stn (109685 pairings)** : corr MAE **1.4619** (vs zéro-terrain 1.464 = INCHANGÉ), bias +0.002,
  raw 1.9111. → **le 1.46 TIENT**, le bug n'a PAS déplacé le headline.
  - VAL lon<0 (23 stn, 15938 pair) : corr **1.3828** vs raw 1.8751 → correction marche TRÈS bien sur l'ouest aussi
    (zéro-terrain ≈1.396 → corrigé 1.383 = −0.013, marginal).
  - VAL lon>=0 (83 stn) : corr 1.4754 (inchangé, jamais buggé).
- **Pourquoi marginal** : stations lon<0 = pop PLAIN (Ibérie peu accidentée) → terrain-zéro ≈ terrain-plat-réel.
  Le bug remplaçait du plat par du plat. Perdigão (vrai relief 105-536m) = seule exception mais régime calme domine.
- **Perdigão centre (376 pair, 47 stn) sur VRAI terrain** : corr **2.30** (vs zéro ~2.19 = ~pareil) raw **1.31** (vs 1.37),
  bias +2.12. Par vent : <3 (n311=83%) corr 2.50/raw 1.30/bias+2.48 ; 3-6 (n60) corr 1.25≈raw 1.26 ; >6 (n5) corr 2.32<raw 2.70.
  → **le "mur Perdigão" n'était PAS un artefact terrain** : c'est la SUR-CORRECTION VENT FAIBLE (IOP 83% <3 m/s, brut
  déjà bon). **Narratif M_I4/M_I5 CONFIRMÉ, PAS confondu.** Vrai terrain ne sauve pas Perdigão (n'avait pas à : résidu = régime calme).
- **Steep-54 held-out M_I5 régime (job 22306888 exit 0, walltime 1h12, valide lon>0)** : HIGH-WIND >6 m/s (n10848)
  raw MAE 4.733 → corr **3.018 (−36%)**, bias −4.67→−2.57 ; >7 (n9735) raw 4.87→corr 3.10. → gain MASSIF vent fort
  CONSERVÉ par le régime (≈ M_I3 3.02), narratif régime confirmé hors Perdigão. `data/validation/phase_H_prime_M_I5c_steep54_windstrat`.
- **DÉCISION** : re-train complet NON justifié (impact <0.01 held-out, ~0.1 Perdigão). Fix L83 gardé. Option papier =
  re-générer dataset propre "par principe" (hygiène) mais gain scientifique nul. → REPRENDRE benchmark FuXi sur base saine
  (1.46 tient, 100m prêt). Décision user re-train-vs-avancer EN ATTENTE.
- ⚠️ Corrige le verdict "AFFECTED/INVALIDÉ" ci-dessous : bug réel (terrain zéro lon<0) mais SANS conséquence matérielle sur les conclusions.
- ✅ **PIPELINE CFD NOT-AFFECTED (prouvé empiriquement 2026-06-12)** : le dataset CFD qui a entraîné le surrogate v2
  utilise un code DIFFÉRENT et CORRECT — `run_multisite_campaign.py` `_open_srtm_tiles`(L114)/`extract_stl`(L154) :
  nommage `abs(lon)` (signe = lettre E/W, pas `floor(-lon)`) + **MOSAÏQUE de toutes les tuiles bbox** (merge L146),
  pas de floor-1-tuile → bords couverts. Vérif empirique terrain frozen `training_v2/*/grid.zarr` : **ct_d_fire_0106
  (lon −7.77 = MÊME régime W007/W008 que le bug Perdigão) relief RÉEL 185-500m std 69** ; ct_c_morpho_0075 (−2.97)
  533-1168m ; ct_g_paragliding_0100 (−5.39) 216-557m ; ct_f_wind_0132 (−9.76 côtier IE) −78/−39m réel ; contrôle
  Alpes (+11.47) 1491-2713m. AUCUN plat-zéro. → **SURROGATE V2 SAIN** (entraîné sur vrai terrain partout). Bug confiné
  au helper inférence M_G6 (post-freeze 2026-05-22). PAS de re-gen CFD ni re-train surrogate. coords/z = vrais centres mailles OF.

## 🔴 BUG terrain `_resolve_dem_path` (lon<0) = CONFIRMÉ AFFECTED 2026-06-11 (preuve empirique)

Trouvé via l'adaptateur FuXi (D2), tracé par investigation dédiée. **VERDICT : AFFECTED, haute confiance, PROUVÉ
empiriquement** (re-run helper + lecture caches disque, pas inféré).
- **Bug** : `services/module2b-surrogate/utils/inference_input.py:83` :
  `lon_idx = int(math.floor(lon)) if lon >= 0 else int(math.floor(-lon))`. Pour lon<0 devrait être `abs(floor(lon))`.
  East lon → OK. Integer west lon → OK par coïncidence. **Non-integer west lon → MAUVAISE tuile (1 à l'est)** :
  −7.73→W007 (devrait W008), −9.13→W009 (devrait W010). Tuile nommée par coin SO → la box tombe hors tuile →
  rasterio remplit le patch 180×180 de **0** (terrain plat zéro).
- **4 artefacts** (tous via build_one→extract_terrain_from_dem→_resolve_dem_path, dir `srtm_tiles/`) :
  - steep parquet : **NON** affecté (279 steep stns lon 6.8–18.35°E) ; 122 stns lon<0 sont en pop **plain** → OUI ces lignes.
  - cache M_I3 grid.zarr : **OUI** (terrain lu = 0.0 pour 7 stns Ibérie/PT : min=max=mean=std=0).
  - **held-out 1.46** : **OUI** — 23/106 val (21.7%) lon<0, entraînées+scorées terrain zéro → **1.46 INVALIDÉ pour ce ~22%**.
  - **Perdigão raw 1.37** : **OUI** — cache perdigao terrain=0.0 à lon −7.726 → **1.37 INVALIDÉ comme claim terrain**
    (DEM plat). ⇒ **TOUT le narratif Perdigão est CONFONDU** : M_H'1c/1f/1g (2.27/2.15/2.32), M_I4 "outlier sur-corrigé",
    M_I4 re-cadré "artefact vent faible", M_I5 régime Perdigão — TOUS calculés sur terrain ZÉRO. À REFAIRE.
- **Preuve empirique A/B** (re-run helper sur Aqua) : Perdigão prod(W007)=0/0/0 vs correct(W008)=105/536/286m ;
  Iberia W3.5=0 vs 541/666m ; contrôle 7.7°E identique. Zéros identiques lus dans les caches disque M_I3+Perdigão.
- **Exposition** : 122/529 stns (23.1%) lon<0 ; train 99/423 (23.4%), val 23/106 (21.7%) ; 107k/605k pairings (17.7%), TOUTES en plain.
- ✅ **NON affecté = steep-54 held-out (−18/−22%)** = LE résultat clé (correction généralise au raide) → INTACT (tout lon>0).
- **À RÉGÉNÉRER** : fix L83 (`abs(floor(lon))`) + mosaic bords (comme adaptateur FuXi déjà) ; re-matérialiser caches
  lon<0 (plain + Perdigão, ~107k) ; re-éval modèle ACTUEL (diagnostic ampleur) ; puis re-train M_I3/M_I5 si impact
  notable + re-évals + audit Perdigão.
- **Action 2026-06-11** : job éval Perdigão M_I5 **22306889 TUÉ** (tournait sur terrain zéro). steep-54 22306888 gardé (valide).
- **FuXi benchmark = en aval du fix** (comparer FuXi-bon-terrain vs nous-bon-terrain). Décision user remédiation EN ATTENTE.

## FuXi-CFD head-to-head benchmark = FAISABLE (recon 2026-06-11)

User veut inférer le MODÈLE FuXi-CFD sur NOS cas + scorer vs MÊMES obs (benchmark obs-vs-obs honnête,
le seul like-for-like). Recon GO-WITH-CAVEATS :
- **Identité** : "Reconstructing fine-scale 3D wind fields with terrain-informed machine learning",
  Lin Chensen et al., Fudan AI Innovation Institute, *Nature Communications* 17:3713 (2026-03-09),
  open access PMC13103448 / nature.com/articles/s41467-026-70562-5. DISTINCT du modèle météo global FuXi.
  = concurrent DIRECT, même tâche → benchmark le plus pertinent.
- **Poids PUBLICS** : ONNX 964 MB HF `linchensen/FuXi-CFD-model` (`model/fuxicfd_model.onnx` + `normalization/`
  + `inference_example/infer.py`). Dataset HF `linchensen/FuXi-CFD-dataset` 253 GB. Zenodo DOI
  10.5281/zenodo.18770845. **Licence CC BY-NC 4.0** (papier OK, PAS commercial → flag track startup).
- **I/O CORRIGÉ (repro DONE 2026-06-11)** : entrée (1,4,300,300), ordre canaux **[u_100m, v_100m, dem, roughness]**
  (vent EN PREMIER — la recon avait inversé). u/v 100m sur 9×9 → zoom bilinéaire scipy ×33 vers 300×300 (pas de
  physique). dem/z0 en MÈTRES sur 301×301 @30m (z0 matche WorldCover). Standardiser par groupe via `scaler_input.npy`.
  Sortie **(1,27,4,300,300)** = (level,var,y,x), vars [u,v,w,k] ; de-norm `pred*std+mean` via `scaler_output.npy`
  (27,4). **10m = niveau index 0** (PAS 1 ; off-by-one corrigé). speed = hypot(u[0,150,150], v[0,150,150]). ONNX <1s CPU.
- **REPRO leur exemple = GREEN** : poids `data/models/fuxicfd_official/` (onnx ~848MB), tourne sur leur inputs.npz
  fourni → reproduit leur outputs.npz (u/v MAE ~0.10, w/k ~0.04) = on maîtrise le pipeline. Contrat exact dans
  `scratch/fuxicfd_io_notes.md`. ⚠️ hf_xet stalle sur le gros fichier → `HF_HUB_DISABLE_XET=1`.
- **Domaine train = SE Chine** complexe, validé 3 tours EU (OPE/Torfhaus/Ispra, terrain DOUX). Nos Alps/Apennins
  = OOD plus dur → test équitable mais à flagger (eux Chine "à l'extérieur", nous EU "à domicile"). Neutraliser
  le reproche = aussi comparer sur LEURS sites valid EU.
- **Aqua /scratch/maitreje/fuxicfd/** = NOS UNet (pas les poids officiels) → télécharger 964 MB HF séparément.
- **Adaptateur = M** : par station, terrain 300×300 30m (SRTM+WC déjà là) + vent 100m 9×9 depuis ERA5 → ONNX
  → niveau-1 pixel central. **Risque #1 = entrée 100m** : ERA5 a u100/v100 NATIFS (re-ingest CDS = fidèle)
  vs dériver log-law (contestable). **Risque #2** OOD steep. **Risque #3** reproduire leur inference_example
  AVANT de croire aux chiffres (conventions/normalisation).
- **Décision user 2026-06-11** : entrée 100m = **ERA5 u100/v100 NATIFS** (ré-ingest CDS, fidèle) + **TOUT sur HPC Aqua**.
- **Plan en cours** : (a) repro GREEN ✅ ; (b) D1 ingestion ERA5 100m = **GREEN**, 5 jobs CDS chaînés : perdigao2017
  **22307411** (R, ~30min) → mam/jja/son/winter **22307412/13/14/15** (afterany, ~10-12h) → stores `era5_100m_*.zarr`
  sous `~/dsw/data/raw/`. Code dual-mode `ingest_era5_europe_hourly.py --surface-only --surface-vars` (NON commité)
  + PBS `configs/hpc/ingest_era5_100m.pbs`. u100/v100 = single-level léger (~95KB/7j). **DONE 2026-06-11 : 5 jobs
  exit 0, 5 stores `era5_100m_{perdigao2017,mam2023,jja2023,son2023,winter2223}.zarr` présents** (bien + rapide que
  ~10-12h, single-level léger) → 100m PRÊT pour scoring FuXi, plus un goulot ;
  (c) D2 adaptateur `fuxicfd_infer_at_stations.py` = **GREEN** (smoke 22308191 exit 0, 10/10 Perdigão, FuXi 10m
  sane mean 1.73 [0.55-4.21] — 100m=PROXY, pas un vrai score ; poids `~/dsw/data/models/fuxicfd_official/` +
  onnxruntime 1.26 dans fuxicfd ; split val 106 stn reproduit via `watertight_station_split` seed=42 sur
  `combined_steep_plain_v2.parquet` ; ⚠️ bug terrain lon<0 → entrée dédiée en haut ; ~20s/pairing CPU → array job
  pour le full) ; (d) scoring full held-out val
  (106 stn dont 54 steep) + Perdigão → tableau MAE FuXi vs nous (1.46) vs ERA5 vs MÊMES obs [run 22316789 TUÉ walltime 4h01 : FuXi ONNX sur
  **CPUExecutionProvider** (onnxruntime CPU-only dans fuxicfd, pas GPU) ~7.8s/pairing → 1616/3607 (45%) ; terrain
  caching OK (stations_cached↑) ; PAS de checkpoint → partiels perdus ; spam pthread_setaffinity (threads non bridés).
  Relancé avec checkpointing + threads bridés + échantillon réduit]. **2026-06-15 : édits perf FAITS localement
  (thread cap dans `FuxiRunner.__init__` de fuxicfd_infer_at_stations.py + checkpoint/200 + resume dans
  score_fuxi_vs_ours.py + PBS NPS12/PNPS10/walltime8h/OMP8 ; py_compile OK ; onnxruntime-gpu écarté = CUDA13 incompat).
  PENDING scp+qsub — Aqua DOWN (SSH Connection refused, 2e panne de la semaine) → surveillance retour Aqua armée** ; (e) comparer
  aussi sur leurs sites valid EU (OPE/Torfhaus/Ispra) pour neutraliser le reproche OOD.

## Site-selection lesson (2026-05-15)

**Before** picking a canary site, always filter sites.csv by
`case_status='solved'` AND `tier in {gold, silver}` on the relevant
`case_ts<NN>`. The freeze manifest
`data/campaign/complex_terrain_v1/manifests/dataset_v2_status.csv` is
authoritative. Approximately 2857 / 12109 cases are `rejected` (mostly
`diverged`) — Pop B (steep terrain) hit RANS convergence limits,
typically slope > 25° + elev > 2500 m.

Why: a Department's WC-heterogeneity audit alone will happily pick
beautiful alpine sites that never converged in v1.

How to apply: in any future canary selection, filter solved-and-gold
FIRST, then rank by WC heterogeneity / z0 contrast on the survivors.

## Case_status vocabulary

The `case_status` column uses `solved` / `diverged` / `early_converged`
/ `mesh_failed` — NOT `ok` / `fail`. The OF solver status JSON written
by `run_multisite_campaign.solve_case` uses `ok` / `solve_failed`.
Don't conflate them.

## Canary base case naming

z0_treatment canary base cases under `/scratch/maitreje/dsw/z0_treatment_canary/<SITE>_ts014/`.
PBS template lives under `configs/hpc/z0_treatment_canary_<SITE>_ts014.pbs`.
PBS submits a `-J 0-3` array (4 variants : wc, wc_cap_0.05, wc_cap_0.01, uniform_0.05).

## 2 m AGL anomaly is an audit bug, not physics (2026-05-18)

The `crop/inflow > 1.0` at 2 m AGL observed on flat canary (1.343 with
best-stack, all four z0_treatment variants on 0056) is an
`inflow_speed_at` normalization issue in
`audit_v2_teacher_wind.py` (and any audit script reusing the same
helper). It is NOT a physical over-acceleration near the wall.

How to apply: exclude 2 m AGL from OFAT decisions in M9 until the audit
script is fixed. Decisions should rest on 10/20/50/100 m AGL stats.

## Stack regen 9k : V8 retenu, pas V1 (2026-05-18, Phase D M10)

Phase D a confirmé : **V8 est le stack optimal**, pas V1.
- V8 = `inletOutlet top U + zeroGrad top p + pg_geo flip + z0_wall=0.005 + wc_capped_0.05`
- V8 @ 10m : crop_mean=0.666, flat_mean=0.651, crest_max=1.864
- V1 @ 10m : crop_mean=0.600, flat_mean=0.581, crest_max=1.170

V9 (control + pg_geo seul, sans z0/wc tuning) : crop_mean=0.632,
crest_max=1.892. V8 légèrement supérieur sur mean (+0.034), V9
marginalement supérieur sur crest_max (+0.028). V8 retenu.

**Levier réel = INTERACTION pg_geo × top_BC ouvert**. Top fermé (slip
+ p=0) bloque la dilatation verticale → la pression géostrophique
n'a plus où respirer → dynamique de relief écrasée. Ce que la Phase B
ridge 2D mono-orientation avait validé ne tient pas sur 3D
multi-orientation.

Avant la regen 9k : valider V8 sur 5-10 sites v2 réels diversifiés
(Pop A continental, topographies variées). C'est la **Phase E** à
ouvrir comme nouvelle mission orchestrator.

## Best-stack écrase la dynamique sur multi-hill (2026-05-18, M9)

L'ablation OFAT propre toutes-choses-égales sur multi-hill (3 collines
analytiques, mesh v2, inflow ERA5 ct_d_fire_0056) RETOURNE la décision
Phase B :
- V0 control : crop_mean=0.757, flat_mean=0.789, crest_max=1.40
- V1 best-stack : crop_mean=0.600, flat_mean=0.581, crest_max=1.17
- V8 -top entier : crest_max=1.86 (le plus haut)

**Top BCs (slip_top + p_top=0) écrasent la dynamique de relief**.
Phase B avait validé best-stack sur 1 ridge 2D mono-orientation —
en 3D multi-orientation, slip_top ferme la circulation top et bride
les accélérations sur crêtes.

**Seul pg_geo flip est un vrai gain** : Δ crop_mean = -0.107 quand
retiré. Confirmé robust par cross-check rotation 0°N
(V1↔V1n |Δ|≤0.001).

**z0_wall=0.005 et wc_capped_0.05 sont non-discriminants** (|Δ|≤0.023)
sur OFAT propre — les décisions M3/M5 silos cross-site ne tiennent
pas.

**Pour la regen 9k : NE PAS adopter V1 tel quel**. Stack candidat
réduit `pg_geo flip + wc_cap (or uniform)` SANS slip_top / sans
p_top=0 / sans z0_wall=0.005 — à valider Phase D sur sites v2 réels
avant fixation finale.

## Diagnostic stall trap: trust the JSON, not qstat (2026-05-18)

When monitoring a PBS array, the absence of qstat output (`Unknown Job
Id`) only means the array left the queue, NOT that tasks failed. Tasks
write their own `solve_status/*.json` AFTER their last successful step.
If a non-critical late step (e.g. matplotlib in audit) crashes,
`solve_status` stays at the last-written value ("running") even though
the solver/export both completed and `grid.zarr` is on disk.

**Verification protocol BEFORE declaring a task failed**:
1. `ls .../cases/<V>/grid.zarr` — does the data exist?
2. `tail log.simpleFoam` — does it end with `End / Finalising parallel
   run`? (Then it's a post-solve issue, not a solver fail.)
3. `cat solve_status/*.json` — what was the last status?

Only after all three say "failed" is the variant actually broken.
The M8 mis-diagnostic (declared 7/11 failed when 11/11 succeeded)
wasted ~1 h of orchestration time. Don't repeat.

## libgeostrophicPressureGradient .so warning is cosmetic (2026-05-18)

In simpleFoam runs with `codedSource` fvOptions (e.g. our
geostrophicPressureGradient), the warning
`Could not load .../dynamicCode/.../lib<name>_<hash>.so` printed
AFTER `End / Finalising parallel run` is an OF runtime cleanup
artifact. The solver converged. Do NOT treat it as a failure
signal.

## WC tif coastal-bug — global audit pending (2026-05-18)

`download_worldcover_per_site.py` is suspected to have a bbox-centering
bug for Mediterranean coastal sites: `ct_d_fire_0170.tif` (Skiathos)
came back 100% water. Other Mediterranean coastal sites in the 9k
campaign may share the bug. The fix and the global audit are
**out of scope** for the multi-hill ablation mission, but must be
opened as a separate task before any 9k regeneration.

How to apply: when selecting a base-case inflow for a canary, always
re-verify the WC class distribution of the site (look at the tif
or the `freeze_dataset_v2_status` row) BEFORE using it. Continental
sites (>50 km from coast) are safe by construction.

## build_terrain_canary.py LOC over ceiling (2026-05-18)

Post-M7 additions push `build_terrain_canary.py` to **1020 LOC**
(hard ceiling 700, soft 500). Department M7 flagged this for a future
SPLIT mission — split `multi_hill` mode into a sibling
`build_multi_hill.py` (or a `canary/` sub-package). NOT a blocker for
M8/M9 because the new code is structured in dedicated private
functions, but any *substantive* future change to this file MUST start
with the SPLIT, not a 5-LOC patch.

## Prior canary table is silo'd cross-site (2026-05-18)

The MANDATE §0 summary table mixes rows from ct_d_fire_0170_ts014 (WC
bug → effectively z0=2e-4), ct_d_fire_0056_ts014 (real WC), and pure
analytic flat/ridge derived from 0170 inflow. The increments
("slip_top → +0.04") cannot be added or compared as if from the same
ablation. This is exactly why M6→M9 ablation is being done on a
single multi-hill mesh with one inflow.

## Sources OBS Phase G — 2026-05-21 RED + pivot NOAA ISD

**Bilan smoke audit 2026-05-21** :

- **M_G2 SYNOP Météo France RED** : URL `donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop.csv` retourne une page HTML "Données non disponibles". URL invalidée par le provider. Code OK mais source bloquée.
- **M_G4 OGIMET RED** : parser HTML ne trouve plus la table dans les pages OGIMET (format serveur changé entre l'écriture de la spec et l'exécution). lxml installé OK mais structure HTML attendue absente.
- **Open-Meteo Historical Weather API = ERA5 reanalysis sur grille**, pas obs in-situ. **NE PAS** l'utiliser comme proxy OBS pour benchmark surrogate v2 → circulaire (le surrogate downscale déjà ERA5).
- **Open-Meteo Forecast API = IFS** : non-circulaire mais pas obs in-situ non plus. Utilisable comme baseline intermédiaire si besoin.

**Pivot 2026-05-21 = NOAA ISD** (Integrated Surface Database) :
- `ftp://ftp.ncdc.noaa.gov/pub/data/noaa/` ou Climate Data Online API
- ~12k stations EU, SYNOP-derived, vraies obs in-situ
- Hourly, multi-decade, vent 10 m AGL + T + RH
- Format ISH (Integrated Surface History) à parser

How to apply : avant d'investir dans un nouveau pipeline d'ingestion "obs", vérifier que la source est bien in-situ (pas reanalysis). Le critère discriminant : si la source fournit des données à toute coordonnée arbitraire = c'est du modèle/grille, pas des stations.

## Erreur Boss rm hâtif (2026-05-21) — lesson learned

J'ai supprimé `_obs_unified.py` et `utils/obs_zarr_writer.py` en pensant qu'ils étaient des orphelins du 1er kill M_G3, sans avoir vérifié les imports. **Ils étaient en réalité utilisés par les 4 scripts ingest_* livrés par M_G1/G1.5/G4**. Heureusement récupérables depuis les engineer logs (Codex apply_patch contient le code complet).

How to apply : avant tout `rm` de fichiers non-tracked, exécuter
`grep -r "filename_without_ext" --include='*.py'` pour vérifier
qu'aucun import ne les référence. CLAUDE.md warning "investigate before
deleting" était précisément pour ce cas.

## Diagnostic M_H'1d′ direction 2026-06-01 — faits mesurés (corriger erreurs antérieures)

**Erreurs Boss de la session à NE PAS reproduire** :
- J'ai d'abord affirmé "cache 8755/49%" : FABRIQUÉ (commandes annulées en cascade jamais
  exécutées). Le vrai cache `phase_H_prime_M_H1_jja_cache` = **60971 dirs présents / 63022
  pairings parquet = 97%**. Les 2051 manques = dates bordure mai/sept (hors JJA). Toutes les
  214 stations couvertes (val-43 : 12061 pairings cachés).
- Leçon : ne JAMAIS écrire un chiffre sans l'avoir lu dans un tool_result réel. Une commande
  bash qui échoue (ex `conda: command not found` en SSH non-interactif → exit 127) **annule
  en cascade tout le batch parallèle** (Write/Edit/Agent inclus) → rien n'est persisté, c'est
  un faux sentiment de progrès. Charger conda en SSH : `module load Miniconda3/24.9.2-0;
  eval "$(conda shell.bash hook)"; conda activate fuxicfd`.

**Parquet `noaa_jja2023_v2.parquet` (63022 rows, 214 stations) = sortie RAW surrogate v2**
(inférence M_G7). Colonnes : `speed_pred,u_pred,v_pred,w_pred` (RAW, PAS corrigé ANN),
`speed_obs,u_obs,v_obs,wind_dir`? non — `u_obs,v_obs,speed_obs`, `speed_era5_baseline,
u10_era5_baseline,v10_era5_baseline`, `era5_time_delta_minutes` (médiane 0 = on-time).
→ Le chiffre CORRIGÉ M_H'1a n'est PAS dans le parquet → eval forward GPU nécessaire.

**Direction (surrogate RAW vs OBS, val-43, mesuré)** :
- surrogate mae_dir **43.3°** med 27.4° biais **+16.4°** (rotation horaire systématique)
- ERA5 mae_dir 34.4° med 19.7° → **le surrogate raw DÉGRADE la direction vs ERA5**
- 22/43 stations val médiane dir >30° ; 15% des pairings >90°
- **Nuance user (importante)** : "différent d'ERA5 ≠ faux" — la déflexion relief-aware du
  surrogate est attendue/désirable ; SEULE la comparaison vs OBS tranche. Ici vs OBS le raw
  est pire qu'ERA5 → déflexion mauvais sens/trop forte. À confirmer sur le CORRIGÉ.
- **Structurel** : ANN M_H'1a entraîné loss VITESSE seule (pas de terme angulaire) → aucune
  incitation à corriger la direction, et peut la dégrader en bougeant u,v pour la norme.
  DEVINE 2024 avait DEUX réseaux séparés (ANN_speed + ANN_direction).

**Vitesse (surrogate RAW vs OBS, val-43)** : mae 1.60 biais -1.06 ; classe high>6m/s mae 2.96
biais **-2.88** (compression slope-0.59 connue). NB : 12488 pairings val ⊃ 3015 val training
→ pas exactement apples-to-apples avec le 1.812 mandate.

**Décision user 2026-06-01** : faire l'eval GPU corrigé (réparer eval_devine_loso.py : filtre
cache au lieu du garde-fou tout-ou-rien). Mesurer dir corrigée vs OBS + déflexion corrigé−ERA5
stratifiée terrain pour distinguer relief-aware utile vs bruit.

## M_H'1d″ eval corrigé GREEN (vitesse) 2026-06-02 — job 22162041 exit 0, walltime 1h09

VITESSE CONFIRMÉE DÉCISIVEMENT (43 stations held-out, channels=4, reproduit exactement
le training M_H'1a) :
- val mae_corrected **1.223** / raw 1.812 / era5 1.315 → **−32.5%**, biais +0.008
- train mae_corrected 1.236 / raw 1.796 → **−31.2%** (gap train↔val quasi nul = pas d'overfit)
- Le corrigé BAT ERA5 (1.223 < 1.315) sur stations jamais vues.
- Par classe vent (val) : faible<3 corr 1.07 / raw 0.88 / era5 0.86 (correction sur-corrige
  le calme) ; moyen 3-7 corr 1.08 / raw 1.76 / era5 1.25 ; fort>7 corr **1.75** / raw 3.58 /
  era5 2.25 (divise l'erreur par 2, bat era5). → gain là où ça compte (vent fort = fire weather).
- Cache inchangé 60971. Couverture 98.2% (kept 60971/62062, dropped mai/sept hors JJA).

DIRECTION = MODULE EVAL BUGGÉ, NE PAS UTILISER ces chiffres :
- eval donne raw 67° / corr 71° / **era5 74°** ; or recalcul direct parquet (même convention
  meteo from-dir) = raw 43° / **era5 34°**. ERA5 (aucun modèle, juste u10/v10) ne peut valoir
  34 ET 74 → bug convention/appariement dans le volet direction de eval_devine_loso.py.
  Corrélation déflexion-terrain (~0, pearson 0.01/−0.08) en hérite → non fiable.
- FIABLE (recalcul direct val-43 vs OBS) : surrogate RAW 43° DÉGRADE vs ERA5 34°. 22/43 >45°.
- **La direction du modèle CORRIGÉ reste non mesurée proprement** (le seul outil qui l'avait
  est buggé). Question "ANN speed-only aide/empire la direction ?" → OUVERTE.

Verdict : SPEED-only validé production-scale honnête (paper NatComms fire : −32% MAE summer,
bat ERA5, surtout vent fort). Direction = (a) soit fixer le bug direction de eval_devine_loso.py
+ re-run pour trancher, (b) soit déférer (fire/FWI = vitesse-dominé) et passer M_H'1c Perdigão IOP.

## M_H'1e DIRECTION verdict 2026-06-02 — bug fixé, ANN speed-only AMÉLIORE la direction

Job 22194874 (H100, walltime 39min). Exit_status=1 mais COSMÉTIQUE (le bloc de vérif PBS
cherche encore l'ancien `loso_summary.json`/colonnes obsolètes ; les vrais livrables
`loso_summary_dir.json` + `pairing_dir.parquet` sont écrits OK à 09:52). À nettoyer : adapter
le PY de vérif du PBS au nouveau schema dir.

SANITY OK : recompute parquet-only ALL val raw 41.2° / era5 32.1° (≈ recalcul Boss 43/34)
→ bug du +33° fantôme (2e obs-store join) bien corrigé.

**Direction val-43 (agrégé sur pairings, speed≥1, arbitre OBS)** :
| | corrigé | raw | era5 |
|---|---|---|---|
| MAE dir | **36.1°** | 41.2° | 32.1° |
| médiane | 22.6° | 26.0° | 18.7° |
| biais | +11.8° | +20.1° | +5.2° |

**L'ANN speed-only (loss SANS terme directionnel) AMÉLIORE la direction** : 36.1° vs raw 41.2°
(−5°), biais +20→+12°. Confirme la nuance user : en bougeant u,v pour la vitesse, la correction
redresse partiellement la déflexion relief-aware excessive du surrogate brut (PAS du bruit).
Ne rattrape pas tout à fait ERA5 (32°) mais l'écart est petit.

Par classe vent (val, MAE dir corr/raw/era5) :
- calme<1 : 90/88/85 (non informatif, exclu headline)
- faible 1-3 : 55/63/50
- moyen 3-7 : 29/34/25
- **fort>7 : 15.9/17.2/13.4** ← régime fire/éolien : direction corrigée excellente, ~égale ERA5

Distribution corrigée : 35% pairings <15°, 60% <30°. Train cohérent (33.3/38.4/29.2, gap
train↔val faible). `pairing_dir.parquet` persisté → toute ré-analyse dir future = CPU-only.

**VERDICT GLOBAL M_H'1a** : speed-only VALIDÉ. Vitesse −32.5% bat ERA5 ; direction non dégradée
(améliorée). Pas besoin d'ANN_direction pour fire/FWI (vent fort dir≈ERA5). ANN_direction =
optionnel futur pour track éolien si on veut fermer les ~4° résiduels en vent faible.
**Next : M_H'1c Perdigão IOP** (propagation spatiale 41 stations dans 6×6 km).

## M_H'1c Perdigão IOP verdict 2026-06-02 — propagation LISSE ✅, mais OOD au centre ⚠️

Job 22195330 (A100, exit 0, 5min, 328 pairings sous-éch. 8ts/station, override store
era5_europe_spring2017_v2.zarr Δt=6h). H100 22195325 tué (redondant). JJA cache intact 60971.

**PROPAGATION SPATIALE (test principal) = PARFAITE** :
- pct_smooth **100%** (0 pic), spike_ratio médian 1.01, p90 1.04
- delta(ring_k)/delta(centre) : ring1 1.01, ring2 0.99, ring3 0.94, ring5 0.90 → décroissance douce
- → la correction NE overfite PAS le pixel calibré ; elle s'applique de façon spatialement
  cohérente sur tout le voisinage. **Verrou propagation aux pixels non-station LEVÉ.**

**JUSTESSE AU CENTRE = DÉGRADE sur Perdigão (inverse du gain NOAA)** :
- MAE centre corr **2.27** vs raw **1.37** ; biais corr +1.92 vs raw +0.58 → sur-correction.
- Cause = DISTRIBUTION SHIFT, pas propagation : ANN entraîné sur JJA-2023 NOAA plaine FR/ES/PT
  (régime vent fort sous-estimé) appliqué à Perdigão mai-juin 2017, double crête microrelief,
  vent faible (raw déjà bon biais +0.58). L'ANN pousse trop haut un régime qui n'en a pas besoin.
- Les 2 résultats sont COHÉRENTS : propagation = propriété structurelle (lisse partout) ;
  justesse = dépend du régime d'entraînement. Perdigão IOP = OOD pour cet ANN JJA-plaine.

**Implications** :
- Architecture DEVINE-style VALIDÉE (propagation physique cohérente confirmée empiriquement).
- M_H'1a déployable POUR son domaine (été, plaine/colline EU, NOAA-like). PAS pour terrain
  alpin/microrelief hiver/printemps sans ré-entraînement incluant ces régimes.
- **Renforce le besoin de M_H'1b 4-saisons + features physiques** : un seul modèle JJA-plaine
  ne couvre pas Perdigão. M_H'1b (winter+MAM+JJA+SON + gradient T/RH) doit réduire ce shift.
- ⚠️ Pour le papier : NE PAS présenter Perdigão comme validation de justesse (c'est OOD) ;
  le présenter comme validation de PROPAGATION (100% lisse) + honnête caveat OOD régime.

**Décision next** : M_H'1b multi-season (le shift Perdigão justifie l'enrichissement saisonnier).

## M_H'1f soumis 2026-06-02 — features physiques stabilité (gate test JJA)

Job retrain 22197196.aqua (H100, 14h walltime, Q). Code GREEN, cache 60971 intact.

**12-dim topo_features** : 8 existants + [8] grad_T_850_surf_n, [9] grad_T_500_850_n,
[10] RH_surface_n, [11] q_surface_n. Dérivés AT READ TIME du grid.zarr (era5_3d/T centre
`[1,1,:]`, axe niveau = era5_pressure_levels hPa ; RH/q via Magnus-Tetens sur t2m,d2m).
Normalisés par constantes physiques PHYS_FEATURE_NORM ((-8,6),(-20,8),(60,25),(0.008,0.005)).
⚠️ grid.zarr n'a PAS de `sp` surface → q_surface utilise p_ref=1013.25 hPa documenté.
Rétrocompat : défaut topo_dim=8 + enable_phys_features=False → chemin M_H'1a byte-identique.
`enable_phys_features` câblé dans les 4 call sites ObsCenteredDataset.

⚠️ **PIÈGE re-éval Perdigão (à faire APRÈS retrain)** : éditer `eval_perdigao_M_H1c.yaml` →
`topo_dim:12` + `enable_phys_features:true` AVANT de lancer audit_devine_perdigao.py avec le
nouveau best.pt, sinon crash 8-dim features vs 12-dim ANN. Le chiffre clef = MAE centre Perdigão
vs 2.27 (baisse = hypothèse validée → GO plan conditionnel ci-dessous).

⚠️ **Piège PBS** : les trainings PBS ont un eval post-train caché (`eval_devine_style.py`) qui
construit AUSSI dataset+ANN → tout changement topo_dim doit l'inclure (sinon train GREEN puis
crash eval). Déjà géré pour M_H'1f.

## M_H'1f retrain TERMINÉ 2026-06-02 — NOAA préservé, Perdigão en cours

Job 22197196 (H100, exit 0, 4h54, 8 epochs). best.pt = ep5. Eval auto post-train sur val-43 :
- corr **mae 1.213** rmse 1.619 bias -0.124 | raw mae 1.812 → **Δmae -0.599** (vs M_H'1a -0.589)
- → les 4 features physiques (topo_dim 12) PRÉSERVENT et améliorent marginalement le NOAA
  (1.213 < M_H'1a 1.223). Convergence ep5 best, ep6-7 plateau. NOAA OK confirmé.

**Re-éval Perdigão M_H'1f VERDICT 2026-06-02** (job 22199665 H100 exit 0, 2min ; A100 22199633 tué) :

| | M_H'1a 8-dim | M_H'1f 12-dim phys | raw |
|---|---|---|---|
| MAE centre | 2.27 | **2.15** | 1.37 |
| bias centre | +1.92 | +1.71 | +0.58 |
| pct_smooth | 100% | 100% | — |

**Hypothèse features physiques = MARGINALEMENT validée, INSUFFISANTE** : features stabilité
(grad_T, RH, q) réduisent la sur-correction de seulement −5% (2.27→2.15), bias +1.92→+1.71.
La correction reste LARGEMENT pire que raw (1.37). Direction du bon sens mais effet trop faible.

**Conclusion = le plafond est l'ÉTAGE B (surrogate frozen)** : il n'a jamais vu de terrain
raide (Pop B exclue RANS), sous-estime structurellement les crêtes Perdigão. Donner 4 scalaires
de contexte à un MLP ~10k ne compense pas. La voie "descripteurs scalaires" PLAFONNE empiriquement.

**→ Next = STL-ENCODER ANN** (plan user) : donner le terrain complet 180×180 à l'ANN via encodeur
CNN pour qu'il apprenne une correction POSITION-dépendante compensant le biais de crête du
surrogate. M_H'1f prouve que c'est le bon levier (le scalaire ne suffit pas). Combiné aux obs
montagne (calibration crête) + 4 saisons. M_H'1b 4-saisons reste utile pour la couverture
régimes mais ne réglera PAS Perdigão seul — c'est le STL-encoder qui est le levier Perdigão.

NOAA M_H'1f préservé (val 1.213 ≤ 1.223). Cache 60971 intact.

## M_H'1g lancé 2026-06-03 — STL-encodeur (gate Perdigão) + ssrd/tcc en parallèle

M_H'1f a tranché : features SCALAIRES insuffisantes sur Perdigão (2.27→2.15). Décision user =
version complète STL-encodeur + radiation/nuages + régime. CAPE confirmé ABSENT de GEE
(CDS-only) → remplacé par ssrd+tcc (présents dans GEE ECMWF/ERA5_HOURLY) = proxies forçage
diurne. grad_T_500_850 (M_H'1f) sert déjà de proxy instabilité. 2 Departments PARALLÈLES :

**M_H'1g-stl** (levier principal) : encodeur CNN terrain 180×180 dans ANNCorrection.
- Archi : 4 conv stride-2 (4→16→32→64→64) GroupNorm+SELU → avgpool → Linear→latent 48d,
  concaténé [era5_flat 408, topo 12]. **92.6k params** (vs 26k legacy), bande anti-overfit OK.
  Flag `use_terrain_encoder` (défaut False → rétrocompat strict, vérifié sur vrai M_H'1f ckpt).
- Config devine_style_M_H1g.yaml (clone M_H1f + use_terrain_encoder:true, terrain_latent_dim:48,
  output surrogate_v2_devine_M_H1g_stl). Job retrain **22207705** (Q, H100, walltime 16h).
- Re-éval Perdigão = job séparé post-retrain via eval_perdigao_M_H1g.pbs (config eval clone la
  shape ANN du TRAIN, pas le défaut M_H1c topo_dim=8). Chiffre clef = MAE centre vs 2.27/1.37.
- Verdict attendu : baisse nette Perdigão = encodeur EST le levier → ajouter ssrd/tcc + obs
  montagne + 4 saisons ; effet faible = étage B surrogate = vrai mur → Phase I / surrogate v3.

**M_H'1g-ingest** (parallèle) : ssrd+tcc GEE → store annexe era5_radcloud_jja2023.zarr +
helper radcloud_at (topo_dim 12→14 au tour suivant). Script `ingest_era5_radcloud_gee.py` livré,
download LOCAL en cours (PID-based, PAS un job Aqua → hpc-watch ne le voit pas, poll manuel via
`ls data/raw/_cache_radcloud | wc -l` ; log /tmp/radcloud_full.log).
⚠️ RYTHME LENT : ~48 h-fichiers/30min → ~20-25h pour 2208 h (JJA). GEE point-par-heure inefficace.
Tourne en parallèle, ne bloque PAS le STL-encodeur (le vrai levier). Si encodeur répare Perdigão
seul → ssrd/tcc optionnel (pourrait être arrêté). Caveat à revoir : batcher le download GEE
(getRegion multi-temps) plutôt que point-par-heure si on relance.

NOAA M_H'1f = 1.213 (référence à ne pas dégrader). Cache 60971 intact.

## M_H'1g STL-encodeur VERDICT 2026-06-03 — N'A PAS battu 2.15, le mur EST l'étage B

Job retrain 22207705 (exit 0, 4h54, best ep6) + re-éval Perdigão 22209205 (exit 0). NOAA
préservé (corr mae **1.227**, bias -0.007, no overfit malgré encodeur CNN 92.6k params).

**Perdigão centre MAE — TROIS leviers ANN testés, tous plafonnent** :
| modèle | Perdigão centre MAE | bias |
|---|---|---|
| raw (cible) | 1.37 | +0.58 |
| M_H'1a (descripteurs 8) | 2.27 | +1.92 |
| M_H'1f (features physiques 12) | 2.15 | +1.71 |
| **M_H'1g (encodeur terrain 180×180)** | **2.32** | +1.95 |

L'encodeur CNN du relief complet N'AIDE PAS (2.32, marginalement pire que scalaire). Propagation
100% lisse partout. → **AUCUNE info donnée à l'ANN ne répare Perdigão.**

**CONCLUSION DÉFINITIVE = le mur est l'ÉTAGE B (surrogate v2 frozen)**, pas l'étage A (ANN).
Le surrogate sous-estime structurellement les crêtes car il n'a JAMAIS vu de terrain raide
(Pop B pentes >25° exclue training, RANS k-ε diverge). L'ANN ne peut pas faire produire au
surrogate un écoulement de crête qu'il ne sait pas calculer, quelle que soit l'entrée. Résultat
scientifique NET : voie "correction via ANN" éliminée empiriquement (3 leviers convergents).

**→ Pour réparer Perdigão (terrain alpin/microrelief), il FAUT un surrogate v3** entraîné sur
terrain raide : re-simuler ~50 cas Pop B avec solveur non-divergent (Zephyr.jl LBM GPU ou OF LES),
= Phase I conditionnelle du mandate. Hors cadre Phase H' (surrogate frozen).

**MAIS — M_H'1a reste pleinement valide pour SON domaine** : été plaine/colline EU (NOAA-like),
−32.5% MAE bat ERA5, direction améliorée, propagation lisse. Livrable paper/déploiement fire
weather plaine. Perdigão = caveat OOD honnête (terrain raide hors-scope v2). ssrd/tcc store prêt
(era5_radcloud_jja2023.zarr) mais inutilisé vu ce verdict.

## RE-SCOPE 2026-06-03 (user) — verdict M_H'1g PRÉMATURÉ, ouvrir M_H'1h ablation terrain raide

**Push-back user décisif** : M_H'1g a testé l'encodeur terrain ENTRAÎNÉ SUR NOAA JJA PLAINE.
L'encodeur a la capacité d'une correction position-dépendante, mais le backprop n'a jamais vu de
station de CRÊTE avec obs → aucun signal pour apprendre la compensation. Conclure « mur = étage B »
depuis M_H'1g = *capacité sans exemples*. Les 3 leviers du plan conditionnel sont indissociables et
seul M_H'1f (features phys) a été partiellement fait (JJA only). **La vraie expérience manquante** =
features physiques + **4 saisons** + **data DEVINE (obs montagne RAIDE)**, jamais combinés.

**Décisions user 2026-06-03** :
1. Source obs montagne = **ingérer SYNOP/AEMET alpin+pyrénéen filtré par PENTE** (pas juste altitude).
   Caveat : altitude ≠ pente. Les 68 stations >500m existantes sont SYNOP NOAA (vallée/plateau,
   ES meseta plate). Le surrogate échoue sur l'accélération de CRÊTE → besoin d'obs haute-pente.
   ⚠️ ingest_synop_meteofr (M_G2) + ingest_ogimet (M_G4) étaient RED 2026-05-21 → vérifier source vivante.
2. Archi = **ablation M_H'1f scalaire (topo_dim 12) VS M_H'1g encodeur** sur même dataset combiné.
   Tranche définitivement scalaire vs position-dépendant QUAND l'ANN a enfin des exemples de crête.

**Gate avant briefs ingestion/training** : investigation M_I0 (read-only) = source obs raide vivante
+ N réaliste stations nouvelles + filtrage par pente (DEM 186 tiles Copernicus + SRTM) + état données
4 saisons Aqua (re-inference 22066177/78/79 du 2026-05-27 complétée ? caches grid.zarr mam/son ?).

**Si M_H'1h (obs montagne + 4 saisons) répare Perdigão** → voie ANN RÉHABILITÉE, étage B PAS le mur.
**Si plafonne encore AVEC obs crête** → ALORS seulement le verdict « mur = surrogate frozen » tient,
et Phase I (surrogate v3 terrain raide) devient justifiée. Obs montagne = pré-requis pour conclure.

## M_I1 DONE GREEN 2026-06-03 — obs montagne raide ingérées (NCEI down → miroir AWS)

- **Blocage résolu** : NCEI `ncei.noaa.gov/pub/data` était DOWN (curl HTTP 000, internet OK ; 0 station
  alpine en cache). Bascule sur miroir AWS `noaa-global-hourly-pds` (user GO). Parser
  `parse_isd_global_hourly_csv` + `fetch_isd_global_hourly_year` + flag `--source aws` ajoutés
  (isd_parser.py / ingest_noaa_isd.py). Aussi cap wall-clock 30s/fichier (stream=True ne borne pas le hang).
- **205 stations alpines** (AT 146, IT 57, CH 2 — NOAA a peu de CH), 2022-2023, 17520h × 205, 99.3% vent
  valide. Store `data/raw/obs_unified_steep.zarr` = 362 prod + 205 = **567 stations** + champ `stations/slope`.
- **Pente (helper slope_at_dem validé)** : Alps médiane 6.4°, max **37.96°** ; **55 >15°**, 79 >10°, 112 >5°.
  Sanity Perdigão ridge **17.0°** / vallée 13.4° → les Alps raides COUVRENT le régime Perdigão (~17°).
  Le régime crête est ENFIN échantillonné (vs training JJA-plaine plat) = prémisse du re-scope satisfaite.
- ⚠️ slope = NaN pour les 362 prod (remplir sur Aqua depuis grid.zarr terrain en M_I2/M_I3). Store/CSV LOCAUX
  → à pousser sur Aqua. Code M_I1 PAS encore commité (filtre +CH/AT/IT, source AWS, slope helper, build).
- **Next = M_I2** : pairings 4 saisons des 205 nouvelles stations (inférence v2) + grid.zarr MAM/SON, sur Aqua.

## M_I2a smoke GREEN 2026-06-03 — surrogate SOUS-PRÉDIT la crête (signal confirmé) ; gap ERA5 grille Est

- Smoke job 22220282 exit 0 (CPU, GPU monopolisé par sweep 8h) : 10 pairings Alps, speed_pred ∈[0.82,1.76]
  physique, DEM+WC résolus à 3488 m (Cervin, lat 45.93/lon 7.70). Contrat v2 OK (terrain_ch=4, era5_dim=408,
  nz=24, Δt=0). **Surrogate sous-prédit le vent de crête : pred 1-1.8 vs obs 5 m/s** = EXACTEMENT le signal
  que la correction DEVINE doit corriger. Biais crête confirmé empiriquement sur vraie station raide (pas que Perdigão).
- Prérequis Aqua comblés Aqua-side (S3 anonyme, PAS de scp lourd) : DEM 38 tuiles + WC 9 tuiles. Poussé :
  `~/dsw/data/raw/obs_unified_alps.zarr` (205 stations slope-finie, 33 MB) + smoke store. PBS `infer_alps_v2{,_smoke}.pbs` prêts.
- ⚠️ **GAP BLOQUANT 4-saisons** : ERA5 hourly mam/jja/son 2023 = grille lon **-10→10E SEULEMENT** →
  **178/205 stations Alpes (Est : AT + IT-Est, lon>10E) HORS grille** pour 3 saisons. Seul `winter2223` (6h)
  couvre l'Est (→26.8E). Full 4-saisons des 205 ⇒ ré-ingérer mam/jja/son ERA5 grille étendue ~18E
  (`ingest_era5_europe_hourly.py`, CDS ≤7j/req ; ref M_H+ : ~2-3h wall/saison). Sinon mam/jja/son = 27 stations Ouest only.
- Plan prod (NON soumis) : 4 jobs ~7.3h total (winter 205 ~5.2h ; jja/mam/son 27→205 après ré-ingest).
  Throughput 3.93 pairings/s. Sortie `alps_<season>_v2.parquet` ~12 MB / ~103k rows.
- **DÉCISION user EN ATTENTE** : (A) ré-ingest ERA5 Est puis 4-saisons complet ; (B) winter-205 now +
  mam/jja/son Ouest-27 only ; (C) winter-205 now EN PARALLÈLE du ré-ingest ERA5, puis les 3 autres saisons.

## DEVINE 2024 obs reference + RE-SCOPE dry-Med (2026-06-04, user)

**DEVINE (Le Toumelin 2024 NPG 31:75) — recherche** : obs = 273 AWS ALPES (214 MeteoSwiss + 54 Météo-France
+ 5 GLACIOCLIM), PROPRIÉTAIRES (repo = code seul, pas d'obs réutilisables). Train ANN sur 218, test 55
held-out par échantillonnage stratifié 6 descripteurs topo (élévation, TPI500, **PENTE**, Laplacien, x/y).
Vent speed+dir 10m horaire. CNN = 7279 ARPS topo gaussiennes synthétiques 30m.
**CLEF** : testé hors Alpes sur 18 Corse + 21 Pyrénées → correction NN+DEVINE **DÉGRADE** (Table 4) ;
seul DEVINE-seul transfère. **Correction Alps-specific** = MÊME OOD que notre M_H'1a JJA-plaine→Perdigão.
→ VALIDE la stratégie multi-région (Alps humide + Med sèche). Notre pente = même levier que leur TPI/slope.

**User 2026-06-04 : élargir aux régimes SECS d'été (fire weather), pas que l'Autriche humide.** Inventaire
NOAA ISD montagne Med (lat34-46, elev>600m, actif post-2020) :
- **Espagne 58 >600m / 20 >1500m** (Sierra Nevada, Ibérique) — **DÉJÀ dans prod ES → déjà dans les 264k
  pairings 4-saisons !** Manque juste = calculer la PENTE des stations prod (NaN) pour flagger les raides.
- **Italie 35 >600m** : Alpes-nord ingérées (M_I1, bbox lat43-49) mais **Apennins sud (lat<43) PAS ingérés**.
- Turquie 54, Atlas Algérie/Maroc 44 (secs, max 2710m) — possibles mais lon→44 / lat→30 = grille ERA5 énorme.
- **Grèce ISD INUTILISABLE** : 53 stations, 0 >800m (airports). À oublier malgré la demande user.
→ Cœur fire dry-summer exploitable = Ibérie (déjà appariée) + Apennins IT-sud (nouveau) + S-France. ERA5
re-ingest mam/jja/son grille étendue. Espagne dry-summer ≈ GRATUIT (déjà dans 264k). Scope géo à confirmer user.

## Scope EU-Med fixé + Aqua DOWN 2026-06-04 — ERA5 re-ingest bloqué (PBS prêt)

- **User a fixé périmètre = cœur EU-Med** (Ibérie + Apennins + Alpes + S-France). Grille ERA5 lon −10→20, lat 35→49.
  Pas de Turquie/Atlas/Balkans (coût grille + couverture). Grèce inexploitable (ISD).
- **M_I2b ERA5 re-ingest** : PBS `configs/hpc/ingest_era5_multiseason_med.pbs` ÉCRIT (grille étendue, stores
  `_med`, ne clobber pas les originaux). MAIS **Aqua DOWN** : SSH "Connection refused" port 22 (Aqua était UP
  plus tôt aujourd'hui — M_I0/M_I2a/smoke 22220282 → panne/maintenance login node QUT transitoire ; internet OK).
  → **qsub le PBS med dès qu'Aqua revient.** 2e panne externe du jour (après NOAA NCEI).
- **BLOQUE par Aqua** : M_I2b (submit ERA5), M_I2d (pairings), M_I3 (training). Tout le compute Aqua.
- **Progrès Aqua-indépendant FAIT** : Apennins ingérés (77 stations IT-sud, NOAA AWS). DEM Apennins 59 tuiles.
  `build_obs_steep.py` GÉNÉRALISÉ (merge prod + liste de blocs raides). Store `obs_unified_steep.zarr` rebuild
  = **644 stations** (362 prod + 205 Alpes + 77 Apennins), slope des 282 new : 71 >15°, 101 >10°, max 38°
  (Apennins médiane 1.75° = bcp côtiers/plats, 16 >15° dry-Med). CSV 644 rows.
- **Pente stations prod FR/ES/PT (NaN)** : à faire SUR AQUA depuis grid.zarr (terrain déjà extrait), PAS de
  gros fetch DEM local. Flag les dry-Med Espagne raides déjà dans les 264k pairings.
- **Au retour d'Aqua, priorités** : (1) qsub `ingest_era5_multiseason_med.pbs` (ERA5 grille étendue, chemin
  critique CDS) ; (2) pente prod depuis grid.zarr ; (3) push obs_unified_steep.zarr (644) ; (4) pairings
  Alpes+Apennins ×4 saisons (M_I2d) ; (5) M_I3 training ablation. Monitor Aqua-recovery posé 2026-06-04.

## M_I2b ERA5 re-ingest SOUMIS 2026-06-04 — Aqua revenu (user "connected")

- Aqua de retour (aquarius02). PBS `ingest_era5_multiseason_med.pbs` poussé + soumis. **3 jobs chaînés afterok** :
  **22224136** mam2023 (Q) → **22224138** jja2023 (H) → **22224139** son2023 (H). Grille `35,-10,49,20`
  (lon −10→20, lat 35→49), sorties `era5_europe_hourly_{mam,jja,son}2023_med.zarr` (n'écrasent PAS les
  originaux). `--no-z --max-days-per-req 7`, walltime 10h/job, ~10-12h total. CDS creds OK.
- Watcher qstat sur 22224139 (dernier de la chaîne) armé → me réveille à la fin.
- **Au retour de la chaîne** : vérifier les 3 stores `_med` écrits, PUIS M_I2d pairings Alpes+Apennins (282
  new) ×{mam,jja,son} sur les stores `_med` + winter2223 (déjà OK). Winter pairings peuvent tourner AVANT
  (store winter prêt). Aussi : push `obs_unified_steep.zarr` (644) sur Aqua + pente prod via grid.zarr.

## M_I2b ERA5 re-ingest = GREEN 2026-06-04 — 3 stores _med écrits, grille étendue confirmée

Chaîne 22224136/38/39 exit 0, ~1h45/saison (~3.5h total, vs 10h walltime). Stores
`era5_europe_hourly_{mam,jja,son}2023_med.zarr` 1.6 Go chacun. Vérif jja_med : lon −10→20, lat 35→49,
2208 ts, pressure{q,t,u,v}+surface{u10,v10,t2m,d2m}. ⚠️ LEÇON : ne PAS conclure "échec rapide" sur un job
sorti vite de la queue — lire Exit_status (qstat -x) + vérifier outputs. Le watcher a vécu 3.5h, pas qq min.
Donnée ERA5 4-saisons EU-Med PRÊTE. → M_I2d lançable (pairings 282 new Alps+Apennins, mam/jja/son sur _med, winter sur winter2223).

## M_I2d pairings SOUMIS 2026-06-04 — 4 jobs parallèles, smoke Apennins GREEN

- 282-new obs store `data/raw/obs_unified_steep_new.zarr` (82 MB, AT146+IT134+CH2) poussé Aqua. DEM/WC
  Apennins fetchés Aqua (58 DEM + 9 WC, S3, dirs `srtm_tiles/` + `worldcover_esa/`). Smoke 22224844 exit 0 :
  Apennins sud (41.7,15.95) DEM résolu (terrain 75→876 m), 12/12 pairings, surrogate UNDER-prédit crête
  (pred 0.67-1.02 vs obs 1.55-3.10) — signal cohérent M_I2a.
- **4 jobs PROD (Q, parallèle, `infer_steep_v2.pbs`)** : jja **22225925**, mam **22225926**, son **22225927**,
  winter **22225928**. ERA5 routing : `_med` (mam/jja/son), `winter2223` (DJF). Sorties
  `~/dsw/data/inference/steep_{season}_v2.parquet`, ~50-60k rows/saison. Walltime ≤12h, qq h attendu.
- Watcher armé sur les 4. **Au retour** : merger steep_*_v2.parquet (~200k) + 264k prod → dataset M_I3.
  Reste : pente prod FR/ES/PT (grid.zarr) pour flagger Espagne dry-Med dans le merge.
- ⚠️ DEM prod Aqua = `~/dsw/data/raw/srtm_tiles/`, WC = `worldcover_esa/` (PAS copernicus_dsm_*).

## Steep pairings GREEN 2026-06-04/05 — raw surrogate NO-GO confirmé À L'ÉCHELLE (le gap à fermer)

4 jobs 22225925/26/27/28 exit 0. `steep_{season}_v2.parquet` = **344 615 pairings** (jja 68362/274stn,
mam 74290/269, son 72634/272, winter 129329/276). Cols : station_id,timestamp,lat,lon,elev,u/v/speed_obs,
u/v/w_pred,speed_pred,*_era5_baseline,era5_time_delta_minutes.
- **surrogate brut vs obs : MAE 1.982 bias −1.447** ; ERA5 vs obs MAE 1.777 bias −0.970 → surrogate brut
  PIRE qu'ERA5 sur terrain raide (attendu).
- **vent fort >6 m/s (n=67730) : surrogate MAE 4.46 bias −4.36** → sous-prédit ≈⅔ les crêtes. Même
  compression que Perdigão, à l'échelle sur 282 stations EU-Med raides. = LE gap que la correction doit fermer.
- **M_I3 = train ablation AVEC ces exemples de crête** (vs M_H'1a plaine→dégradait Perdigão). Watcher leçon :
  fausse-sortie sur blip réseau local "Can't assign requested address" (PAS Aqua, PAS port exhaustion) — re-tester.
- M_I3 prep lancé : besoin = grid.zarr cache training pour steep×4saisons (vérifier si existe/à matérialiser),
  merge steep 344k + prod 264k ≈ 609k, split watertight (hold-out steep + Perdigão), ablation scalaire vs encodeur.

## M_I3a prep GREEN 2026-06-05 — cache à construire (1.45TB/~4h), smoke ENCODEUR prometteur

- **Cache gate** : M_I2d inference rmtree ses grids (`infer_at_stations.py:525`, pas `--keep-grids`) → AUCUN
  cache steep réutilisable. Matérialiser 604,869 grid.zarr ≈ **1.45 TB** (~2.4MB/grid, scratch 272TB OK), **~3-4h**
  array 4 nœuds, PAR SAISON (store correct, `*_med` lon→20E + winter Δt6h) puis `overwrite_cache=false` (cache-hit).
  Convention `{station_id}_{ts_tag}/grid.zarr` identique materialise↔loader → consommé verbatim.
- **Combined** `combined_steep_plain_v2.parquet` = 604,869 pairings / 529 stations (279 steep + 250 plain, overlap 0).
  Split watertight seed=42 : train **423 stn/489,376** ; val **106/115,493** dont **54 stations steep held-out/64,262**.
  Perdigão = éval spatiale séparée (jamais dans le set).
- **Smoke GREEN 2 arms** (job 22229533, 6 stn JJA, 2 ep, exit 0) : scalar val_mae 1.683→1.608 (Δ−0.162) ;
  **encoder 1.623→1.517 (Δ−0.253)** → l'ENCODEUR fait MIEUX avec exemples de crête (le levier M_H'1g qui
  plafonnait en plaine paye dès qu'il a du raide). RAW steep 1.770 bias −0.866. Aucun crash, surrogate frozen 29M OK.
- **Plan prod (À CONFIRMER user)** : cache (~4h, 4 nœuds) PUIS 2 trainings // scalar (H100 ~24h) + encoder (H100 ~30h).
  ⚠️ 8 ep × 489k peut dépasser walltime → ~6 ep (M_H'1a convergeait ep5), best.pt/epoch = reprise OK.
- Fichiers : `configs/training/devine_style_M_I3_{scalar,encoder}{,_smoke}.yaml`, `configs/hpc/devine_style_M_I3_{smoke,materialise_cache,scalar,encoder}.pbs`, `services/module2b-surrogate/{build_combined_steep_plain_parquet,materialise_combined_cache}.py`.

## M_I3 PRODUCTION lancé 2026-06-06 (user GO : cache + 2 bras, 6 ep)

- epochs 8→**6** dans les 2 configs Aqua (M_H'1a convergeait ep5). **Cache array soumis : 22231418[]** (Q,
  -J 0-3, 16cpus, 12h walltime, ~4h attendu, 1.45TB).
- ⚠️ PBS Aqua REJETTE `depend=afterokarray:...[]` ("illegal -W value") → pas de chaînage auto. **Méthode :
  watcher robuste sur 22231418 (tolère rc=255 ssh-blip, exit seulement si rc=1 = ssh-ok+job-absent) → au retour
  je soumets `devine_style_M_I3_scalar.pbs` + `_encoder.pbs` (H100, walltime 24h/30h, best.pt/epoch).**
- LEÇON watcher : la v1 (`! ssh ...`) sortait faux sur blip réseau ; la v2 teste rc==1 explicitement.
- VERDICT attendu (au retour des 2 trainings) : val_mae held-out 54 stations steep + Perdigão centre vs raw
  (Perdigão raw 1.37 ; M_H'1a/f/g plafonnaient 2.27/2.15/2.32). Smoke encoder déjà Δ−0.25 prometteur.

## M_I3 cache GREEN + 2 trainings soumis 2026-06-06

- Cache array 22231418[] : **4/4 tâches Exit 0**, ~4h runtime (watcher a vécu 3.5-4h). Cache 605k grid.zarr
  construit (smoke avait validé la consommation cache-hit). find count en bg (bmpbh8cca, confirmation à venir).
- **2 trainings soumis (gpu_batch, Q)** : scalaire **22231497** (96GB, 24h), encodeur **22231498** (96GB, 30h).
  6 epochs, best.pt/epoch (reprise OK). Cache lu dès epoch 1 → cache vide = crash rapide (pas 20h gâchées).
- Watcher robuste sur les 2 (grep 2223149, rc==1, cap 50h) → me réveille quand LES DEUX finissent.
- **AU RETOUR = VERDICT** : (1) qstat -x Exit_status des 2 ; (2) val_mae held-out 54 stations steep + Δ vs raw
  steep 1.770 ; (3) re-éval Perdigão centre (audit_devine_perdigao) avec les 2 best.pt vs raw 1.37 / M_H'1a-f-g.
  Si Perdigão baisse nettement sous 2.15 → voie ANN réhabilitée par exemples de crête. Sinon → mur = surrogate.

## M_I3 trainings TERMINÉS 2026-06-08 — held-out −22% GREEN, encodeur gagne (verdict Perdigão en cours)

- Les 2 jobs tués au walltime (Exit −29 : scalaire 24h, encodeur 30h) car **~5.5h/epoch** (489k pairings,
  forward surrogate par pairing = bien plus lourd qu'estimé). best.pt/epoch a sauvé le meilleur (ep2 les deux).
- **Held-out val (106 stn dont 54 steep)** : RAW mae 1.909 bias −1.391.
  - scalaire (M_H'1f) best ep2 : **1.518 (Δ−0.391, −20.5%)** bias +0.157
  - encodeur (M_H'1g) best ep2 : **1.482 (Δ−0.427, −22.4%)** bias +0.109 ← **GAGNE**
  - Convergence ep2 puis léger overfit (ep3-4 val remonte) → 6 ep inutiles, kill sans coût.
- **L'encodeur (qui plafonnait M_H'1g en plaine) BAT le scalaire avec exemples de crête** = hypothèse re-scope
  validée à l'échelle. Biais −1.39→+0.1 = quasi corrigé.
- ⚠️ best.pt : scalaire 108KB (ANN 26k), encodeur 379KB (CNN 92.6k). Logs `data/models/surrogate_v2_devine_M_I3_*/train.log`.
- **VERDICT DÉCISIF EN COURS** : steep-54 seul + Perdigão centre (vs raw 1.37 ; M_H'1a/f/g 2.27/2.15/2.32).
  ⚠️ config eval Perdigão DOIT matcher l'archi (scalaire topo_dim12+phys ; encodeur use_terrain_encoder) sinon crash shape.

## M_I4 VERDICT 2026-06-08 — correction GÉNÉRALISE au steep (−22%), Perdigão = outlier sur-corrigé

Job 22258178 exit 0, 3min27, les 2 bras (eval_perdigao_M_I4.pbs).
| | held-out val (54 steep + 52 plain) | Perdigão centre | bias Perdigão |
|---|---|---|---|
| raw | 1.909 (bias −1.39) | **1.37** | +0.58 |
| M_I3 scalaire | 1.518 (−20.5%, bias +0.16) | 2.324 | +2.11 |
| M_I3 encodeur | **1.482 (−22.4%, bias +0.11)** | **2.193** | +1.92 |
(rappel plaine : M_H'1a/f/g Perdigão 2.27/2.15/2.32)

- **SUCCÈS cas steep général** : la correction entraînée AVEC exemples de crête généralise au terrain raide
  held-out (−22%, biais −1.4→+0.1). **Encodeur > scalaire partout** (held-out ET Perdigão) = levier confirmé.
  Au-delà de DEVINE (qui dégradait hors-Alpes). **Voie ANN réhabilitée pour le steep général.**
- **Perdigão reste sur-corrigé** (2.19-2.32 > raw 1.37). CAUSE = **raw Perdigão DÉJÀ bon** (bias +0.58, le
  surrogate NE sous-prédit PAS ce centre) → la correction (calibrée sur le biais général −1.4 m/s) SUR-corrige
  un centre qui n'en avait pas besoin (+0.58→+1.9). **Perdigão = OUTLIER microrelief où raw est déjà juste,
  PAS preuve que le surrogate est un mur.** Propagation 100% lisse les 2.
- **REVISION verdict antérieur** : « mur = surrogate frozen » était TROP CATÉGORIQUE. La correction AMÉLIORE
  le steep en général ; Perdigão est un cas spécial de sur-correction (raw déjà bon). Phase I (surrogate v3)
  MOINS justifiée qu'on pensait — le surrogate fait correctement le steep général, l'ANN le corrige.
- ⚠️ steep-54 ISOLÉ pas encore calculé (−22% = agrégé 54steep+52plain) → à faire pour solidifier le claim steep.
- Livrables : `data/validation/phase_H_prime_perdigao_M_I3_{scalar,encoder}/perdigao_summary.json`.
  best.pt encodeur = modèle retenu (1.482 held-out, biais quasi nul). Branche `feat/devine-style-correction`.

## M_I4 PERDIGÃO RE-CADRÉ 2026-06-09 (skepticism user JUSTE) — sur-correction = artefact VENT FAIBLE

User : « Perdigão a bcp de points/hauteurs, on ne peut pas statuer sur un seul ». Vérifié : obs = 48 mâts × **14
hauteurs** mais SEUL 10m bien échantillonné (90% valid ; autres 2-64%), et **IOP = VENT FAIBLE** (10m mean 1.82,
pairings obs mean 1.64). Éval `audit_devine_perdigao.py` = like-for-like CORRECT (surrogate extrait au niveau AGL
matchant l'obs, ligne 166 ; hauteur 10m argmin). Sous-éch. étalé sur l'IOP. Pas de bug — mais slice non-représentatif.
**Stratification par vent (encodeur, 376 pairings)** :
| obs vent | n | MAE corr | MAE raw | bias corr | bias raw |
|---|---|---|---|---|---|
| <1 | 174 (46%) | 2.65 | 1.53 | +2.65 | +1.48 |
| 1-2 | 79 | 2.25 | 1.12 | +2.25 | +0.75 |
| 2-3 | 58 | 1.64 | 0.95 | +1.61 | +0.02 |
| 3-5 | 52 | **1.24** | 1.33 | +0.25 | −1.24 |
| >5 | 13 | **1.95** | 2.81 | −1.71 | −2.81 |
- **84% des pairings Perdigão <3 m/s** = calme où raw DÉJÀ bon (MAE ~1, biais ~0) → la correction SUR-AJOUTE
  (biais +1.6 à +2.65) → c'est ça le "2.19" agrégé.
- **≥3 m/s (régime sous-prédit), la correction AMÉLIORE Perdigão** : 3-5 corr 1.24<raw 1.33 (biais −1.24→+0.25) ;
  >5 corr 1.95<raw 2.81 (biais −2.81→−1.71). **La correction marche AUSSI à Perdigão, là où elle doit.**
- **VERDICT RE-CADRÉ** : "sur-correction Perdigão" = artefact VENT FAIBLE, PAS échec terrain raide. Le plafond
  M_H'1a/f/g 2.27/2.15/2.32 = MÊME artefact (IOP dominé par le calme, 1 hauteur). Phase I NON justifiée.
- **RÉSIDU RÉEL = sur-correction en VENT FAIBLE** (ajoute du vent quand raw déjà bon). Fixable : loss τ-asymétrie
  pousse à sur-corriger le calme ; gating régime / plus d'exemples calmes. Cohérent vieille note "calme sur-corrige".
- CSV par-pairing : `data/validation/phase_H_prime_perdigao_M_I3_{enc,sca}/perdigao_propagation.csv` (cols speed_obs,
  speed_corr_centre, speed_raw_centre). À solidifier : stratifier le steep-54 held-out par vent aussi.

## M_I5 régime-aware lancé 2026-06-09 — smoke confirme (régime > symétrique), re-train soumis

- Fix sur-correction calme : `loss_mode=regime` (`devine_speed_loss_regime`, train_v2_devine_style.py) =
  pénalité over-pred ×2 en CALME (obs<3 m/s, gate sigmoid calm_width 1.5), garde τ-asym 0.6/0.4 en vent fort.
- **Smoke 22283593 exit 0, 2 variantes (1 ep, subset)** :
  | variante | bias LOW<3 | mae HIGH | bias HIGH |
  | RAW | +0.16 | 2.37 | −2.07 |
  | **régime** | **+1.10** | 1.84 | −0.91 |
  | symétrique τ0.5 | +1.53 | 1.80 | −0.44 |
  → **régime GAGNE** (réduit le + la sur-correction calme en gardant le gain vent-fort). Retenu prod.
- **Re-train régime SOUMIS : 22286112** (encodeur H100 36h, 6 ep, sortie `surrogate_v2_devine_M_I5_encoder`).
- M_I5a solidify eval 22283594 TUÉ walltime (1h trop court : load 64k grid.zarr lent) → **re-soumis 22286113 (4h)**.
- Watcher sur l'éval steep-54 d'abord, puis sur le re-train (~33h). **AU RETOUR** : (1) steep-54 stratifié confirme
  régime ; (2) re-train régime → re-éval Perdigão stratifiée + held-out : le LOW-bias doit chuter, le steep gain rester.

## M_I5a steep-54 stratifié = CONFIRME (2026-06-10) — régime hors Perdigão validé

Job 22286113 exit 0 (4h OK). M_I3 encodeur sur 54 stations crête held-out :
- OVERALL raw MAE 1.947 bias −1.44 → **corr 1.602 (−18%) bias +0.10**.
- <1 m/s (n1089) : raw 0.79 → corr 1.758 (sur-corrigé, raw déjà bon) ; bias +0.65→+1.76.
- **>6 m/s (n10848) : raw MAE 4.73 → corr 3.02 (−36%) ; bias −4.67 → −2.73** = gain ÉNORME en vent fort (régime fire).
→ **histoire régime CONFIRMÉE hors Perdigão** : aide massif vent fort (le régime critique), sur-correction calme.
- **Re-train régime 22286112 (R, 8h/36h) ep0** : val 1.498 (Δ−0.411), bias +0.07. LOW(<3,n45584) bias +1.11 raw
  −0.05 mae 1.283 ; HIGH(n64101) bias −0.672 raw −2.344 mae 1.651<2.640. → calme moins sur-corrigé, vent-fort gardé.
  Watcher sur 22286112 (~33h). AU RETOUR : re-éval Perdigão+held-out stratifiés du modèle régime vs M_I3.

## M_I5 régime re-train DONE 2026-06-11 — val 1.464 (≤M_I3 1.482), calme moins sur-corrigé

Job 22286112 exit 0, 34h, **6 ep complets**. best **ep2 val_mae 1.464** bias +0.07. Trajectoire LOW(<3)/HIGH
bias (full val) : ep2 LOW +0.944 mae 1.174 (raw −0.05/0.88) | HIGH −0.766 mae 1.670 (raw −2.344/2.640).
ep4 LOW +0.896. → **LOW over-correction RÉDUITE** (vs original asym) **en gardant le gain HIGH**. Globalement
≥ M_I3 (1.464 ≤ 1.482). best.pt = `data/models/surrogate_v2_devine_M_I5_encoder/best.pt` (ep2, encodeur arch).
→ Verdict eval régime (Perdigão stratifié + steep-54 stratifié) vs M_I3 = EN COURS. Compare au M_I3 :
Perdigão overall 2.19 (<1 bias +1.76, ≥3 aide) ; steep-54 overall 1.602 (<1 +1.76, >6 corr bias −2.73 mae 3.02).

## ⏸️ REPRISE NOUVELLE SESSION 2026-06-11 — évals régime EN FILE Aqua

Verdict eval régime soumis, **EN FILE A100** (gros array `22306411[*]` occupe les GPU) :
- **22306889** = Perdigão stratifié (régime) → `data/validation/phase_H_prime_perdigao_M_I5_encoder/perdigao_summary.json`
- **22306888** = steep-54 stratifié (régime) → `data/validation/phase_H_prime_M_I5c_steep54*/` (.log)
**À LA REPRISE** : `ssh maitreje@aqua 'qstat -x -f 22306889 22306888 | grep Exit_status'`. Si done+exit0 → lire
les 2 sorties, stratifier par vent, comparer **RÉGIME vs M_I3** (M_I3 ref : Perdigão overall 2.19 / <1 bias +1.76 /
≥3 aide ; steep-54 overall 1.602 / <1 bias +1.76 / >6 corr bias −2.73 mae 3.02 / overall −18%). Si encore Q → re-attendre
(re-`qsub` si tué walltime ; steep-54 = 4h walltime, Perdigão 1h). **Attendu** : régime baisse le biais calme (<3) en
gardant le gain >6. best.pt régime = `data/models/surrogate_v2_devine_M_I5_encoder/best.pt` (ep2, val 1.464 ≤ M_I3 1.482).
⚠️ Watchers de cette session NON persistants → re-checker manuellement. Reste après verdict : figer le modèle retenu
(régime si confirmé, sinon M_I3 encodeur), commit M_I (loss régime + scripts), écrire le résultat. Phase I non justifiée.

## PLAN CONDITIONNEL post-M_H'1f (CADUC — voir verdict M_H'1g ci-dessus : encodeur n'a pas marché)

Si M_H'1f (features physiques scalaires, test JJA) réduit la sur-correction Perdigão → la
version finale de l'ANN combine TROIS leviers, tous DANS le cadre Phase H' (surrogate reste
FROZEN — PAS de surrogate v3) :

1. **STL complet en entrée de l'ANN** (idée user, clarifiée) : passer le champ terrain complet
   180×180 (déjà dans grid.zarr `terrain_2d 4×180×180`, ZÉRO nouveau data) à l'ANN via un
   ENCODEUR CNN, au lieu des 8-12 descripteurs scalaires (mean_topo/std_topo/z0_eff).
   - **Pourquoi puissant** : via le backprop end-to-end à travers le surrogate frozen, l'ANN
     peut apprendre à COMPENSER le biais structurel du surrogate sur terrain raide (Pop B
     exclue → surrogate sous-estime les crêtes). Il pré-amplifie l'ERA5 là où le relief est
     raide. = réparer l'étage B (limite surrogate) VIA l'étage A (ANN), sans re-train surrogate.
   - Descripteurs scalaires mean/std deviennent redondants (l'encodeur les extrait) ; features
     physiques stabilité (gradient T, RH, q) RESTENT (pas dans le terrain).
2. **Sources DEVINE = obs montagne** (réseau type Alpes/Pyrénées) : nécessaires pour CALIBRER
   la compensation crête — sans exemples de terrain raide, l'ANN ne peut pas apprendre à
   compenser. = SEUL nouveau data du plan.
3. **4 saisons** (winter+MAM+JJA+SON) : couverture régimes.

**Les 3 sont indissociables** : STL sans obs montagne = capacité sans exemples ; obs montagne
sans STL = exemples sans perception fine. 

**Caveats** :
- L'ANN n'est plus "small" : encodeur CNN 180×180 → ~100k-1M params (vs ~10k MLP). S'éloigne du
  DEVINE tiny front-end → RISQUE OVERFITTING (mêmes données, +capacité). Surveiller gap train↔val.
- Reste FROZEN-compatible : on ne touche pas au surrogate. Architecture ANN seule change.

**Gate** : ne rien lancer de ce plan tant que M_H'1f n'a pas montré que les features physiques
aident sur Perdigão. Si M_H'1f NO-GO (features ne suffisent pas) → re-discuter : soit aller
direct au STL-encodeur (plus de capacité), soit accepter que l'étage B (surrogate terrain-raide)
soit le vrai plafond → Phase I / surrogate v3 (Zephyr.jl / LES).

## M_H'1a JJA TERMINÉ GREEN 2026-06-01 — Phase H' DEVINE-style validée à production-scale

Job 22135584.aqua, walltime 11h15, exit 0, 8 epochs complets sur 12061 train + 3015 val pairings JJA 2023 (~214 stations FR/ES/PT held-out random val 20%).

**Verdict empirique massif** :

| Métrique | Raw surrogate v2 | M_H'1a corrigé | Gain |
|---|---|---|---|
| **val_mae** | 1.812 m/s | **1.222 m/s** | **−32.5%** |
| **val_bias** | −1.453 m/s | **+0.008 m/s** | bias quasi-nul |
| **best epoch** | — | 5 | converge fast |

Config V2 winner (arbitrage Dept A vs Dept B) : lr=5e-4, grad_clip=1.0, τ=0.6/0.4 DEVINE default, batch_size=4, 8 epochs. Stable, monotone, pas d'overshoot epoch 1.

**Pourquoi JJA gain 3× smoke v3 winter** : surrogate v2 raw a été entraîné sur cas CFD k-ε neutre. En été = vent thermique + canicule = régime physique le plus loin de l'entraînement. Bias raw -1.45 m/s en JJA vs -0.80 en winter = plus de marge à corriger. DEVINE-style comble.

**Implications stratégiques** :
- **Paper NatComms fire weather** : preuve concrète -32% MAE summer wind sur 214 stations EU held-out, bias=0 → exploitable FWI direct
- **Startup downscaling** : surrogate v2 raw était NO-GO (REGRESS -8% vs ERA5). Surrogate + correction DEVINE-style bat ERA5 raw clairement

**Caveats restants** :
- Validation Perdigão IOP (propagation spatiale 41 stations dans 6×6 km) PAS encore faite
- **CORRECTION 2026-06-01** : le split val EST watertight station-disjoint (`watertight_station_split` dans dataset_v2_obs_centered.py:385, appelé inconditionnellement train_v2_devine_style.py:285). val = ~44 stations FR/ES/PT JAMAIS vues à l'entraînement (20% des stations, perdigao exclu). Donc **val_mae 1.222 EST déjà le chiffre LOSO honnête**. L'ancienne note "random 20% pairings" était FAUSSE. La mission M_H'1d "re-train honnête" était un no-op → Department BLOCKED à raison (économise 9h H100).
- 3 saisons restantes (winter/MAM/SON) à faire (loader actuellement mono-era5_store, refactor nécessaire)
- 21 stations Perdigão exclues à raison du flag mandate §5, mais inclues comme test propagation explicite séparée

**Plan suivant** :
1. Validation Perdigão IOP sur best.pt M_H'1a → vérifier que la correction généralise aux pixels non-station
2. LOSO honest spatial sur M_H'1a (re-split 80/20 stations différentes, eval)
3. M_H'1b multi-season (refactor loader + train sur 264k pairings 4 saisons)
4. Si Perdigão GREEN + LOSO honest GREEN → M_H'3 audit final + GO déploiement

## M_H'0 smoke v3 GREEN 2026-05-29 — pipeline DEVINE-style valid empiriquement

Smoke v3 (200 stations × winter2223 × 25k train + 6k val × WC tiles ESA enabled × 15 epochs target, killed walltime à 6 epochs) :

| Epoch | train_mae | val_mae | val_mae_raw | Δmae |
|---|---|---|---|---|
| 0 | 1.498 | 1.405 | 1.602 | **-0.196** |
| 2 | 1.421 | 1.407 | 1.602 | -0.195 |
| 5 | 1.399 | 1.417 | 1.602 | -0.185 |

**Verdict net** (vs smoke v1 ambigu) :
- train + val descendent ensemble (pas d'overfitting)
- **Δmae STABLE -0.18/-0.20 sur 6 epochs** (vs v1 oscillait -0.16/+0.02)
- 0 WC tile missing warnings (vs v1 = 2354)
- Convergence rapide : epoch 0-2 trouve l'amélioration, epochs 2-5 la stabilisent

**Caveat structural** : val_bias passe -0.796 (raw) → +0.5 (corrigé) = **over-correction** par τ asymétrique 0.6/0.4. Le modèle saute le bon point. Fix : tuner τ à 0.55/0.45 ou ajouter loss term `λ × |bias|`.

**Compute estimate pour M_H'1 full** :
- Smoke v3 : 25k train pairings × 33 min/epoch = ~80 ms/pairing
- Full 264k × 30 epochs × batch_size=4 = ~88h ❌ trop long
- Avec batch_size=16 + bf16 + epochs=7 (convergence rapide observée) = ~10h ✅
- Ou batch_size=8 + epochs=10 = ~30h

**Plan M_H'1 full** :
1. Tuner τ à 0.55/0.45 sur smoke court (~2h pour valider bias closer à 0)
2. Full training sur 264k pairings, batch_size=16 (si A100 80GB) ou 8 (si 40GB), bf16, 7-10 epochs
3. Validation : LOSO spatial split + Perdigão IOP 41 stations propagation test

## Phase H' pivot DEVINE-style validé 2026-05-27 — abandon Strategy A E.2

**Décision Boss + user 2026-05-27** : après verdict M_H1 YELLOW (toggle_delta plat 3.4e-6 à epoch 29, canal OBS jamais activé) et lecture du paper DEVINE 2024 (Le Toumelin et al. NPG), **abandon de l'option E.2** (canal OBS in-model) et pivot vers **option B/D revisited inspirée DEVINE** :

```
ERA5_input + topo_features → ANN(MLP small) → ERA5_corrigé → surrogate v2 FROZEN → champ pred
                                                                                       ↓
                                                                         loss au pixel central station seul
                                                                         (τ asymétrique 0.6/0.4)
```

**Architecture DEVINE-style** :
1. Pas de canal OBS in-model. OBS = target seul.
2. ANN simple (MLP 2-3 layers, ~quelques milliers de params) qui corrige l'**input ERA5** (vector 408)
3. Skip connection résiduelle (ANN apprend Δ vs absolute)
4. Surrogate v2 frozen consume corrected ERA5, produit champ pred
5. Patch centré sur station (pixel central 90,90 = station), pas random
6. Loss au pixel central avec τ asymétrique (penalize underestimation comme DEVINE : surrogate v2 slope 0.59 → underestime vent fort)
7. Backprop end-to-end à travers surrogate v2 frozen — l'ANN voit comment le surrogate transforme et apprend à anticiper la compression

**Pourquoi ça marchera (vs E.2 qui a échoué)** :
- DEVINE 2024 a fait exactement ça avec succès : MAE 1.42 m/s, -7.8% vs DEVINE seul, -15.7% vs raw AROME
- Pas de problème d'init faible (skip connection résiduelle absorbe l'init)
- Pas de problème de pixel random (station toujours au centre du patch)
- Pas de signal dilué (loss concentrée sur 1 pixel)
- Pas de loss qui n'incite pas à utiliser OBS (loss EST l'OBS)
- DEVINE est exactement notre setup : CNN surrogate downscaler frozen + small ANN correction in front

**Verrou résiduel** : propagation au reste du domaine présumée par architecture (surrogate v2 frozen propage cohérence physique). **Validation explicite** via **Perdigão IOP 41 stations dans 6×6 km** = golden test set propagation spatiale (seul dataset qui le permet).

**Plan secours si Phase H' plafonne** : recasting vers une architecture pixel-wise complète, ou retour aux approches statistiques (XGBoost stratifié). Mais DEVINE est notre meilleur pari empirique + théorique.

## M_H1 preflight GREEN + full submitted 2026-05-27 — training E.1 en cours

**Preflight 22027837** : 1 epoch sur 32 train + 16 val cases, 18.6s wall sur H100. Exit 0.
- val_mse = **0.097** (mieux que baseline 0.121, le surrogate s'améliore en partant des nouveaux poids OBS init petits + 1 epoch)
- toggle_delta = 2.4e-7 (plat, attendu à 1 epoch)
- toggle_mean_abs_pred_diff = 2.4e-5

**M_H1 full PBS 22029594.aqua RUNNING** depuis 09:29 le 2026-05-27.
- 30 epochs sur 6567 train + 1300 val cases full
- Monitor toggle test à epoch 5/10/15/20/25/30
- Auto-warning si Δ < 1e-4 à epoch 10 (escalate Boss avant gaspiller 6h+)
- ETA 6-12h

## M_H+ Axe 2 ERA5 chained jobs GREEN 2026-05-27 confirmé

Tous les 3 jobs ERA5 ont en réalité **réussi** (le "Time Use 24-46s" dans qstat -x = CPU time, PAS wall time). Wall réel :
- 21982386 (MAM 2023) : 1h51, DONE
- 21982395 (JJA 2023) : 3h11, DONE
- 21982396 (SON 2023) : 2h52, DONE

Stores : `/home/maitreje/dsw/data/raw/era5_europe_hourly_{mam,jja,son}2023.zarr/` 1.2 GB chacun, {coords, pressure, surface{u10,v10,t2m,d2m}} tous présents.

**M_H+ entièrement GREEN** (axe 1 DEM, axe 2 ERA5, axe 3 multiproc).

**Prochaine étape** : après M_H1 full GREEN convergé, lancer re-inference v2 production sur 4 saisons × 362 stations via `qsub -v SEASON=<...> configs/hpc/infer_at_stations_v2.pbs`. Cible parquet étendu ≥200k pairings.

How to apply : leçon "qstat -x Time Use = CPU time only, pas wall" → toujours lire wall séparément via PBS o file ou `qstat -f`. Le diagnostic prématuré sur 24-46s a failli mal interpréter un succès.

## M_H0_smoke YELLOW 2026-05-26 — pipeline OK, canal OBS pas encore activé (smoke trop court)

PBS 21980675.aqua terminé en ~5 min, val_mse=0.12139 identique au baseline 0.121.

**Toggle test révèle problème prévisible** :
- mse_drop0=0.12138843 vs mse_drop1=0.12138876 → Δ = 3.2e-7 négligeable
- mean_abs(pred_drop0 - pred_drop1) = 2.1e-5 (~0%)
- → **Le modèle n'utilise pas encore le canal OBS** après 1 epoch

**Diagnostic** : init `obs_init_std=1e-3` + 1 epoch × 8 cases (~400 updates) → obs_mlp n'a quasi pas bougé. Modèle s'est essentiellement maintenu à baseline (val_mse 0.121 inchangé). C'est **attendu et acceptable** pour un smoke aussi court.

**Validé par smoke** : forward path, init mapping (238 weights loaded, 4 new skipped), eval pipeline, MLflow logging, PBS chain training+eval. Infrastructure GREEN.

**NON validé par smoke** : utilité effective du canal OBS (besoin training plus long).

**Décision Boss (avec accord user) : proceed M_H1 full directement** sur 30 epochs × 6567 cases (~200k updates). Ajout d'un **monitor toggle-test périodique** (epoch 5, 10, 15, 20, 25, 30) comme proxy d'activation du canal — permet abort early si plat à epoch 10 au lieu de gaspiller 6h H100.

**Plan M_H1** :
- Init from `surrogate_v2_e2_stage1_smoke/best.pt` (essentiellement baseline)
- 30 epochs, full splits, walltime PBS 12h
- Monitor toggle test → bascule revise architecture (Strategy B/C/D) si plat à epoch 10

## M_H+ Department PARTIAL_GREEN 2026-05-26 — extension dataset OBS livrée, ERA5 chained PBS en cours

Mission lancée en parallèle de M_H0_smoke, terminée en ~35 min en background.

**Axe 1 (DEM tiles)** GREEN — 100 nouvelles tiles Copernicus DSM 30m via AWS S3 anonyme (`ingest_dem_copernicus.py`, 199 LOC). Couverture étendue : PT, sud ES, nord ES, UK sud, Atlantique côtier. Total 86 → 186 tiles. Resolve smoke validé sur Lisbon/Granada/Coimbra/Faro/Malaga/A Coruña.

**Axe 2 (multi-season ERA5)** PARTIAL — 3 jobs PBS chained `afterok` soumis pour MAM/JJA/SON 2023 (21982386 R + 21982395+96 H). ETA 24-48h. **2 fixes critiques** :
- `PRESSURE_LEVELS dataset_v2` réaligné `[1000,925,850,700,600,500,400,300,250,200]` (l'ancien engineer.md disait `[..., 200, 150]` SANS 600 = faux, vérifié 2026-05-26 sur grid.zarr réel)
- CDS 2026 cost-limit `403 "Your request is too large"` → mitigation `--max-days-per-req 7 --no-z`

**Axe 3 (multiprocessing)** GREEN — `ProcessPoolExecutor` dans `infer_at_stations.py` (+153 LOC) + PBS v2 (16 ncpus, n_prep_workers=8). Smoke 5× speedup (24/24 built, 34s wall). GPU 0% observé sur smoke court — attendu ≥60% en prod 300k pairings.

**Recommended next move** :
1. Attendre completion des 3 ERA5 chained jobs (24-48h). Le `afterok` chain les enchaine automatiquement.
2. Lancer re-inference v2 production : `qsub -v SEASON=mam2023 configs/hpc/infer_at_stations_v2.pbs` × 4 saisons après stores prêts.
3. Re-run audit sur parquet étendu `noaa_seasons_2022_2023.parquet`. Cible ≥200k pairings (winter actuel 57k + ~150k attendus depuis nouvelles tiles+saisons).

Memory candidates persistés : engineer.md §CDS 2026 cost-limit + §Copernicus DSM S3 endpoint + correction §PRESSURE_LEVELS. boss.md cette entrée.

## Phase H ouverte 2026-05-26 — option E.2 verrouillée (fine-tune surrogate v2 avec canal OBS)

**Décision user 2026-05-26** : Phase H = fine-tune **du surrogate v2 lui-même** avec un canal d'entrée OBS, plutôt qu'un 2e étage de correction post-process. Option E.2 = 2 étapes :
- **Stage 1 (E.1)** : fine-tune surrogate v2 sur dataset CFD existant (9252 cases) avec canal OBS synthétique = valeur CFD au pixel random + dropout 50%. Le modèle apprend à ancrer.
- **Stage 2** : fine-tune subséquent sur dataset OBS réel (~273k pairings après extension M_H+), loss au pixel station = OBS réelle. Le modèle apprend à corriger en plus d'ancrer.

**Décisions structurantes verrouillées** :
1. Granularité = surface (10 m AGL) seul. Pas de correction 3D, pas de classification a priori des régimes (thermique/catabatique).
2. Skip M_H0 research/comparaison A/B/D vs E (user fixe E.2 directement). Single Department Codex Engineer pour chaque mission.
3. Extension dataset (M_H+) = prérequis hard parallèle.

**Pourquoi E.2 plutôt que B/D post-process** : cohérence physique garantie par le receptive field du surrogate v2 lui-même (CNN/transformer 3D propage l'ancrage), pas de risque d'incohérence pixel-wise (option A) ni de paramètres dupliqués (option B). À l'inférence sans OBS : canal vide → surrogate fonctionne comme avant (zéro perte).

**Verrou conceptuel à surveiller** : les 9252 sites CFD training ne sont PAS co-localisés avec stations OBS. Stage 1 utilise OBS **synthétique** depuis ground truth CFD → le modèle apprend à recopier la valeur fournie au pixel. Stage 2 force la correction (loss OBS réelle) sur dataset étendu. Si dataset OBS reste insuffisant pour LOSO honest, risque que le stage 2 overfit climatologie au lieu de physique.

**Plan de secours si E.2 plafonne** : bascule vers option B (U-Net post-process sur champ entier) ou D (hybride U-Net pixel-wise training + full-conv inference) comme mission de secours en M_H3 (`RESCOPE Phase H'`).

**Cible empirique M_H2** : MAE global < 1.0 m/s + MAE wind_class=high < 1.5 m/s + bias wind_class=high > -0.5 m/s.

**Missions lancées 2026-05-26 en parallèle (background)** :
- **M_H0_smoke** : Department spawné, Codex/Claude Engineer pour smoke 5-10 cases CFD. ETA 2-3 jours.
- **M_H+** : Department spawné, extension dataset (DEM PT+sud ES + multi-season ERA5 + GPU multiprocessing). ETA 7-10 jours.

## VERDICT FINAL Phase G — M_G9 GO Phase H + Conditional Phase I (2026-05-26)

Production run M_G7 sur Aqua H100 (jobs 21901136 killed walltime → 21950018 + parquet checkpointing) a livré **57,010 pairings exploitables** (95 stations NOAA FR/ES dans bbox tiles DEM, fenêtre winter 2022-23 4 mois). C'est **8350× plus** que le M17 baseline (N=7 ICOS).

**Métriques globales** :
- MAE surrogate v2 = 1.476 m/s
- MAE ERA5 baseline raw = 1.361 m/s
- ΔMAE = -0.115 m/s (surrogate REGRESS -8.4%)
- Affine fit `speed_pred = 0.593·speed_obs + 1.475` (R²=0.606)

**M16/M17 verdict CONFIRMÉ statistiquement** : pattern affine (slope ~0.55, intercept ~+1.5) reproduit à grande échelle. Compression dynamique typique RANS k-ε + wall function.

**Décisions M_G9** (trinité) :

1. **Surrogate v2 raw = NO-GO déploiement**. MAE 1.476 m/s + REGRESS vs ERA5 brut → ne pas exposer aux applications fire/wind/paragliding sans correction.

2. **Phase H DNN bias correction = GO**. Dataset suffisant (57k pairings × 6 strata), pattern affine reproductible exploitable. Cible : DNN correction `speed_corr = f(speed_pred, terrain_class, height_bucket, wind_class, season, era5_freshness)` apprenant à fermer le 8% gap.

3. **Phase I alpine = CONDITIONAL**. summit (N=51) MAE 2.754, bias -2.259, R² 0.08 → RANS 6km insuffisant. Re-simuler ~50 cas alpine domaine 10×10×5 km **si Phase H DNN ne suffit pas pour cette classe**.

**Caveats prod run** :
- 2 jobs walltime kill 6h (CPU-bound `materialise_grid_zarr`, GPU 0-5% utilization). Optimisations future : multiprocessing.Pool sur les prep grid.zarr.
- Coverage limitée à tiles DEM dispo (FR + nord ES, ~26% de la population NOAA). PT zéro stations.
- Single window winter2223 (4 mois). Saisons MAM/JJA/SON à valider via Phase H.

**Code livré** :
- 10 commits Phase G (c577ecf → 01ae5fb)
- Pipeline complet : OBS unifié + surrogate input extract + inference batched + audit stratifié
- Memory engineer.md/department.md enrichies (12 patterns Zarr/PBS/audit)

## Phase G — M_G7 done, inférence surrogate aux stations validée (2026-05-23)

`services/module2b-surrogate/infer_at_stations.py` (514 LOC) +
`utils/inference_batch.py` (121 LOC) +
`configs/hpc/infer_at_stations.pbs` (49 LOC) livrent le pipeline batched
d'inférence aux pairings OBS.

**Smoke confirmé sur Perdigão rne01 mai 2017 IOP** : 10/10 pairings,
`speed_pred` 1.93-2.29 m/s vs `speed_obs` 1.11-2.27 m/s → magnitudes
physiques cohérentes, **légère surestimation typique V0** confirmant le
verdict de la session précédente (biais affine `U_cfd = 0.54·U_obs +
1.88` sur ICOS, M16). À ce stade le surrogate v2 reproduit fidèlement
ce qu'aurait donné un OpenFOAM v2 sur ces coords.

**best.pt local** : `/Users/guillaume/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
(116 MB downloadé Aqua→local).

**Caveats à propager à M_G8** :
- ERA5 d2m requis (legacy store sans d2m crash) → utiliser
  `era5_europe_spring2017_v2.zarr` pour smoke. Production large = re-
  ingest ERA5 hourly avec d2m.
- ERA5 Δt=6h → predictions constantes par blocs 6h ; le parquet
  contient `era5_time_delta_minutes` pour tracer.
- M_G8 doit stratifier audit par `abs(era5_time_delta_minutes)` et
  reporter ERA5-on-time vs ERA5-interpolated séparément.

How to apply : M_G8 = audit OBS vs surrogate stratifié. M_G7 production
peut être lancé via PBS Aqua H100 dès que le full NOAA ISD zarr est
disponible. Penser à régénérer ERA5 hourly Europe avec d2m comme
prérequis production-scale.

## Phase G — M_G6 done, extract input surrogate à coords arbitraires (2026-05-22)

`services/module2b-surrogate/extract_v2_input_at_coords.py` (338 LOC) +
`utils/inference_input.py` (341 LOC) livrent le pipeline pure
`(lat, lon, timestamp) → grid.zarr/input` au format consommé par
`WindV2DatasetViT`. Smoke 3/3 OK sur Perdigão, shapes validées
identiques au reference Aqua `ct_d_fire_0056_case_ts014`.

**Caveats à archiver** :
- z0_eff calculé en geometric mean omnidirectionnel sur patch 3 km
  radius (≠ upstream-only de prepare_inflow original). Choix justifié
  pour inférence sans direction connue, mais à valider en M_G8 vs OBS.
- ERA5 store `era5_europe_spring2017_v2.zarr` est Δt=6h ; pour pairings
  hourly stricts M_G7 doit interpoler en amont OU utiliser un store
  hourly (e.g. `era5_europe.zarr` si Δt=1h, à vérifier).
- Pas de normalisation appliquée à l'écriture du grid.zarr ; runtime
  `WindV2DatasetViT` applique `DEFAULT_NORM ∪ overrides
  (dataset_v2_norm.yaml)` à la lecture.

How to apply : M_G7 inférence doit (1) vérifier la cadence ERA5 du
store utilisé, (2) batched-load les grid.zarr produits par M_G6,
(3) appliquer `WindV2DatasetViT` puis le best.pt, (4) extraire U/V/W
au voxel central (90, 90, k(z_obs=elev+10m)) en interpolant via
`coords/z[180,180,40]`.

## Phase G ouverte 2026-05-21 — extension dataset OBS + inférence surrogate aux stations

**Stratégie validée par user** : pas de re-simulation OF. Inférence
surrogate v2 (best identifié = `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`,
val_mse=0.121, 24 AGL FWI 0-100 m, ERA5 surface in input) aux coordonnées
des stations OBS comme oracle CFD.

**Sources OBS ciblées** : Perdigão IOP 2017 (déjà ingéré) + SYNOP MF
~62 + AEMET ES ~250 + IPMA PT ~120 (via OGIMET archive) + ICOS 7
existants = **~480 stations** (sub-1000 du brief mais >10× seuil
LOSO honest).

**Caveats à archiver** :
- IPMA n'a pas d'archive open historique 2018-2023 → OGIMET (synops
  décodés gratuits) est le fallback obligatoire pour PT.
- AEMET API rate-limit 60 req/min → ingestion en 5 batchs régionaux
  séquentiels (Norte/Centro/Sur/Baleares/Canarias).
- ViT-Large (val_loss 0.4966) tenu en réserve si val_mse plus bas du
  base_agl100_k24 ne se généralise pas en M_G8.

**Pairing strategy** : pas d'index dans cases v2, M_G6 extrait input
surrogate à la volée depuis SRTM+WC+ERA5_europe.zarr. Stratification
4×3×4 = 48 cellules × 30 timestamps × 480 stations ≈ 690k pairings.

How to apply : avant tout audit final, vérifier que la stratification
n'a pas un trou critique sur une classe sous-représentée (alpine
summits, coastal sites). Penser à inclure les sites ICOS Méditerranéen
(FR-Pue, ES-LJu) dans la validation focused car ils sont les pivots
des résultats FWI antérieurs.

## VERDICT FINAL session 2026-05-18/20 : V0 statu quo, ne pas regen 9k (M17)

Après 6 retournements (Phase B → C → D → E → M14 → step-back → M16 → M17),
convergence vers **V0 = dataset v2 actuel = état de l'art**.

**Conclusions empilées** :
- Le "30% déficit" initial (`crop/inflow = 0.696`) est un **artefact
  d'audit** : ne vaut que pour n=20 cas vent fort ≥5 m/s (4% du
  sample500). Médiane sur tout le sample = 0.88. 41% des cas ont
  CFD ≥ ERA5.
- **V0 actuel bat ERA5** sur tall-tower 2020 : MAE médian −18% sur
  15 pairings. **CFD < OBS sur 8/15** (= ERA5 idem 10/15) ; **CFD > OBS
  sur 4/15**. Pas systématique.
- Tous les patches BC (V1, V8, V9, V10) sont des **degrés de liberté
  empiriques** qui flippent décision selon le test set (2D → 3D → réel).
- pg_geo "flip" vs "native" : la convention native est physiquement
  correcte (`source.x = -f·V_g`) mais la calibration sur geopotential
  850-700 hPa = mauvaise altitude (rotation Ekman 30-60° entre 750 hPa
  et surface). Sign optimal site-dépendant → patch fragile, à éviter.
- Sur OBS : biais affine `U_cfd = 0.54·U_obs + 1.88` (R²=0.43) hors
  alpine summits. **Mais ERA5 a la même compression** (a=0.47). C'est
  physique RANS k-ε + mesh grossier + wall functions, pas BC tuning.
- M17 POC XGBoost sur 7 sites : **ne bat pas CFD raw** en LOSO. Avec
  7 sites le ML apprend climatologie (top features = lat/lon/elev,
  CFD importance 0.04). Affine fix est pire que CFD raw partout.
- Dataset OBS actuel insuffisant pour ML correction utile.

**Stack final** = V0 = `inletOutlet top + zeroGradient + pg OFF +
z0=0.05 + wc native + Coriolis ON + Parente ambient + atmNutk +
simpleFoam k-ε 300 iter` = essentiellement **Venkatraman Perdigão WES
2023 adapté aux 9k sites v2**.

**Roadmap post-session** (voir REPORT.md §9) :
- **Phase G** (2-3 sem) : extension dataset OBS à ~1000+ stations
  (Perdigão IOP 2017 + SYNOP FR/ES/PT + ICOS multi)
- **Phase H** (1 sem) : DNN bias correction stratifié (terrain × height
  × wind class × season), inspiré du module 3 precip stratified QM
- **Phase I** (~10× compute, optionnel) : domaine 10×10×5 km pour
  ~50 cas alpine summit

**Leçons orchestration** :
1. 1 cas analytique ≠ vérité universelle (Phase B ridge 2D → Phase C
   multi-hill 3D inversait la décision)
2. Sign d'un patch empirique est suspect (pg_geo flip vs native)
3. Audits CSV proxies fragiles (crop/inflow médian sur 500 cas non
   stratifiés cachait que 96% du sample est vent faible où CFD ≥ ERA5)
4. **Bench OBS direct AVANT d'investir dans stack BC** — on a passé
   3 jours sur BC tuning avant de mesurer que V0 actuel bat ERA5
5. ML correction sur N=7 sites = climatologie pas physique. Seuil
   empirique : ~30-50 sites min pour LOSO honest
6. **Adversarial dual-spawn (A+B convergents) = solide** — les deux
   Departments step-back ont indépendamment conclu V0 statu quo

**Le commit `43f5e90` ("V1 retained for 9k regen") est mis en doute
par cette session**. À traiter via commit fix qui pointe vers le
verdict V0 final.

## Phase E retourne V10 → V1 retenu pour regen 9k (2026-05-18, M13)

Phase E sur 5 sites v2 réels Pop A FR continental (ct_c_morpho_0000,
ct_d_fire_0017, ct_e_mountain_0023, ct_f_wind_onshore_0001,
ct_g_paragliding_0006) : **V1 (best-stack flip) bat V10 (top open
native) sur 4/4 sites complets** en ratio physique
`crop_mean / ERA5_U10_nominal`.

Mean ratio across sites : V0=1.35, V10=1.89, **V1=2.31**.

Le proxy `edge_W` (vent au bord W amont CFD) n'est PAS un bon proxy
d'inflow car le forçage pg_geo change la dynamique d'inflow (V1
edge_W=5.21 vs V10 edge_W=1.14 sur ct_d_fire_0017). La métrique
physique correcte est `crop_mean / ERA5_U10[1,1]` (extrait de
`input/era5_surface/u10` du grid.zarr).

**Pour la regen 9k : V1 final**. Stack :
- top U : slip
- top p : fixedValue 0
- pg_geo : flip
- z0_wall : 0.005 m
- z0 field : wc_capped_0.05
- Coriolis : on

**Leçon orchestration** : le multi-hill analytique a induit en
erreur (recommandait V10) parce que la topographie analytique
manque les structures de recirculation du vrai terrain. Toujours
valider sur sites v2 réels (au moins 5, diversifiés Pop A) avant
adoption finale d'un stack.

**Bug PBS phaseE_5sites.pbs** : oubliait `writeCellCentres` (donc
`0/Cx` manquait pour l'export). Fix : `phaseE_export_only.pbs`
séquentiel qui fait writeCellCentres + export sur cases déjà solved.
Pattern à intégrer si on étend le builder pour mode `site_real`.

## V10 native > V8 flip — décision révisée (2026-05-18, M12)

Test M12 (V10 = V9 + pg native) sur multi-hill : V10 bat TOUTES les
métriques. crop_mean=0.808 (Δ vs V9 flip = +0.176, Δ vs V8 = +0.142),
crest_max=1.961 (le plus haut), flat_mean=0.724.

**Convention pg confirmée** :
- native : `source.x += +7.587e-04 × V`, `source.y += +5.192e-04 × V`
- flip   : opposé (multiplication par -1)
- Sur 0056 (Sierra Andaluza, 37°N, flux 270°W) → native correct.
- Sur 0170 (Skiathos, bug WC) → flip aidait à compenser un fit ERA5
  corrompu. Workaround spécifique, pas règle générale.

**Stack régen 9k révisé (pré-Phase E)** : V10 = inletOutlet + zeroGrad
top + pg native + z0=0.05 + wc native. Plus simple que V8 ET meilleur.

Phase E à valider sur 5 sites v2 diversifiés solved (éviter Pop B
steep terrain qui crash RANS).
