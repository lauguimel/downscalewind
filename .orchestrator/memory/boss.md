# Boss memory — DownscaleWind orchestrator

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
