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
