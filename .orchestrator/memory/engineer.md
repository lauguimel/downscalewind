# Engineer memory — DownscaleWind OF / canary work

## audit_terrain_canary.py order: figure BEFORE CSV (2026-05-18)

`audit_terrain_canary.py` writes the PNG figure BEFORE the CSV. If
matplotlib import / save fails, the CSV is lost AND the
solve_status JSON stays at "running" (never reaches "ok").
Fragile in environments where matplotlib is not installed.

How to apply: when running audit in a new env, first `python -c
"import matplotlib"` to verify, OR run audit with figure step
mocked. A SPLIT mission could reorder to write CSV first, then
figure — until then, beware.

The fix `pip install matplotlib` in conda `fuxicfd` on Aqua is
persistent (added 2026-05-18 by M8-fix).

## writeCellCentres gotcha

`run_multisite_campaign.run_write_cell_centres(mesh_dir)` will overwrite
`<case>/0/p` with a minimal zeroGradient stub IF `0/Cx` is not present.
This is fatal for canary cases that have a real `0/p` with proper BCs
(mappedFile lateral, fixedValue top, etc.).

Always pattern : **backup 0/p → call → restore 0/p**.

```python
p_file = case / "0" / "p"
p_backup = case / "0" / "p.preCC.bak"
if p_file.exists():
    shutil.copy2(p_file, p_backup)
try:
    run_write_cell_centres(case)
finally:
    if p_backup.exists():
        shutil.move(str(p_backup), str(p_file))
```

## Apptainer / weka paths

Aqua `/scratch/maitreje/...` is a symlink to `/mnt/weka/scratch/maitreje/...`.
Inside Apptainer the bind-mount uses the resolved path. The export
script uses `/mnt/weka/scratch/...` explicitly — both work, but a
manifest that mixes the two prefixes is fragile. Always prefer the
realpath form returned by `os.path.realpath()`.

## z0_treatment canary case layout (v2)

```
<canary_root>/
  z0_treatment_canary_manifest.json
  solve_status/<treatment>.json   # running | ok | solve_failed | z0_gen_failed | missing
  cases/case_ts000_<treatment>/
    0/                              # initial fields, U/p/k/eps/T/q
    constant/polyMesh/              # real copy (NOT symlink — Apptainer can't follow)
    constant/boundaryData/section_{0..7}/  # real copies
    constant/boundaryData/terrain/0/z0     # written by generate_z0_field for wc / wc_capped
    constant/fvOptions             # Coriolis + pg_geo with --pg-sign flip
    system/{controlDict,fvSchemes,fvSolution,decomposeParDict}
    300/                           # solve output (U, p, k, eps, T, q)
  export_and_audit_z0_treatment.sh
```

Treatment names get sanitized as directory names: `wc_cap_0.05` → `wc_cap_0p05`.

## Multi-hill builder available (2026-05-18)

`services/module2a-cfd/analysis/build_terrain_canary.py` supports
`--terrain-kind multi_hill --variant {V0..V8,V0n,V1n}` and
`--wind-dir {270|0}`. Spec: 3 cos² hills triangle asymétrique
(N: H=250/L=800, SE: H=200/L=600, SW: H=300/L=1000), domain
6×6×2.5 km, mesh 180×180×40. Knobs map cleanly to top_U / top_p /
pg_geo / z0_wall / z0_field.

Audit script: `audit_terrain_canary.py --terrain-kind multi_hill`
produces CSV with stats per-mask in {crest_N, crest_SE, crest_SW,
lee_N, lee_SE, lee_SW, crop, flat, pdf}.

**Known bug kept-with-flag**: 2 m AGL rows in the CSV carry
`comment=known_buggy_inflow_speed_at_2m`. Filter `height_m != 2`
in any OFAT decision logic until the underlying
`audit_v2_teacher_wind.inflow_speed_at` is fixed.

## Multi-hill masks must be per-hill (2026-05-18)

For ablation cases with multiple hills, do NOT use a single global
crest/lee mask — it hides per-hill asymmetry (windward vs lee, big vs
small hill). Define `crest_k = {terrain ≥ z_base + 0.85·H_k}` and
`lee_k = {projection s on wind dir ∈ [0.25, 2.0]·L_k} ∩ {terrain ≤
z_base + 0.3·H_k}` per hill k, then aggregate (max-over-hills,
mean-over-hills) for the global stat. A single global crest mask
typically picks only the tallest hill.

## ESA WC z0 lookup (matches generate_z0_field.py)

10 tree → 0.5  | 20 shrub → 0.05  | 30 grass → 0.03  | 40 crop → 0.05
50 built → 1.0 | 60 bare → 0.005 | 70 snow → 0.002 | 80 water → 0.0002
90 wet → 0.05  | 95 mangrove → 0.5 | 100 moss → 0.005
