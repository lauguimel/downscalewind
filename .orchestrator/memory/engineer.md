# Engineer memory — DownscaleWind OF / canary work

## OBS append_obs_data MUST be vectorized (2026-05-25, NOAA prod root cause)

The ORIGINAL bottleneck on NOAA prod (~11 min/station, 60+ h ETA) was
NOT the chunks layout — it was a per-timestamp RMW loop inside
`shared/obs_io.py:append_obs_data`:

```python
# BAD (315k Zarr ops per station for 52608 ts × 6 vars)
for row_idx, time_idx in enumerate(time_indices):
    current = np.asarray(arr[time_idx, station_idx, :], dtype=np.float32)
    current[dest_height_idx] = values[row_idx]
    arr[time_idx, station_idx, :] = current
```

Fixed by reading the full station column once, mutating in memory,
writing back once:

```python
# GOOD (1 read + 1 write per var per station)
time_selector = (slice(t_lo, t_hi), station_idx, slice(None))  # contiguous case
current = np.asarray(arr[time_selector], dtype=np.float32)
for col, h_idx in enumerate(dest_height_idx):
    current[:, h_idx] = values[:, col]
arr[time_selector] = current
```

Result: 362 stations × 6 vars × 52608 ts ingested in ~30 min (vs
60+ h with the loop). 1000× speedup.

How to apply: ANY zarr writer that takes per-element data MUST stage
the full slice in memory before writing. Never loop element-by-element
through a Zarr array — every access decompresses+recompresses a whole
chunk. The chunks layout optimization helps but is secondary.

## OBS unified Zarr chunks: bundle stations (2026-05-24, NOAA prod fix)

`_obs_unified.write_unified_obs_zarr` originally used
`chunks = (min(720, n_times), 1, n_heights)` → 1 chunk per
(time_block, station, height). For NOAA prod (356 stations × 52608
hourly timestamps), this generated **~26344 chunks per variable ×
6 variables = ~158k fsync+metadata operations** → 31 MB/h write
rate, 80+ h ETA.

Fix:
```python
chunks = (
    min(8760, n_times),       # up to one year per chunk
    max(1, min(n_stations, 64)),  # bundle stations
    n_heights,
)
```

→ ~36 chunks per variable, total writes O(seconds) instead of hours.
Validated on Perdigão smoke (13 chunks total vs 64078 before).

How to apply: any future Zarr writer for tall sparse multi-station
data MUST bundle the small axis (here: stations) into the chunk.
Single-axis-1 chunks fragment filesystem badly with Blosc + Zarr V3
metadata fanout (1 file per chunk).

## ERA5 hourly + d2m Europe pipeline ready (2026-05-23, M_G6.5)

`services/data-ingestion/ingest_era5_europe_hourly.py` produit un Zarr
ERA5 Europe hourly (Δt=1h) avec **d2m** inclus, débloquant M_G7
production sans le bug "predictions constantes en blocs 6h".

Variables surface canoniques pour surrogate v2 (alias CDS) :
```
["10m_u_component_of_wind", "10m_v_component_of_wind",
 "2m_temperature", "2m_dewpoint_temperature"]
```

**Ne PAS oublier `2m_dewpoint_temperature` (d2m)** sinon
`build_era5_baseline_tensor(mode='surface')` crash au runtime.

Pressure levels canoniques (dataset v2) :
```
[1000, 925, 850, 700, 500, 400, 300, 250, 200, 150]
```

Différent de l'ancien `era5_europe.zarr` (qui avait 600 au lieu de 150).
Si on re-ingère le legacy, harmoniser.

How to apply : pour M_G7 production massive, utiliser
`data/raw/era5_europe_hourly_*.zarr` (à créer en re-running
ingest_era5_europe_hourly.py sur la fenêtre de prod 2018-2023).

## ERA5 d2m required for surrogate v2 surface input (2026-05-23, M_G7)

`src.dataset_v2.build_era5_baseline_tensor(mode='surface')` consumes
`{t2m, d2m, u10, v10}` from the ERA5 store. Legacy
`data/raw/era5_europe.zarr` is missing `d2m` → crashes the surrogate.
Use `era5_europe_spring2017_v2.zarr` (has d2m) for IOP smoke. For
production at Europe scale, re-ingest ERA5 hourly with d2m included.

How to apply: any new ERA5 ingester for Phase G / Phase H must include
d2m. Smoke tests must verify presence before consuming.

## ERA5 6-hour cadence yields predictions constant in 6-h blocks (2026-05-23, M_G7)

When the ERA5 store has Δt=6h (typical for cheap Europe extracts),
`extract_v2_input_at_coords` rounds OBS timestamps to the nearest ERA5
sample. Consecutive hourly OBS within the same 6h ERA5 window receive
the SAME surrogate input → SAME prediction. Visible in smoke as 4-then-6
identical rows.

Mitigation: ingest ERA5 hourly for the inference period, OR aggregate
OBS to 6h cadence for fair comparison. Track via
`era5_time_delta_minutes` column in the parquet output.

How to apply: M_G8 audit must stratify by `abs(era5_time_delta_minutes)`
and report performance separately for "ERA5-on-time" (Δt=0) vs
"ERA5-interpolated" (Δt>30 min).

## Surrogate v2 input dim formula (2026-05-22, M_G6)

For the production checkpoint
`~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`,
the input expected by `WindV2DatasetViT.__getitem__` is exactly:

```
era5_flat_dim = 4 * N_p * 9 + 4 * 9 + N_p + 2
              = (4 vars × N_p plevels × 3×3) + (4 surf vars × 3×3) + N_p plevels + (lat + z0_eff)
```

With `N_p=10` pressure levels → `era5_flat_dim = 408`. Plus
`terrain_2d: (2, 180, 180)`, `geo: (2, 180, 180, 24)` for the AGL grid.

How to apply: any extract_v2_input pipeline must respect this signature
exactly. The contract is enforced at runtime in
`services/module2b-surrogate/src/dataset_v2_vit.py` — read it before
inventing fields.

## Zarr 3.1.5 `create_dataset` removed; use `create_array` (2026-05-22, M_G6)

`zarr.Group.create_dataset(name, data=arr)` fails with
`TypeError: missing 'shape'` on zarr 3.1.5. Pattern:

```python
a = g.create_array(name, shape=arr.shape, dtype=arr.dtype, ...)
a[...] = arr
```

How to apply: every new Zarr writer in this codebase. The existing
`shared/data_io.py` and `shared/obs_io.py` already follow this pattern.

## `np.datetime64(int, "ns")` does not work; cast via array (2026-05-22, M_G6)

`np.datetime64(int_ns, "ns")` raises. Use:

```python
ts = np.array(int(int_ns)).astype("datetime64[ns]")
```

How to apply: any int64-ns → ISO conversion. Recurring pattern in
timestamp handling across pipelines.

## NOAA isd-history.csv columns have SPACES (not underscores) (2026-05-21, M_G2bis)

The NOAA CSV ships columns `"STATION NAME"`, `"ELEV(M)"`, etc. (literal
spaces, not underscores). After `pd.read_csv`, normalize via:

```python
history.columns = [str(c).strip().replace(" ", "_") for c in history.columns]
```

Otherwise downstream code that expects `STATION_NAME` raises
`ValueError: isd-history.csv missing required columns`.

How to apply: any NOAA / NOAA-derived CSV ingester must do this column
normalization at read time. Also true for legacy NOAA CDO datasets.

## `station_id_str` must be Python str ≤16 chars, not bytes (2026-05-21, M_G3)

When appending data to `shared/obs_io.py` via `append_obs_data(..., station_id=...)`,
pass a Python `str` (truncated to ≤16 chars matching the `S16` array width).
The helper does `stations["station_id"].to_numpy() == station_id` (string
compare) and encodes to bytes internally. Passing bytes causes silent
type-coerce failures that yield zero matches and no error.

How to apply: any Engineer writing to `obs_unified_*.zarr/` uses string
station IDs throughout (`f"aemet_{idema}"[:16]`, `f"synop_{numer_sta}"`,
etc.), never `b"..."`.

## Zarr 3.1.5 UnstableSpecificationWarning on fixed-width bytes (2026-05-21, M_G5)

Zarr 3.1.5 emits `UnstableSpecificationWarning` for fixed-width bytes
arrays (`S16`, `S2`, `NullTerminatedBytes`). When writing
`stations/{station_id, source, country}` or similar bytes arrays, the
warning is noisy. `shared/obs_io.py` filters the warning at import:

```python
import warnings
import zarr
warnings.filterwarnings("ignore", category=zarr.UnstableSpecificationWarning)
```

How to apply: any new Engineer code that creates bytes arrays in Zarr
3.x stores must replicate this filter, OR accept noisy smoke output.

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
