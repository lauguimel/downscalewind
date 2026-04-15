# Campaign: complex_terrain_v1

Multi-site CFD campaign for wind + T + q downscaling surrogate, focused on
terrain where CFD adds value over ERA5 25km (fire-prone Mediterranean, mountains,
onshore wind farms in complex terrain). Supersedes the `9k` campaign (which
contained 450 offshore/coastal sites of limited scientific value).

## Scope

~750 sites × 15 timestamps = ~11 250 runs.

| Category | N | Rationale |
|----------|---|-----------|
| D_fire    | 250 | Mediterranean fire-prone (Provence, Cévennes, Corse, Catalogne, Alentejo, Greece, Sicily, Sierra Nevada) |
| E_mountain | 200 | High relief (Alps FR/IT/CH, Pyrenees, Dolomites, Apennines, Carpathians) |
| F_wind_onshore | 150 | Wind farms in complex terrain (Alps, Pyrenees, Galicia, Hautes-Fagnes) — NOT offshore |
| C_morpho  | 150 | Random complex terrain sampling (stratified slope × aspect) |

## Pipeline improvements vs 9k

1. **Mesh refined**: 30m horizontal (vs 67m) to match SRTM native + reduce aspect ratio (AR ≈ 8 vs 18)
2. **Surface anchor in inflow**: t2m / d2m / u10 / v10 injected via PCHIP spline
3. **Per-site ERA5**: each site has its own ERA5 3×3 BC (vs Perdigão 1D fallback)
4. **Direct grid export**: OF case → `grid.zarr` (128×128×32) in one step, no intermediate site zarr
5. **Complete metadata**: physics flags, mesh info, solver diagnostics, site metadata preserved for training reproducibility

## Directory layout

```
complex_terrain_v1/
├── README.md              # this file
├── sites.csv              # 750 sites (site_id, lat, lon, elev, group, country, ...)
├── run_matrix.csv         # ~11250 runs (run_id, site_id, timestamp, lat, lon, group, ...)
├── manifests/
│   ├── sites.yaml         # authoritative site metadata (machine-readable)
│   ├── campaign.yaml      # campaign-level parameters (mesh, solver, ERA5 source)
│   └── splits.yaml        # train/val/test site assignment (geographic, watertight)
├── cases/                 # one grid.zarr per run (direct output of new pipeline)
│   └── {site_id}_case_ts{NNN}/
│       ├── grid.zarr/
│       └── inflow.json
├── meshes/                # cached meshes per site (1 mesh reused for 15 timestamps)
│   └── {site_id}/
│       ├── constant/polyMesh/
│       └── mesh_meta.json
├── inflow_profiles/       # intermediate inflow files for debug/audit
└── debug/                 # validation cases (Perdigão IOP, FR-Pue heatwave, ES-LJu synoptic)
```

## Manifest schemas

### `manifests/sites.yaml`

Per-site metadata, authoritative source for site attributes.

```yaml
schema_version: 1
campaign: complex_terrain_v1
created: '2026-04-14'
sites:
  - site_id: ct_d_fire_000
    group: D_fire
    lat: 43.72
    lon: 4.12
    elevation_m: 185.0
    country: FR
    climate_zone: Csa            # Köppen
    std_elev_local_m: 48.0       # std of SRTM in 5km radius
    mean_slope_deg: 8.2
    era5_source: era5_campaign_v3/era5_site_ct_d_fire_000.zarr
    era5_timestamps_file: run_matrix.csv
    nearest_obs:
      station_id: null           # populated post-hoc if match with Meteo-France/ICOS
      station_network: null
      distance_km: null
  # ... ×~750
```

### `manifests/campaign.yaml`

Campaign-wide parameters (mesh, solver, physics).

```yaml
schema_version: 1
campaign: complex_terrain_v1
created: '2026-04-14'
mesh:
  inner_size_m: 2000
  horizontal_resolution_m: 30.0
  cells_per_block_xy: 6        # 6×6 per 200m block → 30m
  inner_blocks: 10
  height_m: 2500
  cells_z: 80
  grading_z: 30
  first_cell_m_approx: 3.7
  aspect_ratio_first_cell: 8.1
  cylinder_radius_m: 7000
solver:
  name: simpleFoam
  n_iter: 2000
  turbulence_model: kOmegaSST
  transport_T: passive
  transport_q: passive
physics:
  coriolis_enabled: true
  canopy_enabled: false
  wall_function: parente2011
inflow:
  pressure_levels_used: true
  surface_anchor: [t2m, d2m, u10, v10]
  interpolation: pchip           # u/v/T/q profiles
  source: data/raw/era5_campaign_v3/
era5_ingest:
  domain_bbox: [35.75, -10.00, 55.25, 25.00]  # [lat_min, lon_min, lat_max, lon_max]
  periods:
    - [2017-05-01, 2017-06-30]   # IOP spring
    - [2022-06-01, 2022-09-30]   # JJA fire season
  levels_hPa: [200, 300, 500, 700, 850, 925, 975, 1000]
  single_request: true            # ingest_era5_europe.py
```

### `manifests/splits.yaml`

Geographic split (watertight per site).

```yaml
schema_version: 1
campaign: complex_terrain_v1
split_strategy: geographic_watertight
ratio: [0.70, 0.15, 0.15]
train:
  - ct_d_fire_000
  - ct_e_mountain_042
  # ...
val:
  - ct_d_fire_001
  # ...
test:
  - ct_d_fire_003
  # ...
```

## Data product per case: `grid.zarr`

One zarr store per CFD solve, 128×128×32 grid (log-spaced AGL 5m → 5000m).

### Inputs (read at training time)
- `input/terrain`: (128, 128) elevation [m]
- `input/z0`: (128, 128) WorldCover roughness [m]
- `input/era5/{u, v, T, q, k}`: (32,) 1D profile at AGL levels
- `input/era5_3d/{u, v, T, q, k}`: (3, 3, 32) 3×3 ERA5 grid around site
- `input/era5_surface/{t2m, d2m, u10, v10}`: (3, 3) ERA5 surface vars

### Targets (from CFD)
- `target/U`: (128, 128, 32, 3) velocity
- `target/T`: (128, 128, 32) temperature
- `target/q`: (128, 128, 32) humidity
- `target/k`: (128, 128, 32) TKE *(optional)*
- `target/epsilon`: (128, 128, 32) dissipation *(optional)*
- `target/nut`: (128, 128, 32) turbulent viscosity *(optional)*

### Residuals (CFD - ERA5 lifted, for training stability)
- `residual/U`, `residual/T`, `residual/q`, `residual/k`

### Coordinates
- `coords/x`, `coords/y` (2D grid in local UTM metres, centered on site)
- `coords/z_agl` (32,) log-spaced heights
- `coords/elev` (128, 128) terrain elevation at each grid cell

### Attributes
- `site_id`, `group`, `timestamp_iso`, `era5_source_path`
- `mesh.*` (inner_size, resolution, cells_z, AR, first_cell_m)
- `physics.*` (coriolis, canopy, T_passive, q_passive, turbulence_model)
- `inflow.*` (u_hub, u_star, z0_eff, T_ref, q_ref, Ri_b, wind_dir, t2m_K, d2m_K, u10_ms, v10_ms)
- `solver.*` (n_iter, wall_time_s, residual_U, residual_p, residual_T, converged)

## Training split policy

Watertight geographic: each site is exclusively in train, val, or test. Never
mix timestamps of the same site across splits. Enforced by `dataset_wind9k.py`
(checks `manifests/splits.yaml` against site_id in each case).

## Commands reference

```bash
# 1. Build sites
python services/module2a-cfd/build_sites_complex_terrain.py \
  --srtm data/raw/srtm_europe.tif \
  --worldcover data/raw/worldcover_europe.tif \
  --climate data/raw/koppen_geiger.tif \
  --n-d-fire 250 --n-e-mountain 200 --n-f-wind 150 --n-c-morpho 150 \
  --out data/campaign/complex_terrain_v1/sites.csv \
  --manifest data/campaign/complex_terrain_v1/manifests/sites.yaml

# 2. Extract ERA5 per site (from europe.zarr)
python services/data-ingestion/extract_era5_per_site.py \
  --sites data/campaign/complex_terrain_v1/sites.csv \
  --europe data/raw/era5_europe.zarr data/raw/era5_europe_2022jja.zarr \
  --out data/raw/era5_campaign_v3/

# 3. Build run matrix
python services/module2a-cfd/build_run_matrix_9k.py \
  --sites data/campaign/complex_terrain_v1/sites.csv \
  --n-per-site 15 \
  --out data/campaign/complex_terrain_v1/run_matrix.csv

# 4. Submit PBS array (post-maintenance)
ssh aqua "cd ~/dsw && qsub -t 1-10 configs/hpc/campaign_complex_terrain_v1.pbs"
```

## Validation gates before production

1. ✅ Inflow smoke test : FR-Pue canicule 2022-08-10 → T_surface ≈ 33°C, u@10m ≈ u10 ERA5
2. ⏳ Perdigão IOP test : 2017-05-04 22:00 + 2017-06-05 12:00 → profil vertical tours vs obs
3. ⏳ ES-LJu synoptic test : 2022-02-25 → u@sensor vs obs (27.9 m/s)
4. ⏳ Mesh AR check : max AR first cell < 10 partout (AR_max stocké en attrs)
5. ⏳ Metadata completeness : tous les attrs présents et non-NaN sur 10 cases random
