#!/bin/bash
# run_tbm_remote.sh — Run TBM study entirely on UGA server
#
# Usage (from local Mac):
#   ssh UGA 'bash /home/guillaume/dsw/scripts/run_tbm_remote.sh /home/guillaume/dsw/configs/poc_tbm_physics.yaml 24'
#
# Everything runs on UGA: STL generation, TBM mesh, init, solve, reconstruct.
# No scp needed. Results stay on UGA, rsync back when needed.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [nprocs]}"
NPROCS="${2:-24}"
PYTHON=/home/guillaume/miniconda3/bin/python
SCRIPTS=/home/guillaume/dsw/scripts
SITE_CFG=/home/guillaume/dsw/configs/perdigao.yaml
SRTM=/home/guillaume/dsw/srtm_perdigao_30m.tif
ERA5=/home/guillaume/dsw/era5_perdigao.zarr
TBM_IMAGE=terrainblockmesher:of24
OF_IMAGE=microfluidica/openfoam:latest

# Parse study name from config
STUDY=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['study']['name'])")
CASES_DIR=/home/guillaume/dsw/cases/$STUDY
mkdir -p $CASES_DIR

echo "[$(date +%H:%M:%S)] === Study: $STUDY ($NPROCS cores) ==="

# Generate everything with Python
cd $SCRIPTS
$PYTHON -c "
import yaml, json, shutil, subprocess, sys, numpy as np
from pathlib import Path
from generate_mesh_tbm import generate_mesh_tbm
from generate_mesh import generate_mesh, dem_to_stl
from prepare_inflow import prepare_inflow

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
with open('$SITE_CFG') as f:
    site_cfg = yaml.safe_load(f)

study = cfg['study']
cases_dir = Path('$CASES_DIR')
site_lat = site_cfg['site']['coordinates']['latitude']
site_lon = site_cfg['site']['coordinates']['longitude']

# Inflow
inflow_json = cases_dir / 'inflow.json'
if not inflow_json.exists():
    print(f'[{__import__(\"datetime\").datetime.now():%H:%M:%S}] Generating inflow...')
    prepare_inflow(era5_zarr='$ERA5', timestamp=study['timestamp'],
        site_lat=site_lat, site_lon=site_lon, output_json=inflow_json)

# STL
stl_path = cases_dir / 'terrain.stl'
if not stl_path.exists():
    print(f'[{__import__(\"datetime\").datetime.now():%H:%M:%S}] Generating STL...')
    first_case = next(iter(cfg['cases'].values()))
    tbm_cfg = first_case.get('tbm', cfg.get('terrainBlockMesher', {}))
    radius = tbm_cfg.get('cylinder', {}).get('radius', 7000) * 1.1
    DL = 1/111000; DO = 1/(111000*np.cos(np.radians(site_lat)))
    dem_to_stl(srtm_tif=Path('$SRTM'), out_stl=stl_path,
        bounds_lonlat=(site_lon-radius*DO, site_lat-radius*DL,
                       site_lon+radius*DO, site_lat+radius*DL),
        resolution_m=30, site_lat=site_lat, site_lon=site_lon)

# Generate cases
for case_id, case_cfg in cfg['cases'].items():
    case_dir = cases_dir / f'case_{case_id}'
    tbm_cfg = case_cfg.get('tbm', cfg.get('terrainBlockMesher', {}))
    print(f'[{__import__(\"datetime\").datetime.now():%H:%M:%S}] === {case_id} ===')

    # TBM mesh
    if not (case_dir / 'constant/polyMesh/points').exists():
        print(f'  Meshing...')
        generate_mesh_tbm(stl_path=stl_path, case_dir=case_dir, config=tbm_cfg, keep_tmp=False)

    # Templates
    n_sec = tbm_cfg.get('cylinder',{}).get('n_sections',8)
    lat_patches = [f'section_{i}' for i in range(n_sec)]
    if not (case_dir / 'system/controlDict').exists():
        print(f'  Templates...')
        generate_mesh(site_cfg=site_cfg, resolution_m=1000, context_cells=1,
            output_dir=case_dir, srtm_tif=None, inflow_json=inflow_json,
            domain_km=tbm_cfg.get('cylinder',{}).get('radius',7000)*2/1000,
            domain_type='cylinder', solver_name=case_cfg.get('solver','simpleFoam'),
            thermal=case_cfg.get('thermal',False),
            coriolis=case_cfg.get('coriolis',True),
            transport_T=case_cfg.get('transport_T', study.get('transport_T',False)),
            n_iter=study.get('n_iterations',500),
            write_interval=study.get('write_interval',100),
            lateral_patches=lat_patches)

    # decomposeParDict
    dp = case_dir / 'system/decomposeParDict'
    if not dp.exists():
        dp.write_text('FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\n'
                      f'numberOfSubdomains $NPROCS;\nmethod scotch;\n')

    # Copy helpers
    for s in ['init_from_era5.py','reconstruct_fields.py']:
        src = Path('$SCRIPTS') / s
        if src.exists(): shutil.copy2(src, case_dir / s)
    shutil.copy2(inflow_json, case_dir / 'inflow.json')

    # writeCellCentres + init
    u_file = case_dir / '0/U'
    if not u_file.exists() or 'nonuniform' not in u_file.read_text()[:3000]:
        print(f'  Init fields...')
        subprocess.run(['docker','run','--rm','-v',f'{case_dir}:/case','-w','/case',
            '$OF_IMAGE','bash','-c','postProcess -func writeCellCentres -time 0 > /dev/null 2>&1'],
            capture_output=True, timeout=300)
        subprocess.run([sys.executable,'init_from_era5.py','--case-dir','.','--inflow','inflow.json'],
            cwd=case_dir, capture_output=True, timeout=120)

    print(f'  Ready')
print('=== All cases generated ===')
"

echo "[$(date +%H:%M:%S)] === Solving all cases ==="

# Solve each case
for case_dir in $CASES_DIR/case_*; do
    case_name=$(basename $case_dir)
    NPROCS_CASE=$($PYTHON -c "
import yaml
cfg=yaml.safe_load(open('$CONFIG'))
cid='${case_name#case_}'
print(cfg['cases'].get(cid,{}).get('nprocs',$NPROCS))
")

    echo "[$(date +%H:%M:%S)] Solving $case_name ($NPROCS_CASE cores)..."

    if [ "$NPROCS_CASE" -gt 1 ]; then
        docker run --rm --cpus=$NPROCS_CASE --memory=16g \
            -v "$case_dir":/case -w /case $OF_IMAGE bash -c \
            "foamDictionary system/decomposeParDict -entry numberOfSubdomains -set $NPROCS_CASE && \
             rm -rf processor* && \
             decomposePar -force > /dev/null 2>&1 && \
             mpirun --allow-run-as-root -np $NPROCS_CASE simpleFoam -parallel > /case/log.simpleFoam 2>&1"

        # Reconstruct
        cd $case_dir
        $PYTHON reconstruct_fields.py --case-dir . --time 500 --write-foam 2>/dev/null
        cd $SCRIPTS
    else
        docker run --rm --cpus=4 \
            -v "$case_dir":/case -w /case $OF_IMAGE bash -c \
            "simpleFoam > /case/log.simpleFoam 2>&1"
    fi

    echo "[$(date +%H:%M:%S)] Done: $(grep ClockTime $case_dir/log.simpleFoam 2>/dev/null | tail -1)"
done

echo "[$(date +%H:%M:%S)] === ALL DONE: $STUDY ==="
