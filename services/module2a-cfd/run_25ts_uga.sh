#!/usr/bin/env bash
# run_25ts_uga.sh — Run 25-timestamp campaign 100% on UGA
#
# Prereqs on UGA:
#   - /home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/ (reference case with mesh)
#   - /home/guillaume/dsw/era5_perdigao.zarr (ERA5 data)
#   - /home/guillaume/dsw/scripts/ (Python scripts + templates)
#   - /home/guillaume/miniconda3/ (Python 3.13 + zarr 3.1.5 + scipy)
#
# What it does:
#   1. Copy polyMesh from reference case to all 25 cases
#   2. For each timestamp: prepare_inflow → render templates → init_from_era5
#   3. Solve 2 cases at a time (24 cores each on 48-core machine)
#
# Usage (from local):
#   scp run_25ts_uga.sh UGA:/home/guillaume/dsw/scripts/
#   ssh UGA "cd /home/guillaume/dsw/scripts && bash run_25ts_uga.sh"

set -euo pipefail

PYTHON=/home/guillaume/miniconda3/bin/python
SCRIPTS=/home/guillaume/dsw/scripts
CASES=/home/guillaume/dsw/cases/poc_tbm_25ts
ERA5=/home/guillaume/dsw/era5_perdigao.zarr
OF_IMAGE=microfluidica/openfoam:latest
NPROCS=24
N_ITER=500
SITE_LAT=39.716
SITE_LON=-7.740

# Timestamps from the CSV (25 entries)
TIMESTAMPS=(
"2017-05-04 18:00:00"
"2017-05-07 12:00:00"
"2017-05-10 06:00:00"
"2017-05-11 12:00:00"
"2017-05-12 12:00:00"
"2017-05-12 18:00:00"
"2017-05-13 12:00:00"
"2017-05-15 00:00:00"
"2017-05-15 06:00:00"
"2017-05-15 12:00:00"
"2017-05-19 12:00:00"
"2017-05-24 00:00:00"
"2017-05-25 06:00:00"
"2017-05-29 00:00:00"
"2017-05-29 18:00:00"
"2017-05-30 12:00:00"
"2017-06-01 12:00:00"
"2017-06-05 18:00:00"
"2017-06-09 18:00:00"
"2017-06-11 12:00:00"
"2017-06-14 12:00:00"
"2017-06-23 00:00:00"
"2017-06-24 00:00:00"
"2017-06-28 12:00:00"
"2017-06-30 06:00:00"
)

REF_CASE=$CASES/case_ts00

echo "[$(date +%H:%M:%S)] === 25-timestamp campaign on UGA ==="
echo "[$(date +%H:%M:%S)] Reference mesh: $REF_CASE"

# -----------------------------------------------------------------------
# Step 1: Prepare all cases (sequential — fast, ~2s each)
# -----------------------------------------------------------------------
for i in "${!TIMESTAMPS[@]}"; do
    CASE_ID=$(printf "ts%02d" $i)
    TS="${TIMESTAMPS[$i]}"
    CASE_DIR=$CASES/case_$CASE_ID

    # Skip if already solved
    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
        echo "[$(date +%H:%M:%S)] $CASE_ID: already solved, skipping"
        continue
    fi

    echo "[$(date +%H:%M:%S)] Preparing $CASE_ID ($TS)..."

    # Create case dir
    mkdir -p "$CASE_DIR"

    # Copy polyMesh from reference (if not present)
    if [ ! -f "$CASE_DIR/constant/polyMesh/points" ]; then
        mkdir -p "$CASE_DIR/constant"
        cp -r "$REF_CASE/constant/polyMesh" "$CASE_DIR/constant/"
    fi

    # Copy templates + system from reference (then overwrite with per-case inflow)
    if [ ! -f "$CASE_DIR/system/controlDict" ]; then
        cp -r "$REF_CASE/system" "$CASE_DIR/"
    fi

    # Copy helper scripts
    cp "$SCRIPTS/init_from_era5.py" "$CASE_DIR/" 2>/dev/null || true
    cp "$SCRIPTS/reconstruct_fields.py" "$CASE_DIR/" 2>/dev/null || true

    # Prepare inflow for this timestamp
    if [ ! -f "$CASE_DIR/inflow.json" ]; then
        $PYTHON "$SCRIPTS/prepare_inflow.py" \
            --era5-zarr "$ERA5" \
            --timestamp "$TS" \
            --site-lat $SITE_LAT --site-lon $SITE_LON \
            --output "$CASE_DIR/inflow.json" 2>/dev/null
    fi

    # Render templates (0/U, 0/k, etc.) with per-case inflow
    if [ ! -f "$CASE_DIR/0/U" ] || ! grep -q "nonuniform" "$CASE_DIR/0/U" 2>/dev/null; then
        # Render templates using generate_mesh (only templates, mesh already exists)
        $PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPTS')
from generate_mesh import generate_mesh
import yaml, json

with open('$SCRIPTS/../configs/sites/perdigao.yaml') as f:
    site_cfg = yaml.safe_load(f)

generate_mesh(
    site_cfg=site_cfg, resolution_m=1000, context_cells=1,
    output_dir='$CASE_DIR', srtm_tif=None,
    inflow_json='$CASE_DIR/inflow.json',
    domain_km=14, domain_type='cylinder',
    solver_name='simpleFoam', thermal=False,
    coriolis=True, transport_T=True,
    n_iter=$N_ITER, write_interval=$N_ITER,
    lateral_patches=['section_0','section_1','section_2','section_3',
                     'section_4','section_5','section_6','section_7'],
)
" 2>/dev/null

        # writeCellCentres (copy from reference if available)
        if [ -f "$REF_CASE/0/Cx" ] && [ ! -f "$CASE_DIR/0/Cx" ]; then
            cp "$REF_CASE/0/Cx" "$REF_CASE/0/Cy" "$REF_CASE/0/Cz" "$CASE_DIR/0/"
        else
            docker run --rm -v "$CASE_DIR":/case -w /case $OF_IMAGE \
                bash -c "postProcess -func writeCellCentres -time 0 > /dev/null 2>&1"
        fi

        # init_from_era5
        $PYTHON "$CASE_DIR/init_from_era5.py" \
            --case-dir "$CASE_DIR" --inflow "$CASE_DIR/inflow.json" 2>/dev/null
    fi

    echo "[$(date +%H:%M:%S)] $CASE_ID ready"
done

# -----------------------------------------------------------------------
# Step 2: Solve all cases (2 at a time, 24 cores each)
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Solving (2 cases × 24 cores) ==="

solve_case() {
    local CASE_ID=$1
    local CASE_DIR=$CASES/case_$CASE_ID

    # Skip if already solved
    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
        echo "[$(date +%H:%M:%S)] $CASE_ID: already solved"
        return 0
    fi

    docker run --rm --cpus=$NPROCS --memory=16g \
        -v "$CASE_DIR":/case -w /case $OF_IMAGE bash -c \
        "foamDictionary system/decomposeParDict -entry numberOfSubdomains -set $NPROCS && \
         rm -rf processor* && \
         decomposePar -force > /dev/null 2>&1 && \
         for d in processor*/; do ln -sf ../../constant/boundaryData \${d}constant/boundaryData; done && \
         mpirun --allow-run-as-root -np $NPROCS simpleFoam -parallel > /case/log.simpleFoam 2>&1"

    # Reconstruct
    cd "$CASE_DIR"
    $PYTHON reconstruct_fields.py --case-dir . --time $N_ITER --write-foam 2>/dev/null
    cd "$SCRIPTS"

    local CLOCK=$(grep ClockTime "$CASE_DIR/log.simpleFoam" 2>/dev/null | tail -1)
    echo "[$(date +%H:%M:%S)] $CASE_ID done: $CLOCK"
}

# Run 2 at a time
PIDS=()
for i in "${!TIMESTAMPS[@]}"; do
    CASE_ID=$(printf "ts%02d" $i)

    # Skip if already solved
    if [ -f "$CASES/case_$CASE_ID/$N_ITER/U" ]; then
        continue
    fi

    solve_case "$CASE_ID" &
    PIDS+=($!)

    # When 2 are running, wait for both
    if [ ${#PIDS[@]} -ge 2 ]; then
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        PIDS=()
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "[$(date +%H:%M:%S)] === ALL 25 DONE ==="
echo "[$(date +%H:%M:%S)] Results in $CASES/case_ts*/500/"
