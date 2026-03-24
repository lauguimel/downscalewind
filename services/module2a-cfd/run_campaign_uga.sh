#!/usr/bin/env bash
# run_campaign_uga.sh — Run CFD campaign 100% on UGA (no SCP per case)
#
# Usage:
#   bash run_campaign_uga.sh z0_sensitivity   # 10 cases: 5×uniform + 5×WorldCover
#   bash run_campaign_uga.sh 25ts             # 25 timestamps, z0 uniform
#   bash run_campaign_uga.sh 25ts_wc          # 25 timestamps, z0 WorldCover
#
# Prereqs on UGA:
#   /home/guillaume/dsw/era5_perdigao.zarr
#   /home/guillaume/dsw/worldcover_perdigao.tif  (for WorldCover runs)
#   /home/guillaume/dsw/scripts/{prepare_inflow.py, init_from_era5.py, ...}
#   /home/guillaume/dsw/scripts/templates/       (Jinja2 OF templates)
#   /home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/constant/polyMesh/  (reference mesh)

set -euo pipefail

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
PYTHON=/home/guillaume/miniconda3/bin/python
SCRIPTS=/home/guillaume/dsw/scripts
ERA5=/home/guillaume/dsw/era5_perdigao.zarr
WORLDCOVER=/home/guillaume/dsw/worldcover_perdigao.tif
CANOPY_HEIGHT=/home/guillaume/dsw/canopy_height_perdigao.tif
OF_IMAGE=microfluidica/openfoam:latest
NPROCS=24
N_ITER=500
SITE_LAT=39.716
SITE_LON=-7.740
SITE_CFG=/home/guillaume/dsw/configs/sites/perdigao.yaml

# Reference mesh (from first successful TBM run)
REF_MESH=/home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/constant/polyMesh
REF_CC=/home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/0  # Cx, Cy, Cz

# All 25 timestamps (from k-means clustering)
ALL_TS=(
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

# 5 timestamps for sensitivity studies (diverse wind conditions)
SENS_INDICES=(0 3 14 17 23)

# -----------------------------------------------------------------------
# Parse mode
# -----------------------------------------------------------------------
MODE=${1:-z0_sensitivity}
echo "[$(date +%H:%M:%S)] === Campaign mode: $MODE ==="

declare -a CASE_IDS=()
declare -a CASE_TS=()
declare -a CASE_Z0=()     # "uniform" or "worldcover"
declare -a CASE_CANOPY=() # "true" or "false"

case "$MODE" in
    z0_sensitivity)
        CASES_DIR=/home/guillaume/dsw/cases/poc_tbm_z0_sensitivity
        for i in "${!SENS_INDICES[@]}"; do
            idx=${SENS_INDICES[$i]}
            CASE_IDS+=("u$(printf '%02d' $i)")
            CASE_TS+=("${ALL_TS[$idx]}")
            CASE_Z0+=("uniform")
            CASE_CANOPY+=("false")
        done
        for i in "${!SENS_INDICES[@]}"; do
            idx=${SENS_INDICES[$i]}
            CASE_IDS+=("w$(printf '%02d' $i)")
            CASE_TS+=("${ALL_TS[$idx]}")
            CASE_Z0+=("worldcover")
            CASE_CANOPY+=("false")
        done
        ;;
    canopy)
        CASES_DIR=/home/guillaume/dsw/cases/poc_tbm_canopy
        Z0_CANOPY=0.01  # bare ground under canopy — NOT WorldCover
        for i in "${!SENS_INDICES[@]}"; do
            idx=${SENS_INDICES[$i]}
            CASE_IDS+=("c$(printf '%02d' $i)")
            CASE_TS+=("${ALL_TS[$idx]}")
            CASE_Z0+=("uniform")
            CASE_CANOPY+=("true")
        done
        ;;
    25ts)
        CASES_DIR=/home/guillaume/dsw/cases/poc_tbm_25ts
        for i in "${!ALL_TS[@]}"; do
            CASE_IDS+=("ts$(printf '%02d' $i)")
            CASE_TS+=("${ALL_TS[$i]}")
            CASE_Z0+=("uniform")
            CASE_CANOPY+=("false")
        done
        ;;
    25ts_wc)
        CASES_DIR=/home/guillaume/dsw/cases/poc_tbm_25ts_wc
        for i in "${!ALL_TS[@]}"; do
            CASE_IDS+=("ts$(printf '%02d' $i)")
            CASE_TS+=("${ALL_TS[$i]}")
            CASE_Z0+=("worldcover")
            CASE_CANOPY+=("false")
        done
        ;;
    *)
        echo "Usage: $0 {z0_sensitivity|25ts|25ts_wc|canopy}"
        exit 1
        ;;
esac

N_CASES=${#CASE_IDS[@]}
echo "[$(date +%H:%M:%S)] $N_CASES cases in $CASES_DIR"
mkdir -p "$CASES_DIR"

# -----------------------------------------------------------------------
# Step 1: Prepare all cases (sequential, ~2s each)
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Step 1: Prepare cases ==="

for i in "${!CASE_IDS[@]}"; do
    CID=${CASE_IDS[$i]}
    TS="${CASE_TS[$i]}"
    Z0_MODE=${CASE_Z0[$i]}
    CASE_DIR=$CASES_DIR/case_$CID

    # Skip if already solved
    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
        echo "[$(date +%H:%M:%S)] $CID: already solved, skip"
        continue
    fi

    CANOPY_MODE=${CASE_CANOPY[$i]:-false}
    echo -n "[$(date +%H:%M:%S)] $CID ($TS, z0=$Z0_MODE, canopy=$CANOPY_MODE)... "
    mkdir -p "$CASE_DIR/constant" "$CASE_DIR/0"

    # Copy polyMesh from reference
    if [ ! -f "$CASE_DIR/constant/polyMesh/points" ]; then
        cp -r "$REF_MESH" "$CASE_DIR/constant/"
    fi

    # Copy cell centres
    if [ ! -f "$CASE_DIR/0/Cx" ]; then
        cp "$REF_CC/Cx" "$REF_CC/Cy" "$REF_CC/Cz" "$CASE_DIR/0/"
    fi

    # Copy helper scripts
    cp "$SCRIPTS/init_from_era5.py" "$SCRIPTS/reconstruct_fields.py" "$CASE_DIR/" 2>/dev/null || true

    # Prepare inflow
    if [ ! -f "$CASE_DIR/inflow.json" ]; then
        $PYTHON "$SCRIPTS/prepare_inflow.py" \
            --era5 "$ERA5" --case "$TS" \
            --lat $SITE_LAT --lon $SITE_LON \
            --output "$CASE_DIR/inflow.json" 2>/dev/null
    fi

    # Generate z0 WorldCover boundaryData (if needed)
    if [ "$Z0_MODE" = "worldcover" ] && [ ! -f "$CASE_DIR/constant/boundaryData/terrain/0/z0" ]; then
        $PYTHON "$SCRIPTS/generate_z0_field.py" \
            --case-dir "$CASE_DIR" --worldcover "$WORLDCOVER" \
            --site-lat $SITE_LAT --site-lon $SITE_LON 2>/dev/null
    fi

    # Render templates
    Z0_FLAG="False"
    [ "$Z0_MODE" = "worldcover" ] && Z0_FLAG="True"
    CANOPY_FLAG="False"
    [ "$CANOPY_MODE" = "true" ] && CANOPY_FLAG="True"

    if [ ! -f "$CASE_DIR/system/controlDict" ]; then
        # Override z0 in inflow.json for canopy cases (bare ground = 0.01m)
        if [ "$CANOPY_MODE" = "true" ] && [ -n "${Z0_CANOPY:-}" ]; then
            $PYTHON -c "
import json
with open('$CASE_DIR/inflow.json') as f: d = json.load(f)
d['z0_eff'] = $Z0_CANOPY
with open('$CASE_DIR/inflow.json', 'w') as f: json.dump(d, f, indent=2)
"
        fi

        $PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPTS')
from generate_mesh import generate_mesh
import yaml
with open('$SITE_CFG') as f:
    site_cfg = yaml.safe_load(f)
generate_mesh(
    site_cfg=site_cfg, resolution_m=1000, context_cells=1,
    output_dir='$CASE_DIR', srtm_tif=None,
    inflow_json='$CASE_DIR/inflow.json',
    domain_km=14, domain_type='cylinder',
    solver_name='simpleFoam', thermal=False,
    coriolis=True, transport_T=True,
    canopy_enabled=$CANOPY_FLAG,
    n_iter=$N_ITER, write_interval=$N_ITER,
    lateral_patches=['section_0','section_1','section_2','section_3',
                     'section_4','section_5','section_6','section_7'],
    z0_mapped=$Z0_FLAG,
)
" 2>/dev/null
    fi

    # Generate LAD + Cd fields from ETH Canopy Height (after templates, before init)
    if [ "$CANOPY_MODE" = "true" ] && [ ! -f "$CASE_DIR/0/LAD" ]; then
        $PYTHON "$SCRIPTS/generate_lad_field.py" \
            --case-dir "$CASE_DIR" --landcover "$CANOPY_HEIGHT" \
            --site-lat $SITE_LAT --site-lon $SITE_LON 2>&1 | tail -3
    fi

    # Ensure decomposeParDict
    DP=$CASE_DIR/system/decomposeParDict
    if [ ! -f "$DP" ]; then
        echo 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}' > "$DP"
        echo "numberOfSubdomains $NPROCS; method scotch;" >> "$DP"
    fi

    # init_from_era5 (cell + boundary data)
    if [ ! -f "$CASE_DIR/0/U" ] || ! grep -q "nonuniform" "$CASE_DIR/0/U" 2>/dev/null; then
        $PYTHON "$CASE_DIR/init_from_era5.py" \
            --case-dir "$CASE_DIR" --inflow "$CASE_DIR/inflow.json" 2>/dev/null
    fi

    echo "ready"
done

# -----------------------------------------------------------------------
# Step 2: Solve (2 at a time, 24 cores each)
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Step 2: Solve ($N_CASES cases, 2×${NPROCS} cores) ==="

solve_case() {
    local CID=$1
    local CASE_DIR=$CASES_DIR/case_$CID

    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
        echo "[$(date +%H:%M:%S)] $CID: already solved"
        return 0
    fi

    local T0=$(date +%s)

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

    local T1=$(date +%s)
    local DT=$((T1 - T0))
    local CLOCK=$(grep ClockTime "$CASE_DIR/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "[$(date +%H:%M:%S)] $CID done: solve=${CLOCK}s, total=${DT}s"
}

# Run 2 at a time
PIDS=()
for i in "${!CASE_IDS[@]}"; do
    CID=${CASE_IDS[$i]}

    if [ -f "$CASES_DIR/case_$CID/$N_ITER/U" ]; then
        continue
    fi

    solve_case "$CID" &
    PIDS+=($!)

    if [ ${#PIDS[@]} -ge 2 ]; then
        for pid in "${PIDS[@]}"; do
            wait $pid || true
        done
        PIDS=()
    fi
done

for pid in "${PIDS[@]}"; do
    wait $pid || true
done

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === DONE: $MODE ($N_CASES cases) ==="
echo ""
echo "Results:"
for CID in "${CASE_IDS[@]}"; do
    if [ -f "$CASES_DIR/case_$CID/$N_ITER/U" ]; then
        CLOCK=$(grep ClockTime "$CASES_DIR/case_$CID/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
        echo "  $CID: OK (${CLOCK}s)"
    else
        echo "  $CID: FAILED"
    fi
done
