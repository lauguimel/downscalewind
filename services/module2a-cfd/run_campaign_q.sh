#!/usr/bin/env bash
# run_campaign_q.sh — 100-case smoke campaign with T+q transport (100% on UGA)
#
# All cases share the same TBM mesh (Perdigão, ~165k cells).
# Solves 4 in parallel (4 × 24 = 96 cores).
#
# Usage:
#   ssh UGA
#   cd /home/guillaume/dsw
#   nohup bash scripts/run_campaign_q.sh > log_campaign_q.txt 2>&1 &
#
# Prereqs on UGA:
#   /home/guillaume/dsw/era5_perdigao.zarr
#   /home/guillaume/dsw/worldcover_perdigao.tif
#   /home/guillaume/dsw/scripts/{prepare_inflow.py, init_from_era5.py, generate_mesh.py, ...}
#   /home/guillaume/dsw/scripts/templates/openfoam/    (Jinja2 templates with q.j2)
#   /home/guillaume/dsw/configs/sites/perdigao.yaml
#   /home/guillaume/dsw/timestamps_100.csv

set -euo pipefail

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
PYTHON=/home/guillaume/miniconda3/bin/python
SCRIPTS=/home/guillaume/dsw/scripts
ERA5=/home/guillaume/dsw/era5_perdigao.zarr
WORLDCOVER=/home/guillaume/dsw/worldcover_perdigao.tif
SITE_CFG=/home/guillaume/dsw/configs/sites/perdigao.yaml
OF_IMAGE=microfluidica/openfoam:latest
NPROCS=24
N_ITER=500
SITE_LAT=39.716
SITE_LON=-7.740
MAX_PARALLEL=4    # 4 × 24 = 96 cores

CASES_DIR=/home/guillaume/dsw/cases/poc_100ts_q
TIMESTAMPS_CSV=/home/guillaume/dsw/timestamps_100.csv

# -----------------------------------------------------------------------
# Read timestamps from CSV
# -----------------------------------------------------------------------
declare -a ALL_TS=()
while IFS=, read -r ts rest; do
    [[ "$ts" == "timestamp" ]] && continue  # skip header
    ALL_TS+=("$ts")
done < "$TIMESTAMPS_CSV"

N_CASES=${#ALL_TS[@]}
echo "[$(date +%H:%M:%S)] === Campaign: $N_CASES cases, ${MAX_PARALLEL}×${NPROCS} cores ==="
mkdir -p "$CASES_DIR"

# -----------------------------------------------------------------------
# Step 0: Reference mesh (reuse from previous 25ts campaign)
# -----------------------------------------------------------------------
# The TBM mesh is site-specific (Perdigão terrain), already generated.
# We reuse it for all 100 cases — only inflow/init changes between cases.
REF_MESH=/home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/constant/polyMesh
REF_CC=/home/guillaume/dsw/cases/poc_tbm_25ts/case_ts00/0  # Cx, Cy, Cz

if [ ! -f "$REF_MESH/points" ]; then
    echo "ERROR: Reference mesh not found at $REF_MESH"
    echo "Run the 25ts campaign first, or point REF_MESH to an existing TBM mesh."
    exit 1
fi
echo "[$(date +%H:%M:%S)] Reference mesh: $REF_MESH (OK)"

# -----------------------------------------------------------------------
# Step 1: Prepare all cases (parallel Python, fast)
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Step 1: Prepare $N_CASES cases ==="

prepare_case() {
    local IDX=$1
    local TS="${ALL_TS[$IDX]}"
    local CID=$(printf "ts%03d" $IDX)
    local CASE_DIR=$CASES_DIR/case_$CID

    # Skip if already solved
    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
        return 0
    fi

    mkdir -p "$CASE_DIR/constant" "$CASE_DIR/0"

    # Copy shared mesh
    if [ ! -f "$CASE_DIR/constant/polyMesh/points" ]; then
        cp -r "$REF_MESH" "$CASE_DIR/constant/"
    fi

    # Copy cell centres
    if [ ! -f "$CASE_DIR/0/Cx" ]; then
        cp "$REF_CC/Cx" "$REF_CC/Cy" "$REF_CC/Cz" "$CASE_DIR/0/"
    fi

    # Copy helper scripts
    cp "$SCRIPTS/init_from_era5.py" "$SCRIPTS/reconstruct_fields.py" "$CASE_DIR/" 2>/dev/null || true

    # Prepare inflow (with q)
    if [ ! -f "$CASE_DIR/inflow.json" ]; then
        $PYTHON "$SCRIPTS/prepare_inflow.py" \
            --era5 "$ERA5" --case "$TS" \
            --lat $SITE_LAT --lon $SITE_LON \
            --output "$CASE_DIR/inflow.json" 2>/dev/null
    fi

    # z0 WorldCover
    if [ ! -f "$CASE_DIR/constant/boundaryData/terrain/0/z0" ]; then
        $PYTHON "$SCRIPTS/generate_z0_field.py" \
            --case-dir "$CASE_DIR" --worldcover "$WORLDCOVER" \
            --site-lat $SITE_LAT --site-lon $SITE_LON 2>/dev/null || true
    fi

    # Render templates (with transport_q=True)
    if [ ! -f "$CASE_DIR/system/controlDict" ]; then
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
    coriolis=True, transport_T=True, transport_q=True,
    canopy_enabled=False,
    n_iter=$N_ITER, write_interval=$N_ITER,
    lateral_patches=['section_0','section_1','section_2','section_3',
                     'section_4','section_5','section_6','section_7'],
    z0_mapped=True,
)
" 2>/dev/null
    fi

    # Ensure decomposeParDict
    local DP=$CASE_DIR/system/decomposeParDict
    if [ ! -f "$DP" ]; then
        cat > "$DP" << EOFD
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains $NPROCS;
method scotch;
EOFD
    fi

    # Init from ERA5 (cell + boundary data for U, k, epsilon, T, q)
    if [ ! -f "$CASE_DIR/0/U" ] || ! grep -q "nonuniform" "$CASE_DIR/0/U" 2>/dev/null; then
        $PYTHON "$CASE_DIR/init_from_era5.py" \
            --case-dir "$CASE_DIR" --inflow "$CASE_DIR/inflow.json" 2>/dev/null
    fi
}

# Prepare in batches of 8 (parallel Python)
PIDS=()
N_PREPARED=0
for i in $(seq 0 $((N_CASES - 1))); do
    CID=$(printf "ts%03d" $i)

    # Skip already solved
    if [ -f "$CASES_DIR/case_$CID/$N_ITER/U" ]; then
        N_PREPARED=$((N_PREPARED + 1))
        continue
    fi

    prepare_case $i &
    PIDS+=($!)
    N_PREPARED=$((N_PREPARED + 1))

    # Limit parallel Python processes (avoid memory issues)
    if [ ${#PIDS[@]} -ge 8 ]; then
        for pid in "${PIDS[@]}"; do wait $pid || true; done
        PIDS=()
        echo "[$(date +%H:%M:%S)] Prepared $N_PREPARED / $N_CASES"
    fi
done
for pid in "${PIDS[@]}"; do wait $pid || true; done
echo "[$(date +%H:%M:%S)] All $N_CASES cases prepared"

# -----------------------------------------------------------------------
# Step 2: Solve (MAX_PARALLEL at a time, NPROCS cores each)
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Step 2: Solve ($N_CASES cases, ${MAX_PARALLEL}×${NPROCS} cores) ==="

solve_case() {
    local IDX=$1
    local CID=$(printf "ts%03d" $IDX)
    local CASE_DIR=$CASES_DIR/case_$CID

    if [ -f "$CASE_DIR/$N_ITER/U" ]; then
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

    # Clean processor dirs
    rm -rf "$CASE_DIR"/processor*

    local T1=$(date +%s)
    local DT=$((T1 - T0))
    local CLOCK=$(grep ClockTime "$CASE_DIR/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "[$(date +%H:%M:%S)] $CID done: solve=${CLOCK:-?}s, total=${DT}s"
}

# Solve MAX_PARALLEL at a time
PIDS=()
N_SOLVED=0
for i in $(seq 0 $((N_CASES - 1))); do
    CID=$(printf "ts%03d" $i)

    if [ -f "$CASES_DIR/case_$CID/$N_ITER/U" ]; then
        N_SOLVED=$((N_SOLVED + 1))
        continue
    fi

    solve_case $i &
    PIDS+=($!)

    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
        for pid in "${PIDS[@]}"; do wait $pid || true; done
        PIDS=()
        N_SOLVED=$((N_SOLVED + ${#PIDS[@]} + 0))
    fi
done
for pid in "${PIDS[@]}"; do wait $pid || true; done

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === DONE ==="
N_OK=0
N_FAIL=0
TOTAL_CLOCK=0
for i in $(seq 0 $((N_CASES - 1))); do
    CID=$(printf "ts%03d" $i)
    if [ -f "$CASES_DIR/case_$CID/$N_ITER/U" ]; then
        N_OK=$((N_OK + 1))
        CLOCK=$(grep ClockTime "$CASES_DIR/case_$CID/log.simpleFoam" 2>/dev/null | tail -1 | awk '{print $NF}')
        TOTAL_CLOCK=$((TOTAL_CLOCK + ${CLOCK:-0}))
    else
        N_FAIL=$((N_FAIL + 1))
        echo "  FAILED: $CID"
    fi
done

echo ""
echo "  Cases:     $N_OK / $N_CASES OK ($N_FAIL failed)"
echo "  Solve time: ${TOTAL_CLOCK}s total, $((TOTAL_CLOCK / N_OK))s avg"
echo "  Fields:    U k epsilon T q nut p"
echo "  Results:   $CASES_DIR"

# Check q is present in a solved case
SAMPLE=$(ls -d "$CASES_DIR"/case_ts*/$N_ITER/q 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
    echo "  q field:   PRESENT ✓"
else
    echo "  q field:   MISSING ✗"
fi
