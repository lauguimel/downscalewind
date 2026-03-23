#!/bin/bash
# run_on_server.sh — Run OpenFOAM case on UGA server via Docker
#
# Usage:
#   ./run_on_server.sh <case_dir> [nprocs]
#
# Example:
#   ./run_on_server.sh /home/guillaume/dsw/cases/case_res100 48
#
# The script:
#   1. Runs cartesianMesh (if no polyMesh)
#   2. Writes cell centres
#   3. Runs simpleFoam (serial or parallel)
#   4. Logs to case_dir/log.*

set -euo pipefail

CASE_DIR="${1:?Usage: $0 <case_dir> [nprocs]}"
NPROCS="${2:-1}"
IMAGE="microfluidica/openfoam:latest"

echo "=== OpenFOAM run: $(basename $CASE_DIR), nprocs=$NPROCS ==="
echo "Case: $CASE_DIR"

# Check image
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "$IMAGE"; then
    echo "ERROR: Docker image $IMAGE not found. Load with:"
    echo "  docker load < openfoam_2512_cfmesh.tar.gz"
    exit 1
fi

run_of() {
    docker run --rm \
        -v "$CASE_DIR":/case -w /case \
        "$IMAGE" \
        bash -c "$1"
}

# Step 1: Mesh (skip if polyMesh exists)
if [ ! -f "$CASE_DIR/constant/polyMesh/points" ]; then
    echo "[$(date +%H:%M:%S)] Meshing..."
    run_of "cartesianMesh > /case/log.cartesianMesh 2>&1"
    echo "[$(date +%H:%M:%S)] Mesh done. $(grep 'cells:' $CASE_DIR/log.cartesianMesh 2>/dev/null || echo '')"
else
    echo "[$(date +%H:%M:%S)] Mesh exists, skipping."
fi

# Step 2: Write cell centres (for init_from_era5.py)
if [ ! -d "$CASE_DIR/0/cellCentre" ] && [ ! -f "$CASE_DIR/0/C" ]; then
    echo "[$(date +%H:%M:%S)] Writing cell centres..."
    run_of "postProcess -func writeCellCentres -time 0 > /case/log.writeCellCentres 2>&1"
fi

# Step 3: Init from ERA5 (needs python with numpy/scipy — run on host)
if ! grep -q "nonuniform" "$CASE_DIR/0/U" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Initializing fields from ERA5..."
    python3 "$CASE_DIR/init_from_era5.py" --case-dir "$CASE_DIR" --inflow "$CASE_DIR/inflow.json"
fi

# Step 4: Solve
echo "[$(date +%H:%M:%S)] Running simpleFoam (nprocs=$NPROCS)..."
if [ "$NPROCS" -gt 1 ]; then
    SOLVE_CMD="foamDictionary system/decomposeParDict -entry numberOfSubdomains -set $NPROCS && \
               decomposePar -force && \
               for d in processor*/; do ln -sf ../../constant/boundaryData \${d}constant/boundaryData; done && \
               mpirun --allow-run-as-root -np $NPROCS simpleFoam -parallel > /case/log.simpleFoam 2>&1 && \
               reconstructPar -latestTime >> /case/log.simpleFoam 2>&1"
else
    SOLVE_CMD="simpleFoam > /case/log.simpleFoam 2>&1"
fi
run_of "$SOLVE_CMD"

echo "[$(date +%H:%M:%S)] Done. $(grep 'ExecutionTime' $CASE_DIR/log.simpleFoam | tail -1)"
echo "Final Ux residual: $(grep 'Ux' $CASE_DIR/log.simpleFoam | tail -1 | awk -F'Initial residual = ' '{print $2}' | awk -F, '{print $1}')"
