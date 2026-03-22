#!/bin/bash
# run_batch_server.sh — Run multiple OF cases in parallel on UGA server
#
# Usage:
#   ./run_batch_server.sh /home/guillaume/dsw/cases [nprocs_per_case]
#
# Runs all case_* directories in parallel, each with nprocs_per_case cores.
# Default: 24 cores per case (fits 4 cases on 96 cores).

set -euo pipefail

CASES_DIR="${1:?Usage: $0 <cases_dir> [nprocs_per_case]}"
NPROCS="${2:-24}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Batch run: $(ls -d $CASES_DIR/case_* | wc -l) cases, $NPROCS cores each ==="

PIDS=()
for case_dir in "$CASES_DIR"/case_*; do
    [ -d "$case_dir" ] || continue
    case_name=$(basename "$case_dir")
    echo "[$(date +%H:%M:%S)] Starting $case_name..."
    "$SCRIPT_DIR/run_on_server.sh" "$case_dir" "$NPROCS" > "$case_dir/run.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} cases to complete..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo "=== Batch complete: $((${#PIDS[@]} - FAILED))/${#PIDS[@]} succeeded ==="

# Summary
for case_dir in "$CASES_DIR"/case_*; do
    [ -f "$case_dir/log.simpleFoam" ] || continue
    case_name=$(basename "$case_dir")
    time=$(grep "ClockTime" "$case_dir/log.simpleFoam" | tail -1 | awk -F'= ' '{print $NF}' | awk '{print $1}')
    resid=$(grep "Ux" "$case_dir/log.simpleFoam" | tail -1 | awk -F'Initial residual = ' '{print $2}' | awk -F, '{print $1}')
    cells=$(grep "cells:" "$case_dir/log.cartesianMesh" 2>/dev/null | awk '{print $2}')
    echo "  $case_name: ${cells:-?} cells, ${time:-?}s, Ux=$resid"
done
