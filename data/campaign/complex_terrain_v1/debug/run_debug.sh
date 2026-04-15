#!/usr/bin/env bash
# run_debug.sh — Launch 3 debug cases for complex_terrain_v1 pipeline validation.
#
# Pre-requisites (all should be satisfied post-maintenance):
#   - conda env 'downscalewind' activated
#   - ERA5 data available:
#       data/raw/era5_perdigao.zarr
#       data/raw/era5_fr-pue.zarr
#       data/raw/era5_es-lju.zarr
#   - SRTM Europe available: data/raw/srtm_europe.tif
#   - Docker/Apptainer with OF v2512 image
#
# Usage:
#   bash data/campaign/complex_terrain_v1/debug/run_debug.sh
#
# Outputs:
#   data/campaign/complex_terrain_v1/debug/{perdigao_iop_nocturnal,...}/
#     ├── case_ts000/                      # OF case
#     ├── grid.zarr                         # ML training format
#     └── validation.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CAMPAIGN_DIR="$ROOT/data/campaign/complex_terrain_v1"
DEBUG_DIR="$CAMPAIGN_DIR/debug"
SITE_MANIFEST="$CAMPAIGN_DIR/manifests/sites.yaml"
CAMPAIGN_MANIFEST="$CAMPAIGN_DIR/manifests/campaign.yaml"

# Docker/Apptainer runtime — auto-detect
if command -v apptainer &>/dev/null && [[ -f "$HOME/dsw/containers/openfoam_v2512.sif" ]]; then
    RUNTIME="apptainer"
    OF_IMAGE="$HOME/dsw/containers/openfoam_v2512.sif"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
    OF_IMAGE="microfluidica/openfoam:latest"
else
    echo "ERROR: neither docker nor apptainer found" >&2
    exit 1
fi

echo "Runtime: $RUNTIME"
echo "OF image: $OF_IMAGE"

# ─── Helper: run one debug case ───────────────────────────────────────────
run_case() {
    local CASE_NAME="$1"
    local SITE_ID="$2"
    local LAT="$3"
    local LON="$4"
    local TIMESTAMP="$5"
    local ERA5="$6"

    local CASE_DIR="$DEBUG_DIR/$CASE_NAME"
    echo ""
    echo "═════════════════════════════════════════════════════════════"
    echo "  $CASE_NAME ($SITE_ID)"
    echo "  lat=$LAT lon=$LON ts=$TIMESTAMP"
    echo "  ERA5: $ERA5"
    echo "═════════════════════════════════════════════════════════════"

    mkdir -p "$CASE_DIR"

    # Build a minimal run_matrix.csv for one run
    local MATRIX="$CASE_DIR/run_matrix.csv"
    cat > "$MATRIX" <<EOF
run_id,site_id,timestamp,lat,lon,group,priority,status
run_000000,$SITE_ID,$TIMESTAMP,$LAT,$LON,debug,high,pending
EOF

    # Launch single-site campaign run
    python "$ROOT/services/module2a-cfd/run_multisite_campaign.py" \
        --run-matrix "$MATRIX" \
        --srtm "$ROOT/data/raw/srtm_europe.tif" \
        --era5-zarr "$ERA5" \
        --output "$CASE_DIR" \
        --n-iter 2000 \
        --n-cores 24 \
        --n-parallel 1 \
        --runtime "$RUNTIME" \
        --of-image "$OF_IMAGE" \
        --site-manifest "$SITE_MANIFEST" \
        --campaign-manifest "$CAMPAIGN_MANIFEST" \
        --grid-export on \
        --grid-export-device cuda \
        2>&1 | tee "$CASE_DIR/campaign.log"

    echo "  → $CASE_DIR finished"
}

# ─── Cases ────────────────────────────────────────────────────────────────

run_case "perdigao_iop_nocturnal"  "debug_perdigao"  39.7129  -7.7360  "2017-05-04T22:00:00"  "$ROOT/data/raw/era5_perdigao.zarr"

run_case "fr_pue_canicule"  "debug_fr_pue"  43.7414  3.5958  "2022-08-10T12:00:00"  "$ROOT/data/raw/era5_fr-pue.zarr"

run_case "es_lju_synoptic"  "debug_es_lju"  37.1283  -2.9531  "2022-02-25T12:00:00"  "$ROOT/data/raw/era5_es-lju.zarr"

# ─── Validation report ────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════════"
echo "  Validation"
echo "═════════════════════════════════════════════════════════════"
python "$ROOT/services/module2a-cfd/validate_debug_cases.py" \
    --debug-dir "$DEBUG_DIR" \
    --spec "$DEBUG_DIR/debug_cases.yaml" \
    --output "$DEBUG_DIR/validation_report.md"

echo ""
echo "Report: $DEBUG_DIR/validation_report.md"
