#!/bin/bash
# deploy_icos_uga.sh — Sync ICOS campaign data + code to UGA and launch
#
# Run this after ERA5 ingestion is complete:
#   bash deploy_icos_uga.sh          # sync + dry-run
#   bash deploy_icos_uga.sh --run    # sync + launch
#
set -euo pipefail

UGA="UGA"
REMOTE="$UGA:~/dsw"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
RUN_FLAG="${1:-}"

echo "=== Deploy ICOS CFD campaign to UGA ==="
echo "Local:  $LOCAL"
echo "Remote: $REMOTE"
echo

# ── 1. Check ERA5 Zarr completeness ────────────────────────────────────────
echo "[1/5] Checking ERA5 stores..."
MISSING=0
for site in ope ipr hpb jfj puy trn sac; do
    zarr="data/raw/era5_${site}.zarr"
    if [ -d "$LOCAL/$zarr" ] && [ -f "$LOCAL/$zarr/zarr.json" ]; then
        echo "  OK: $zarr"
    else
        echo "  MISSING: $zarr"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some ERA5 stores missing. Wait for ingestion to complete."
    exit 1
fi

# ── 2. Sync code ───────────────────────────────────────────────────────────
echo
echo "[2/5] Syncing code to UGA..."
rsync -avz --delete \
    --include='*.py' --include='*.yaml' --include='*.j2' --include='*.csv' \
    --include='*/' --exclude='__pycache__' --exclude='.cache' --exclude='*.pyc' \
    "$LOCAL/services/module2a-cfd/" "$REMOTE/services/module2a-cfd/"

rsync -avz "$LOCAL/shared/" "$REMOTE/shared/"
rsync -avz "$LOCAL/configs/sites/" "$REMOTE/configs/sites/"

# ── 3. Sync ERA5 + run_matrix ──────────────────────────────────────────────
echo
echo "[3/5] Syncing ERA5 data..."
for site in ope ipr hpb jfj puy trn sac; do
    echo "  ERA5 $site..."
    rsync -avz "$LOCAL/data/raw/era5_${site}.zarr/" "$REMOTE/data/raw/era5_${site}.zarr/"
done

echo
echo "[4/5] Syncing run_matrix..."
ssh $UGA "mkdir -p ~/dsw/data/campaign/icos_fwi_v1"
rsync -avz "$LOCAL/data/campaign/icos_fwi_v1/run_matrix.csv" \
    "$REMOTE/data/campaign/icos_fwi_v1/run_matrix.csv"

# ── 4. Verify on UGA ──────────────────────────────────────────────────────
echo
echo "[5/5] Remote verification..."
ssh $UGA "
echo 'SRTM:'; ls ~/dsw/data/srtm_europe.tif 2>/dev/null || echo '  MISSING!'
echo 'Docker:'; docker images 2>/dev/null | grep -E 'microfluidica|terrainblock' || echo '  MISSING!'
echo 'Run matrix:'; wc -l ~/dsw/data/campaign/icos_fwi_v1/run_matrix.csv 2>/dev/null || echo '  MISSING!'
echo 'ERA5:'; ls -d ~/dsw/data/raw/era5_ope.zarr ~/dsw/data/raw/era5_jfj.zarr 2>/dev/null || echo '  MISSING!'
echo 'Python:'; python3 -c 'import numpy,pandas,xarray,zarr,rasterio,jinja2; print(\"All imports OK\")' 2>&1
"

if [ "$RUN_FLAG" = "--run" ]; then
    echo
    echo "=== Launching campaign (nohup) ==="
    # Use nohup + screen so the campaign survives SSH disconnect
    # --era5-zarr points to ANY site's Zarr — process_site() auto-resolves
    # to era5_<site_id>.zarr in the same directory (per-site ERA5 convention)
    ssh $UGA "cd ~/dsw && nohup python3 services/module2a-cfd/run_multisite_campaign.py \
        --run-matrix data/campaign/icos_fwi_v1/run_matrix.csv \
        --srtm data/srtm_europe.tif \
        --era5-zarr data/raw/era5_ope.zarr \
        --output data/campaign/icos_fwi_v1/cases \
        --n-cores 24 \
        --n-parallel 4 \
        --n-iter 500 \
        > data/campaign/icos_fwi_v1/campaign.log 2>&1 &
    echo \$! > data/campaign/icos_fwi_v1/campaign.pid
    echo \"PID: \$(cat data/campaign/icos_fwi_v1/campaign.pid)\"
    echo 'Campaign launched. Monitor: ssh UGA tail -f ~/dsw/data/campaign/icos_fwi_v1/campaign.log'
    "
else
    echo
    echo "=== Dry-run ==="
    ssh $UGA "cd ~/dsw && python3 services/module2a-cfd/run_multisite_campaign.py \
        --run-matrix data/campaign/icos_fwi_v1/run_matrix.csv \
        --srtm data/srtm_europe.tif \
        --era5-zarr data/raw/era5_ope.zarr \
        --output data/campaign/icos_fwi_v1/cases \
        --dry-run"
    echo
    echo "Dry-run OK. Run with --run to launch the campaign."
fi
