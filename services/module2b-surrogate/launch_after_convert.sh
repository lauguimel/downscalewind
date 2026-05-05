#!/bin/bash
# launch_after_convert.sh — Wait for conversion, then assemble + train
# Run on UGA: nohup bash launch_after_convert.sh > ~/dsw/log_train.txt 2>&1 &
set -euo pipefail

PYTHON=~/miniconda3/bin/python
TRAINING_DIR=~/dsw/data/cfd-database/training_1500
MODELS_DIR=~/dsw/data/models/module2b_poc
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo " Waiting for conversion to finish..."
echo "========================================="

# Wait for the convert process to finish
while pgrep -f convert_stacked_to_training > /dev/null 2>&1; do
    sleep 30
    SITES_DONE=$(ls -d "$TRAINING_DIR"/site_* 2>/dev/null | head -1 | xargs -I{} sh -c 'ls -d ${1%/*}/site_* 2>/dev/null | wc -l' _ {})
    echo "$(date +%H:%M:%S) Conversion in progress... (~${SITES_DONE} sites)"
done

echo "$(date +%H:%M:%S) Conversion complete!"
echo ""

# Count exported cases
N_CASES=$(find "$TRAINING_DIR" -name "grid.zarr" -maxdepth 2 | wc -l)
echo "Exported cases: $N_CASES"

# Step 1: Assemble dataset
echo ""
echo "[1/2] Assembling dataset.yaml..."
$PYTHON "$SCRIPT_DIR/../module2a-cfd/assemble_training_zarr.py" \
    --input "$TRAINING_DIR" \
    --output "$TRAINING_DIR/dataset.yaml" \
    --train-frac 0.72 \
    --val-frac 0.12
echo "[1/2] Done."

# Step 2: Train U-Net (volume variant)
echo ""
echo "[2/2] Training U-Net 3D (volume) on A6000..."
mkdir -p "$MODELS_DIR"

$PYTHON "$SCRIPT_DIR/train.py" \
    --model unet \
    --data-dir "$TRAINING_DIR" \
    --dataset "$TRAINING_DIR/dataset.yaml" \
    --output "$MODELS_DIR" \
    --epochs 100 \
    --lr 1e-3 \
    --batch-size 8 \
    --variant volume

echo ""
echo "========================================="
echo " Training complete!"
echo " Model: $MODELS_DIR/unet_volume/"
echo "========================================="
