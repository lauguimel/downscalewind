#!/bin/bash
# run_training_uga.sh — End-to-end: convert → assemble → train on UGA A6000
#
# Usage:
#   bash run_training_uga.sh [--skip-convert] [--skip-assemble] [--model unet]
#
set -euo pipefail

PYTHON=~/miniconda3/bin/python
CAMPAIGN_DIR=~/dsw/campaign_1500
TRAINING_DIR=~/dsw/data/cfd-database/training_1500
MODELS_DIR=~/dsw/data/models/module2b_poc
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKERS=24

# Parse arguments
SKIP_CONVERT=false
SKIP_ASSEMBLE=false
MODEL=unet
VARIANT=volume
EPOCHS=100
BATCH_SIZE=4
LR=1e-3

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --skip-assemble) SKIP_ASSEMBLE=true; shift ;;
        --model) MODEL=$2; shift 2 ;;
        --variant) VARIANT=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --lr) LR=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo " DownscaleWind — Surrogate Training"
echo "========================================="
echo "Campaign: $CAMPAIGN_DIR"
echo "Training: $TRAINING_DIR"
echo "Model:    $MODEL ($VARIANT)"
echo "Epochs:   $EPOCHS"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "========================================="

# Step 1: Convert stacked Zarr → per-case format
if [ "$SKIP_CONVERT" = false ]; then
    echo ""
    echo "[1/3] Converting stacked Zarr → per-case training format..."
    echo "      Workers: $WORKERS"
    time $PYTHON "$SCRIPT_DIR/convert_stacked_to_training.py" \
        --input "$CAMPAIGN_DIR" \
        --output "$TRAINING_DIR" \
        --workers "$WORKERS" \
        --half-extent 2000 \
        --grid-size 128 \
        --r-fine 1000 \
        --r-context 3500
    echo "[1/3] Done."
else
    echo "[1/3] Skipping conversion (--skip-convert)"
fi

# Step 2: Assemble dataset.yaml
if [ "$SKIP_ASSEMBLE" = false ]; then
    echo ""
    echo "[2/3] Assembling dataset index..."
    # assemble_training_zarr.py is in module2a-cfd
    ASSEMBLER="$SCRIPT_DIR/../module2a-cfd/assemble_training_zarr.py"
    time $PYTHON "$ASSEMBLER" \
        --input "$TRAINING_DIR" \
        --output "$TRAINING_DIR/dataset.yaml" \
        --train-frac 0.72 \
        --val-frac 0.12
    echo "[2/3] Done."
else
    echo "[2/3] Skipping assembly (--skip-assemble)"
fi

# Step 3: Train
echo ""
echo "[3/3] Training $MODEL ($VARIANT) on GPU..."
echo "      Dataset: $TRAINING_DIR/dataset.yaml"
echo "      Output:  $MODELS_DIR"
echo "      Epochs:  $EPOCHS, LR: $LR, Batch: $BATCH_SIZE"

mkdir -p "$MODELS_DIR"

TRAIN_ARGS=(
    --model "$MODEL"
    --data-dir "$TRAINING_DIR"
    --dataset "$TRAINING_DIR/dataset.yaml"
    --output "$MODELS_DIR"
    --epochs "$EPOCHS"
    --lr "$LR"
    --batch-size "$BATCH_SIZE"
)

if [ "$MODEL" = "unet" ]; then
    TRAIN_ARGS+=(--variant "$VARIANT")
fi

time $PYTHON "$SCRIPT_DIR/train.py" "${TRAIN_ARGS[@]}"

echo ""
echo "========================================="
echo " Training complete!"
echo " Model: $MODELS_DIR/$MODEL/"
echo "========================================="
