#!/bin/bash
# run_ablation.sh — Run architecture ablation study on A6000
set -euo pipefail

PY=/home/guillaume/miniconda3/bin/python
TRAIN=/home/guillaume/dsw/services/module2b-surrogate/train.py
EVALSC=/home/guillaume/dsw/services/module2b-surrogate/evaluate.py
DATA=/home/guillaume/dsw/data/cfd-database/training_9k
DATASET=$DATA/dataset.yaml
OUT=/home/guillaume/dsw/data/models/module2b_9k
EVAL_OUT=/home/guillaume/dsw/data/validation/surrogate_9k

echo "========================================"
echo " Ablation study — 3 architectures"
echo "========================================"

# 1. U-Net base=64
echo ""
echo "=== [1/3] U-Net base=64 ==="
$PY -u $TRAIN --model unet --data-dir $DATA --dataset $DATASET --output $OUT \
    --epochs 100 --lr 1e-3 --batch-size 4 --num-workers 4 \
    --variant volume --base-features 64

$PY -u $EVALSC --weights $OUT/unet_volume/best_model.pt \
    --data-dir $DATA --dataset $DATASET --output $EVAL_OUT/unet64 --inner-pad 32

# 2. FNO 3D
echo ""
echo "=== [2/3] FNO 3D ==="
$PY -u $TRAIN --model fno --data-dir $DATA --dataset $DATASET --output $OUT \
    --epochs 100 --lr 1e-3 --batch-size 8 --num-workers 4 \
    --fno-width 32 --fno-modes 16 16 8 --fno-layers 4

$PY -u $EVALSC --weights $OUT/fno/best_model.pt \
    --data-dir $DATA --dataset $DATASET --output $EVAL_OUT/fno --inner-pad 32

# 3. Factored 2D+1D
echo ""
echo "=== [3/3] Factored 2D+1D ==="
$PY -u $TRAIN --model factored --data-dir $DATA --dataset $DATASET --output $OUT \
    --epochs 100 --lr 1e-3 --batch-size 8 --num-workers 4 \
    --base-features 32

$PY -u $EVALSC --weights $OUT/factored/best_model.pt \
    --data-dir $DATA --dataset $DATASET --output $EVAL_OUT/factored --inner-pad 32

echo ""
echo "========================================"
echo " Ablation complete!"
echo "========================================"
