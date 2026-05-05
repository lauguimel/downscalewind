#!/bin/bash
set -euo pipefail

ROOT="$HOME/dsw/data/models"
OUT_ROOT="$ROOT/eval_v2_physical"
SPLIT="${1:-test}"
CROP_KM="${2:-}"
QSUB_EXTRA=""

if [ -n "$CROP_KM" ]; then
  OUT_ROOT="$ROOT/eval_v2_physical_crop_${CROP_KM}km"
  QSUB_EXTRA=",CROP_KM=$CROP_KM"
fi

qsub -v MODEL_TYPE=fno,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_fno3d_resid_s4/best.pt",OUT="$OUT_ROOT/surrogate_v2_fno3d_resid_s4/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
qsub -v MODEL_TYPE=fno,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_fno3d_resid_s4_agl/best.pt",OUT="$OUT_ROOT/surrogate_v2_fno3d_resid_s4_agl/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
qsub -v MODEL_TYPE=vit,PRESET=base,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_vit_base_resid_s4_geo/best.pt",OUT="$OUT_ROOT/surrogate_v2_vit_base_resid_s4_geo/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
qsub -v MODEL_TYPE=vit,PRESET=base,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_vit_base_resid_s4_geo_agl/best.pt",OUT="$OUT_ROOT/surrogate_v2_vit_base_resid_s4_geo_agl/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
qsub -v MODEL_TYPE=vit,PRESET=large,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_vit_large_resid_s4_geo/best.pt",OUT="$OUT_ROOT/surrogate_v2_vit_large_resid_s4_geo/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
qsub -v MODEL_TYPE=vit,PRESET=large,SPLIT="$SPLIT"$QSUB_EXTRA,WEIGHTS="$ROOT/surrogate_v2_vit_large_resid_s4_geo_agl/best.pt",OUT="$OUT_ROOT/surrogate_v2_vit_large_resid_s4_geo_agl/$SPLIT" configs/hpc/evaluate_v2_physical.pbs
