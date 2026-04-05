#!/bin/bash
# Submit a sweep of transport surrogate training jobs on Aqua HPC.
# Uses the train_transport_surrogate.pbs script with -v variable overrides.
#
# Usage (from ~/dsw on Aqua):
#   bash configs/hpc/sweep_transport_surrogate.sh

PBS_SCRIPT="$HOME/dsw/configs/hpc/train_transport_surrogate.pbs"

submit() {
    local NAME="$1"
    local MODEL="$2"
    local BS="$3"
    local LR="$4"
    local EXTRA="$5"

    echo "Submitting: $NAME ($MODEL, bs=$BS, lr=$LR) $EXTRA"
    qsub -N "train_${NAME}" \
         -v "MODEL=$MODEL,RUN_NAME=$NAME,BATCH_SIZE=$BS,LR=$LR,EXTRA_ARGS=$EXTRA" \
         "$PBS_SCRIPT"
}

# ── UNet3D variants ──────────────────────────────────────────────────────────

submit "unet_f16"  unet 4 1e-3 "--base-features 16"
submit "unet_f32"  unet 2 1e-3 "--base-features 32"
submit "unet_f48"  unet 1 5e-4 "--base-features 48"

# ── FNO3D variants ───────────────────────────────────────────────────────────

submit "fno_w16_l3"  fno 4 1e-3 "--width 16 --n-layers 3 --modes 4 8 8"
submit "fno_w32_l4"  fno 2 1e-3 "--width 32 --n-layers 4 --modes 8 16 16"
submit "fno_w48_l6"  fno 1 5e-4 "--width 48 --n-layers 6 --modes 12 24 24"

# ── PINO variants (physics-informed) ────────────────────────────────────────

submit "unet_f32_pino"    unet 2 1e-3 "--base-features 32 --lambda-pde 0.01 --lambda-bc 0.1"
submit "fno_w32_l4_pino"  fno  2 1e-3 "--width 32 --n-layers 4 --modes 8 16 16 --lambda-pde 0.01 --lambda-bc 0.1"

echo ""
echo "=== 8 jobs submitted ==="
echo "Monitor with: qstat -u maitreje"
echo "Results in: /scratch/maitreje/fuxicfd/models/"
