#!/bin/bash
# Submit a sweep of transport surrogate training jobs on Aqua HPC.
# Each variant gets its own PBS job with a unique run name.
#
# Usage (from ~/dsw on Aqua):
#   bash configs/hpc/sweep_transport_surrogate.sh

DATA_DIR="/scratch/maitreje/fuxicfd/extracted"
OUTPUT_DIR="/scratch/maitreje/fuxicfd/models"
SCRIPT_DIR="$HOME/dsw/services/module2b-surrogate"
EPOCHS=100

submit_job() {
    local RUN_NAME="$1"
    local MODEL="$2"
    shift 2
    local EXTRA_ARGS="$@"

    echo "Submitting: $RUN_NAME ($MODEL) $EXTRA_ARGS"

    qsub -N "train_${RUN_NAME}" \
         -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=H100 \
         -l walltime=24:00:00 \
         -j oe \
         -m ae \
         -- /bin/bash -l -c "
cd \$PBS_O_WORKDIR
nvidia-smi || exit 1
module load Miniconda3 || module load Anaconda3
eval \"\\\$(conda shell.bash hook)\"
conda activate fuxicfd
python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')\"

cd $SCRIPT_DIR
python train_transport_surrogate.py \\
    --data-dir $DATA_DIR \\
    --output $OUTPUT_DIR \\
    --model $MODEL \\
    --run-name $RUN_NAME \\
    --epochs $EPOCHS \\
    --num-workers 4 \\
    $EXTRA_ARGS
"
}

# ── UNet3D variants ──────────────────────────────────────────────────────────

# Small U-Net (fewer parameters, faster, baseline)
submit_job "unet_f16"  unet --base-features 16 --batch-size 4 --lr 1e-3

# Medium U-Net (default)
submit_job "unet_f32"  unet --base-features 32 --batch-size 2 --lr 1e-3

# Large U-Net (more capacity)
submit_job "unet_f48"  unet --base-features 48 --batch-size 1 --lr 5e-4

# ── FNO3D variants ───────────────────────────────────────────────────────────

# Small FNO (few modes, fast)
submit_job "fno_w16_l3"  fno --width 16 --n-layers 3 --modes 4 8 8 --batch-size 4 --lr 1e-3

# Medium FNO (default)
submit_job "fno_w32_l4"  fno --width 32 --n-layers 4 --modes 8 16 16 --batch-size 2 --lr 1e-3

# Large FNO (more modes + layers)
submit_job "fno_w48_l6"  fno --width 48 --n-layers 6 --modes 12 24 24 --batch-size 1 --lr 5e-4

# ── PINO variants (physics-informed) ────────────────────────────────────────

# UNet PINO (medium + PDE residual + BC loss)
submit_job "unet_f32_pino"  unet --base-features 32 --batch-size 2 --lr 1e-3 \
    --lambda-pde 0.01 --lambda-bc 0.1

# FNO PINO (medium + PDE residual + BC loss)
submit_job "fno_w32_l4_pino"  fno --width 32 --n-layers 4 --modes 8 16 16 --batch-size 2 --lr 1e-3 \
    --lambda-pde 0.01 --lambda-bc 0.1

echo ""
echo "=== 8 jobs submitted ==="
echo "Monitor with: qstat -u maitreje"
echo "Results in: $OUTPUT_DIR/"
