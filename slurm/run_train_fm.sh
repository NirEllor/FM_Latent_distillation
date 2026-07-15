#!/bin/bash
# Train Flow-Matching models on latents from custom ConvAutoencoder checkpoints.
# Submits one sbatch job per AE latent dimension (64, 128, 256, 384, 512, 1024).
#
# Scale factors are computed inline for each job by running compute_ae_scale_factor.py,
# ensuring they match the actual checkpoint statistics.
#
# Usage:
#   bash slurm/run_train_fm.sh                        # no prior job dependency
#   bash slurm/run_train_fm.sh afterok:JID1:JID2:...  # with optional dependency (e.g., AE training)
#
# The optional dependency arg chains this job submission to prior SLURM jobs.
# For example, if AE training job IDs are 12345 12346, run:
#   bash slurm/run_train_fm.sh "afterok:12345:12346"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

# ============================================================================
# FM Training Configuration
# ============================================================================

MODEL_TYPE="ddpm++"
LR=2e-4
EMA=true
EMA_DECAY=0.9999
BATCH_SIZE=128
NUM_EPOCH=1000
SAVE_STEP=50
GRAD_CLIP=1.0

# EDM/DDPM++ architecture: sized for 4x4 spatial latent grid (image_size=32, f=8)
# Using fewer layers (ch_mult = 1 2 2) than the full (1 2 2 2) to avoid collapsing resolution
CH_MULT="1 2 2"
ATTN_RES="4"

# ============================================================================
# Submit training jobs
# ============================================================================

echo "Submitting FM training jobs with --ae_type conv_ae"
echo "Model: $MODEL_TYPE | LR: $LR | Epochs: $NUM_EPOCH | Batch: $BATCH_SIZE"
echo "Scale factors will be computed inline per checkpoint"
echo ""

IDS=()
for DIM in "${DIMS[@]}"; do
  # Pre-compute expected channel count
  NUM_CHANNELS=$((DIM / 16))

  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c4 --time=2-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=fm_train_d${DIM} \
    --wrap "$RUN SCALE_FACTOR=\$(python compute_ae_scale_factor.py --ckpt $CHECKPOINT_DIR/ae_${DIM}.pt --latent_dim $DIM 2>/dev/null | grep -- '--scale_factor' | tail -1 | awk '{print \$2}') && echo \"Scale factor: \$SCALE_FACTOR\" && python train_flow_latent.py \
      --exp latent_${DIM} \
      --dataset cifar10 \
      --datadir $DATADIR \
      --ae_type conv_ae \
      --ae_latent_dim $DIM \
      --pretrained_autoencoder_ckpt $CHECKPOINT_DIR/ae_${DIM}.pt \
      --scale_factor \$SCALE_FACTOR \
      --model_type $MODEL_TYPE \
      --image_size 32 \
      --f 8 \
      --num_in_channels $NUM_CHANNELS \
      --num_out_channels $NUM_CHANNELS \
      --ch_mult $CH_MULT \
      --attn_resolutions $ATTN_RES \
      --num_epoch $NUM_EPOCH \
      --lr $LR \
      --batch_size $BATCH_SIZE \
      --use_ema \
      --ema_decay $EMA_DECAY \
      --save_content \
      --save_content_every $SAVE_STEP" \
    | awk '{print $NF}')

  echo "  ✓ Submitted fm_train dim=$DIM (channels=$NUM_CHANNELS) → Job $JOB"
  IDS+=($JOB)
done

echo ""
echo "All jobs submitted. IDs: ${IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER              # all your jobs"
echo "  squeue -j ${IDS[0]}           # first job details"
echo "  tail -f slurm-${IDS[0]}.out   # stream first job output"
