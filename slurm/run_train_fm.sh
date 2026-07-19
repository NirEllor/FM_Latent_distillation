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

# Verify AE checkpoint files exist
echo "Checking for ConvAutoencoder checkpoints in: $CHECKPOINT_DIR"
for DIM in "${DIMS[@]}"; do
  CKPT_FILE="$CHECKPOINT_DIR/ae_${DIM}.pt"
  if [ ! -f "$CKPT_FILE" ]; then
    echo "✗ ERROR: Checkpoint not found: $CKPT_FILE"
    echo "  Make sure ae_<dim>.pt files are in the checkpoint directory."
    exit 1
  fi
  echo "  ✓ Found ae_${DIM}.pt"
done
echo ""

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

# ============================================================================
# FM Training Configuration
# ============================================================================

MODEL_TYPE="ddpm++"
LR=2e-4
EMA=true
EMA_DECAY=0.9999
BATCH_SIZE=8
GRAD_CHECKPOINTING=true
MIXED_PRECISION="fp16"
NUM_EPOCH=1000
SAVE_STEP=50
GRAD_CLIP=1.0

# EDM/DDPM++ architecture: sized for 4x4 spatial latent grid (image_size=32, f=8)
# Using fewer layers (ch_mult = 1 2 2) than the full (1 2 2 2) to avoid collapsing resolution
CH_MULT="1 2 2"
ATTN_RES="4"

# Base model channels (nf). This determines the bottleneck capacity.
# With ch_mult=[1,2,2], bottleneck_channels = nf * 2.
# Formula: nf = latent_dim * 2 (ensures bottleneck capacity ≈ 64× input channels)
# Examples with 60GB memory allocation:
#   ae_64 (4 ch)  → nf=128 → bottleneck=256 channels
#   ae_128 (8 ch) → nf=256 → bottleneck=512 channels
#   ae_256 (16 ch)→ nf=512 → bottleneck=1024 channels
#   ae_384 (24 ch)→ nf=768 → bottleneck=1536 channels
#   ae_512 (32 ch)→ nf=1024→ bottleneck=2048 channels
#   ae_1024(64 ch)→ nf=2048→ bottleneck=4096 channels

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

  # Adaptive batch size based on model size (with increased memory, NF can be full capacity)
  # NF formula: nf = latent_dim * 2 (ensures 64× bottleneck capacity)
  NF=$((DIM * 2))

  if [ $DIM -le 128 ]; then
    ADAPTIVE_BATCH=8
  elif [ $DIM -le 256 ]; then
    ADAPTIVE_BATCH=6
  elif [ $DIM -le 384 ]; then
    ADAPTIVE_BATCH=4
  else
    ADAPTIVE_BATCH=2
  fi

  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=60G -c4 --time=4-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=fm_train_d${DIM} \
    --wrap "bash -c '$RUN SCALE_FACTOR=\$(python compute_ae_scale_factor.py --ckpt $CHECKPOINT_DIR/ae_${DIM}.pt --latent_dim $DIM 2>/dev/null | grep -- \"--scale_factor\" | tail -1 | awk \"{print \\\$2}\") && echo \"Scale factor: \$SCALE_FACTOR\" && ACCELERATE_MIXED_PRECISION=$MIXED_PRECISION PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train_flow_latent.py \
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
      --nf $NF \
      --ch_mult $CH_MULT \
      --attn_resolutions $ATTN_RES \
      --num_epoch $NUM_EPOCH \
      --lr $LR \
      --batch_size $ADAPTIVE_BATCH \
      --use_grad_checkpointing \
      --save_content \
      --save_content_every $SAVE_STEP'" \
    | awk '{print $NF}')

  echo "  ✓ Submitted fm_train dim=$DIM (channels=$NUM_CHANNELS, nf=$NF, bottleneck=$((NF*2)), batch=$ADAPTIVE_BATCH) → Job $JOB"
  IDS+=($JOB)
done

echo ""
echo "All jobs submitted. IDs: ${IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER              # all your jobs"
echo "  squeue -j ${IDS[0]}           # first job details"
echo "  tail -f slurm-${IDS[0]}.out   # stream first job output"
