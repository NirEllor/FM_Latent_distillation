#!/bin/bash
# Verify FM model shapes via SLURM job (needs GPU memory)
# Usage: bash slurm/run_verify_fm_shapes.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

# Submit verification as a SLURM job
JOB=$(sbatch $NODE_ARGS \
  --mem=50G -c4 --gres=gpu:1 --time=00:30:00 \
  --job-name=verify_fm_shapes \
  --wrap "bash -c '$RUN python slurm/verify_fm_shapes.py'" \
  | awk '{print $NF}')

echo "Submitted verification job: $JOB"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB"
echo "  tail -f slurm-${JOB}.out"
