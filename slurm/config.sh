#!/bin/bash
# Shared configuration — sourced by every run_*.sh script in the slurm directory.
# This file sets up the environment and paths for SLURM job submission.

# ============================================================================
# EDIT THESE FOR YOUR CLUSTER ACCOUNT
# ============================================================================

# Email address for SLURM notifications (job start/end/error)
EMAIL="ellorwaizner.nir@mail.huji.ac.il"

# Additional SLURM node arguments (e.g., "--exclude=node42" to skip certain nodes)
NODE_ARGS=""

# Absolute path to this repository on the cluster. Edit to match your setup.
# Example: /cs/labs/raananf/ellorw.nir/FM_Latent_distillation
PROJECT="/cs/labs/raananf/<USER>/FM_Latent_distillation"

# Virtual environment activation + PYTHONPATH setup
# Edit the venv path to match your cluster account.
# The RUN variable is sourced in every sbatch --wrap command to ensure the env is active.
# Example: /cs/labs/raananf/ellorw.nir/venv/bin/activate
RUN="source /cs/labs/raananf/<USER>/venv/bin/activate && cd $PROJECT && export PYTHONPATH=$PROJECT:\$PYTHONPATH &&"

# ============================================================================
# STANDARD SETTINGS (usually no edit needed)
# ============================================================================

# ConvAutoencoder latent dimensions to train FM models for
DIMS=(64 128 256 384 512 1024)

# Directory containing ae_<dim>.pt checkpoints
# Can be absolute or relative to $PROJECT
# You must have: ae_64.pt, ae_128.pt, ae_256.pt, ae_384.pt, ae_512.pt, ae_1024.pt in this directory
# Example absolute: /cs/labs/raananf/<USER>/checkpoints
# Example relative: ./checkpoints (relative to $PROJECT)
CHECKPOINT_DIR="checkpoints"

# Directory containing CIFAR-10 data (relative to $PROJECT)
# The dataset will auto-download to this path if missing
DATADIR="./data"
