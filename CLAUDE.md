# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of **"Flow Matching in Latent Space"** (https://arxiv.org/abs/2307.08698), a generative model framework that performs flow matching in the latent spaces of pretrained autoencoders. This enables efficient, high-resolution image synthesis (CelebA-HQ, FFHQ, LSUN, ImageNet) with improved computational efficiency compared to pixel-space diffusion models.

### Key Features
- Flow matching training in VAE latent spaces for computational efficiency
- Multiple model architectures: DiT (Vision Transformers) and ADM (U-Nets)
- Support for conditional generation: label conditioning, inpainting, semantic-to-image synthesis
- Multi-GPU training via `accelerate` framework with automatic mixed precision support
- ODE-based sampling with multiple solver options (adaptive: dopri5; fixed-step: euler, heun)

## Technical Skills

### AE: Autoencoder (VAE in this project)

**What it is**: A Variational Autoencoder (VAE) that compresses high-dimensional pixel-space images into low-dimensional latent representations. In this project, it's a **frozen pretrained model** from Hugging Face Diffusers library.

**Architecture**:
- **Encoder**: Downsamples images by factor of 8 (e.g., 256×256 → 32×32) while increasing channel dimension
- **Latent space**: 4-channel representation (standard for Stable Diffusion VAE)
- **Decoder**: Upsamples latents back to pixel space

**Key Parameters**:
- **Compression factor `f=8`**: Determines latent resolution (latent_res = image_size / f)
  - 256×256 image → 32×32 latent
  - 512×512 image → 64×64 latent
  - 1024×1024 image → 128×128 latent
- **Scale factor `0.18215`**: Normalizes latent distribution to ~N(0,1) for stable training
  - Applied in training: `latent = encoder(image) * scale_factor`
  - Applied in inference: `image = decoder(latent / scale_factor)`

**Code Location**:
- VAE loading: `train_flow_latent.py:75` → `AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt)`
- Default pretrained model path: typically `stabilityai/sd-vae-ft-mse` from Hugging Face
- Encoder is in `models/encoder.py` (custom implementation for reference)

**Important Properties**:
1. **Frozen during training**: `first_stage_model.eval()` and `param.requires_grad = False`
   - Flow matching model learns in latent space, not pixel space
   - Saves 64× computation (8³) compared to pixel-space diffusion
2. **Why scale_factor matters**: 
   - Without it: latents have variance >> 1, making training numerically unstable
   - With it: latents are normalized, enabling stable gradient descent
   - Must be consistent between training and inference, or model outputs will be corrupted
3. **Latent dimensionality**: Always 4 channels (`--num_in_channels 4`, `--num_out_channels 4`)

**Usage Pattern**:
```python
# Encode: images → latents
images = batch["image"]  # shape: (B, 3, H, W), range: [-1, 1]
with torch.no_grad():
    latents = first_stage_model.encode(images).latent_dist.sample()
    latents = latents * scale_factor  # normalize

# Decode: latents → images
with torch.no_grad():
    latents_denorm = latents / scale_factor
    images_recon = first_stage_model.decode(latents_denorm / 0.18215).sample
```

**Troubleshooting**:
- **Black/corrupted output**: Check if scale_factor is being applied correctly
- **Training instability**: Verify VAE is frozen and gradients don't flow through it
- **Shape mismatch**: Remember latent spatial dims are 1/8 of image dims

---

#### Custom ConvAutoencoder Integration

You can also use **custom-trained ConvAutoencoder checkpoints** (e.g., `ae_64.pt` ... `ae_1024.pt`) instead of the Hugging Face VAE. This is useful for training on CIFAR-10 with custom architectures.

**Latent Dimensions Supported**:
- `ae_64.pt` → 4 channels (64 / 16) at 4×4 spatial
- `ae_128.pt` → 8 channels (128 / 16) at 4×4 spatial
- `ae_256.pt` → 16 channels (256 / 16) at 4×4 spatial
- `ae_384.pt` → 24 channels (384 / 16) at 4×4 spatial
- `ae_512.pt` → 32 channels (512 / 16) at 4×4 spatial
- `ae_1024.pt` → 64 channels (1024 / 16) at 4×4 spatial

**Key Differences from SD-VAE**:
- **Input images**: Must be in `[0, 1]` (not `[-1, 1]`). Automatically handled if using `--ae_type conv_ae`.
- **Deterministic encoding**: No KL sampling — `.encode()` returns a fixed representation
- **Scale factor**: Not `0.18215`. Must be computed per checkpoint via `compute_ae_scale_factor.py` (see below)
- **Architecture**: ResNet-based encoder/decoder, fully convolutional

**Training with Custom AE**:

1. **Compute scale factor** (once per checkpoint):
   ```bash
   python compute_ae_scale_factor.py --ckpt <path/to/ae_256.pt> --latent_dim 256
   ```
   Output: `Recommended scale_factor for latent_dim=256: 1.234567`

2. **Train FM model**:
   ```bash
   accelerate launch --num_processes 1 train_flow_latent.py --exp cifar_convae_256 \
     --dataset cifar10 --datadir ./data \
     --ae_type conv_ae --ae_latent_dim 256 \
     --pretrained_autoencoder_ckpt <path/to/ae_256.pt> \
     --scale_factor 1.234567 \
     --image_size 32 --f 8 --batch_size 32 --num_epoch 100 \
     --model_type DiT-B/2 --num_classes 1 --label_dropout 0.1 \
     --lr 2e-4 --save_content --save_content_every 10
   ```
   Notes:
   - `--ae_type conv_ae` activates custom AE mode
   - `--ae_latent_dim` specifies which checkpoint to load
   - `--num_in_channels` and `--num_out_channels` are **auto-derived** from latent_dim (no need to specify)
   - CIFAR-10 images are automatically kept in `[0, 1]` when using conv_ae

3. **Sampling & FID Evaluation**:
   ```bash
   python test_flow_latent.py --exp cifar_convae_256 --dataset cifar10 \
     --ae_type conv_ae --ae_latent_dim 256 \
     --pretrained_autoencoder_ckpt <path/to/ae_256.pt> \
     --scale_factor 1.234567 \
     --epoch_id 100 --batch_size 100 --n_sample 10000 \
     --model_type DiT-B/2 --num_classes 1
   ```

**Unified AE Interface** (`models/ae_backends.py`):
- `load_first_stage_model(args, device, dtype)` → loads either SD-VAE or ConvAE based on `--ae_type`
- `encode_to_latent(model, x, args)` → handles both `.encode()` APIs and shape conversions
- `decode_from_latent(model, z, args)` → handles both `.decode()` APIs, remaps [0,1]→[-1,1] for ConvAE so downstream code is unchanged
- All training/sampling scripts use these helpers, so both AE types work seamlessly

**File Locations**:
- Custom AE class: `models/conv_autoencoder.py`
- Unified backend interface: `models/ae_backends.py`
- Scale-factor calibrator: `compute_ae_scale_factor.py` (repo root)

**Troubleshooting**:
- **"ae_latent_dim must be specified"**: Always pass `--ae_latent_dim <dim>` when using `--ae_type conv_ae`
- **Shape mismatches**: FM model's `num_in_channels`/`num_out_channels` are auto-set to `latent_dim // 16`; verify the FM checkpoint was trained with matching settings
- **Poor sample quality**: Check that `--scale_factor` matches what was used during training (print output from `compute_ae_scale_factor.py`)
- **Decode output looks wrong**: Verify the checkpoint file exists and is uncorrupted (should have `"state_dict"` key)

---

### FM: Flow Matching

**What it is**: A modern generative modeling framework that learns a continuous flow (trajectory) from noise to data in latent space. More stable and efficient than diffusion models.

**Core Concept**:
- **Goal**: Learn a vector field that transforms random noise (z₁ ~ N(0,I)) into data samples (z₀ ~ p_data)
- **Formulation**: At each time t ∈ [0,1], predict velocity v(z_t, t) such that following it from t=1 to t=0 reconstructs data
- **Advantage over diffusion**: Direct regression instead of iterative denoising, potentially fewer function evaluations

**Mathematical Foundation**:
```
Probability path: p_t(z) = (1-t)·p_data(z) + t·p_noise(z)
Time: t ∈ [0, 1] where t=0 is data, t=1 is noise

ODE: dz/dt = v(z_t, t) where z solves backward from t=1 to t=0

Loss: L = ||v_θ(z_t, t) - (z_1 - z_0)||²_2  (mean squared error in velocity space)
```

**Code Location**:
- Flow matching loss computation: Search `train_flow_latent.py` for MSE loss or velocity prediction
- ODE solver integration: `torchdiffeq.odeint()` in `sample_from_model()` function
- Model forward pass: `model(t, z_t)` returns predicted velocity

**Training Process**:
1. Sample random time: `t ~ Uniform[0, 1]` (or from discrete distribution)
2. Sample data and noise: `z_0 ~ p_data` (VAE-encoded images), `z_1 ~ N(0,I)`
3. Interpolate: `z_t = (1-t)·z_0 + t·z_1` (linear interpolation in latent space)
4. Target velocity: `v_target = z_1 - z_0` (direction from data to noise)
5. Predicted velocity: `v_pred = model(t, z_t)` (what model learns)
6. Loss: `L = ||v_pred - v_target||²`
7. Backprop and update: `optimizer.step()`

**Sampling (Inference)**:
```
z_1 ~ N(0, I)  # start from pure noise
Solve ODE: dz/dt = v_θ(z_t, t) from t=1.0 to t=0.0
z_0 = result   # this is the latent sample
image = VAE_decode(z_0 / scale_factor)
```

**Solver Options** (in `torchdiffeq`):

| Solver | Type | NFE | Speed | Accuracy | Use Case |
|--------|------|-----|-------|----------|----------|
| `dopri5` | Adaptive | ~20-50 | Fast | Good | Default for sampling |
| `dopri8` | Adaptive | ~30-100 | Slower | Excellent | When accuracy matters |
| `euler` | Fixed-step | = STEPS | Fastest | Lower | Quick sampling |
| `heun` | Fixed-step | = STEPS | Fast | Medium | Fast high-quality |
| `rk4` | Fixed-step | = STEPS | Medium | Good | Balanced |

**Conditional Flow Matching**:
- **Label conditioning**: Append class embedding to noise vector before ODE solve
- **Inpainting**: Set initial noise on masked regions only, keep original latents on unmasked
- **Semantic synthesis**: Condition on segmentation map embeddings concatenated to latent

**Key Hyperparameters**:
- **`--lr` (learning rate)**: Controls step size; smaller for large models (1e-4 to 2e-4)
- **ODE tolerances** (`atol`, `rtol`): Relative/absolute error tolerances for adaptive solvers
  - `atol=1e-5, rtol=1e-5` (default) provides good balance
  - Tighter tolerances → longer sampling, higher accuracy
  - Looser tolerances → faster sampling, lower accuracy
- **Time schedule**: Currently linear interpolation (could use non-linear schedules for better results)

**Advantages**:
1. **Fewer function evaluations**: ~20-50 NFE vs 50-100+ for diffusion
2. **Direct objective**: Velocity regression is simpler than denoising
3. **Theoretical guarantees**: Paper provides Wasserstein-2 bound on reconstructed vs true distribution
4. **Flexible conditioning**: Easy to add conditions (labels, masks, semantic maps)

**Troubleshooting**:
- **Diverging loss**: Check learning rate, gradient clipping, batch normalization impact
- **Poor sample quality**: Verify time schedule is [1.0 → 0.0], check VAE scale_factor
- **High NFE count**: Increase ODE tolerances or switch to fixed-step solver
- **Mode collapse**: Try increasing batch size, adding noise during training

**Related Concepts**:
- **Diffusion models**: Flow matching's predecessor; FM is more efficient
- **Score-based models**: Related framework; FM is arguably cleaner
- **Neural ODE**: Underlying mathematical framework (solving continuous dynamics)

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```
Requires Python 3.10+ and PyTorch 1.13.1+ with CUDA support (training requires GPU).

## Git Workflow

**For every work completed, follow this workflow:**

1. **Stage changes**:
   ```bash
   git add <file>  # Add specific files you modified
   ```

2. **Commit with a clear message** (explain the "why", not just the "what"):
   ```bash
   git commit -m "Brief description of change"
   # Example: "refactor: optimize VAE latent encoding for inference speed"
   # Example: "fix: correct scale_factor application in loss computation"
   # Example: "feat: add support for semantic-guided inpainting"
   ```

3. **Push to GitHub**:
   ```bash
   git push origin main
   ```

Keep commits focused and atomic—one logical change per commit. This makes the history clear and enables easy reversion if needed.

### Key Directory Structure
- **root scripts**: `train_flow_latent.py`, `test_flow_latent.py` (main entry points)
- **models/**: Network architectures (`DiT.py`, `EDM.py`, guided_diffusion U-Net, `encoder.py`)
- **datasets_prep/**: Dataset loaders and transforms (LMDB, LSUN, ImageNet, CIFAR10)
- **downstream_tasks/**: Conditional generation (inpainting, semantic synthesis)
- **sampler/**: ODE sampling utilities (Karras samplers, NFE counting)
- **pytorch_fid/**: FID metric computation
- **bash_scripts/**: Pre-configured training/testing commands
- **slurm/**: SLURM cluster submission scripts for distributed training

## SLURM Cluster Training (Distributed FM on Custom AE Latents)

If you have 6 trained `ConvAutoencoder` checkpoints and want to train FM models on their latents in parallel across your SLURM cluster, use the provided scripts.

### Prerequisites
1. **ConvAutoencoder checkpoints** (`ae_<dim>.pt` for each dim):
   - Place in `checkpoints/` directory (or update `CHECKPOINT_DIR` in `slurm/config.sh`)
   - Files needed: `ae_64.pt`, `ae_128.pt`, `ae_256.pt`, `ae_384.pt`, `ae_512.pt`, `ae_1024.pt`
   - Each checkpoint must match the format (has `{"latent_dim": ..., "state_dict": ...}`)

2. **CIFAR-10 data**:
   - Will auto-download to `./data` if missing
   - Or point `DATADIR` in `slurm/config.sh` to existing dataset

### Configuration
Edit `slurm/config.sh`:
- `EMAIL`: SLURM notification email
- `PROJECT`: absolute path to this repo on your cluster
- `RUN`: venv activation command (edit the venv path)
- `CHECKPOINT_DIR`: where your `ae_<dim>.pt` files are stored (relative or absolute)

### Submit Jobs
```bash
# Submit all 6 FM training jobs (one per AE dim)
bash slurm/run_train_fm.sh

# Or with job dependency chaining (e.g., wait for prior AE training)
bash slurm/run_train_fm.sh "afterok:12345:12346"
```

### What Gets Trained
Each job trains an **independent FM model** in the latent space of its corresponding AE:
- **latent_64 FM**: 4-channel (64÷16) latents, `ddpm++` U-Net
- **latent_128 FM**: 8-channel latents, `ddpm++` U-Net
- **latent_256 FM**: 16-channel latents, `ddpm++` U-Net
- **latent_384 FM**: 24-channel latents, `ddpm++` U-Net
- **latent_512 FM**: 32-channel latents, `ddpm++` U-Net
- **latent_1024 FM**: 64-channel latents, `ddpm++` U-Net

All 6 FM models train **in parallel**, each using 1 GPU for 1000 epochs (≈2 days per job).

### Key Details
- **Latent-space training**: Images → AE encoder (deterministic, no sampling) → latents → FM learns velocity **in latent space only**
- **Deterministic encoding**: `ConvAutoencoder.encode()` is fully deterministic (no KL sampling). Same image always → same latent
- **Scale factor**: Computed **inline per job** by `compute_ae_scale_factor.py` — ensures correct normalization per checkpoint
- **Architecture**: `ddpm++` (EDM-style conv U-Net), sized for 4×4 spatial latent grid
- **Checkpoints**: Saved to `saved_info/latent_flow/cifar10/latent_<dim>/` every 50 epochs

---

## Training

### General Training Command
```bash
accelerate launch [--num_processes N] [--multi_gpu] train_flow_latent.py --exp <exp_name> \
  --dataset <dataset> --datadir <path> --batch_size <bs> --num_epoch <epochs> \
  --image_size <size> --f 8 --num_in_channels 4 --num_out_channels 4 \
  --lr <lr> --scale_factor 0.18215 --save_content --save_content_every 10
```

### Model Variants
- **DiT (Vision Transformer)**: `--model_type DiT-B/2` or `DiT-L/2` (requires `--num_classes`, `--label_dropout`)
- **ADM (U-Net)**: `--use_origin_adm` (uses guided_diffusion architecture)
- **EDM**: Default if neither option specified (modern scaling)

### Important Parameters
- `--f`: Latent compression factor (typically 8)
- `--scale_factor`: VAE latent normalization (standard: 0.18215)
- `--num_classes`: For label conditioning (e.g., 1000 for ImageNet, 1 for unconditional)
- `--ch_mult`: Channel multipliers for U-Net layers (e.g., "1 2 3 4")
- `--attn_resolution`: Resolution levels with attention (e.g., "16 8 4")
- `--use_grad_checkpointing`: Enable for large models to save memory
- `--use_ema`: Apply exponential moving average to model weights (default: disabled)

### Example Training Configs
Refer to `bash_scripts/run.sh` for complete configuration examples covering CelebA-256, FFHQ, LSUN, ImageNet at various resolutions (256-1024px) with both DiT and ADM architectures.

### Resuming Training
- Model checkpoints saved to `saved_info/latent_flow/<dataset>/<exp>/` every `--save_content_every` epochs
- To resume: manually set `--start_epoch` if the training script supports it (check arg parsing in `train_flow_latent.py`), or delete checkpoints and restart

## Testing / Sampling

### Single-GPU Sampling
```bash
bash bash_scripts/run_test.sh test_args/celeb256_dit.txt
```
Arguments file format (example from `test_args/celeb256_dit.txt`):
```
MODEL_TYPE=DiT-L/2
EPOCH_ID=475
DATASET=celeba_256
EXP=celeb_f8_dit
METHOD=dopri5
STEPS=0
USE_ORIGIN_ADM=False
IMG_SIZE=256
```

- `METHOD`: Solver choice (`dopri5` for adaptive, `euler`/`heun` for fixed-step)
- `STEPS`: Number of fixed steps (0 = adaptive, set > 0 for fixed-step solvers)
- `EPOCH_ID`: Checkpoint epoch to load

### Multi-GPU Evaluation (FID Computation)
```bash
bash bash_scripts/run_test_ddp.sh <args_file>  # unconditional
bash bash_scripts/run_test_cls_ddp.sh <args_file>  # conditional generation
```
Requires 8 GPUs by default (configurable via DDP setup).

### Pre-computed Checkpoints
Download pre-trained models from Google Drive links in README and place in:
```
saved_info/latent_flow/<DATASET>/<EXP>/model_<EPOCH_ID>.pth
```

### Sampling Options
- `--measure_time`: Report wall-clock sampling time
- `--compute_nfe`: Count number of function evaluations (adaptive solvers only)
- `--use_karras_samplers`: Enable fixed-step solvers; requires `METHOD` and `STEPS` adjustment
- `--cfg_scale`: Classifier-free guidance scale (> 1.0 for guidance, = 1.0 for none)

### FID Evaluation
1. **Download pre-computed stats**:
   ```bash
   # From Google Drive: https://drive.google.com/drive/folders/1BXCqPUD36HSdrOHj2Gu_vFKA3M3hJspI
   # Place in pytorch_fid/
   ```

2. **Compute new dataset stats**:
   ```bash
   python pytorch_fid/compute_dataset_stat.py \
     --dataset <name> --datadir <path> --image_size <size> --save_path <output>
   ```

3. **Compute FID score**:
   ```bash
   python pytorch_fid/fid_score.py <path_to_generated> <path_to_real>
   ```

## Downstream Tasks

### Setup
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Image Inpainting
**Train**:
```bash
python downstream_tasks/train_flow_latent_inpainting.py --exp inpainting_kl \
  --dataset celeba_256 --batch_size 64 --lr 5e-5 --scale_factor 0.18215 \
  --num_epoch 500 --image_size 256 --num_in_channels 9 --num_out_channels 4 \
  --ch_mult 1 2 3 4 --attn_resolution 16 8 --num_process_per_node 2 --save_content
```

**Test**:
```bash
python downstream_tasks/test_flow_latent_inpainting.py --exp inpainting_kl \
  --dataset celeba_256 --batch_size 64 ...
python pytorch_fid/cal_inpainting.py <generated> <ground_truth>
```

### Semantic Synthesis
**Train**:
```bash
python downstream_tasks/train_flow_latent_semantic_syn.py --exp semantic_kl \
  --dataset celeba_256 --batch_size 64 --lr 5e-5 --scale_factor 0.18215 \
  --num_epoch 175 --image_size 256 --num_in_channels 8 --num_out_channels 4 \
  --ch_mult 1 2 3 4 --attn_resolution 16 8 --num_process_per_node 2 --save_content
```

**Test**:
```bash
python downstream_tasks/test_flow_latent_semantic_syn.py --exp semantic_kl \
  --dataset celeba_256 --batch_size 64 ...
python pytorch_fid/fid_score.py <generated> <ground_truth>
```

## Code Architecture

### Training Flow (`train_flow_latent.py`)
1. Initialize accelerator for multi-GPU training
2. Load dataset via `datasets_prep.get_dataset()`
3. Create flow-matching model via `models.create_network()`
4. Load frozen pretrained VAE (`diffusers.AutoencoderKL`)
5. Training loop:
   - Encode images to latent space via VAE encoder
   - Sample random noise and time steps
   - Compute flow-matching loss (L2 distance between predicted and target velocity)
   - Backward pass and optimizer step
   - EMA update (if enabled)
   - Save checkpoints and sample visualizations

### Model Creation (`models/__init__.py`)
- `create_network()`: Factory function that routes to DiT, EDM, or ADM based on config
- **DiT**: Vision Transformer architecture from `DiT.py` (modern, efficient)
- **EDM**: Style-based diffusion U-Net from `EDM.py`
- **ADM**: OpenAI's guided diffusion U-Net from `guided_diffusion/unet.py` (legacy, uses `--use_origin_adm`)

### Sampling (`test_flow_latent.py`)
1. Load trained model from checkpoint
2. Sample noise from N(0,I) in latent space
3. Solve ODE from t=1.0 to t=0.0 using `torchdiffeq.odeint()`
4. Decode latent via VAE decoder to pixel space
5. Compute FID metrics using Inception network

### Dataset Loading (`datasets_prep/__init__.py`)
- Supports: CIFAR10, ImageNet, LSUN (church, bedroom), CelebA-HQ, FFHQ
- Uses LMDB format for large datasets (CelebA, FFHQ)
- Standard transforms: resize, horizontal flip, normalize to [-1, 1]

## Key Design Decisions & Patterns

### Latent Space Training
- All models operate on **8x-compressed VAE latents** (f=8), not pixel space
- Scale factor `0.18215` normalizes latent distribution (important for numerical stability)
- VAE is frozen during training (pretrained diffusers model)

### ODE Integration
- `torchdiffeq` library wraps scipy/Dopri5 solvers and custom implementations
- **Adaptive solvers** (dopri5, dopri8): Variable step size, lower NFE, numerical precision
- **Fixed solvers** (euler, heun, rk4): Deterministic steps, used with Karras sampler
- Solved backward in time: t=[1.0, 0.0] (from noise to data)

### Distributed Training
- `accelerate` framework abstracts multi-GPU/multi-node details
- `accelerate.Accelerator` handles device placement, mixed precision, DDP, gradient accumulation
- Config: `--num_processes` (single node) or `--multi_gpu` + `--num_processes` (multi-node)

### Model Checkpointing
- Saved as `.pth` files with model state dict
- Located in `saved_info/latent_flow/<DATASET>/<EXP>/model_<EPOCH>.pth`
- Optimizer and scheduler state not persisted (training doesn't resume state, restarts from scratch)

### Conditional Generation (Downstream)
- **Label conditioning**: Prepend class embedding to noise
- **Inpainting**: Concatenate mask + original latents to noise (9 channels: 4 noise + 4 original + 1 mask)
- **Semantic synthesis**: Concatenate segmentation map embeddings (8 channels: 4 noise + 4 embeddings)

## Common Development Tasks

### Adding a New Dataset
1. Create loader in `datasets_prep/__init__.py` under `get_dataset()`
2. Implement standard transforms: `Resize → RandomHorizontalFlip → ToTensor → Normalize((-1, 1))`
3. Return torch dataset object supporting `__len__` and `__getitem__`
4. Update bash_scripts for training commands

### Modifying the Model
1. Edit architecture in `models/DiT.py`, `EDM.py`, or `guided_diffusion/unet.py`
2. Update `models/create_network()` if adding new model types
3. Ensure output shape is `(batch, 4, H//8, W//8)` for 256×256 images
4. Test with dummy batch: `model(torch.randn(2, 4, 32, 32), t_batch)`

### Changing Loss Functions
1. Loss computation in `train_flow_latent.py` (search for "loss" or "mse")
2. Flow matching loss: MSE between predicted velocity and target velocity in latent space
3. Ensure loss is scalar and differentiable for `loss.backward()`

### Debugging Training
- Check `saved_info/latent_flow/<EXP>/log.txt` for per-epoch losses and sampling time
- Visualizations saved every `--save_content_every` epochs in `saved_info/latent_flow/<EXP>/samples/`
- Use `--measure_time` flag to profile sampling speed
- Add `torch.autograd.set_detect_anomaly(True)` in training loop for NaN detection

### Running on Different Hardware
- **Single GPU**: `accelerate launch --num_processes 1 train_flow_latent.py ...`
- **Multiple GPUs (same node)**: `accelerate launch --multi_gpu --num_processes 8 train_flow_latent.py ...`
- **Mixed precision (BF16)**: Add `--mixed_precision bf16` to accelerate launch
- **Lower memory**: Enable `--use_grad_checkpointing`, reduce `--batch_size`, increase `--num_accumulation_steps`

## Important Constants & Conventions

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `--f` | 8 | VAE compression; output latent size = image_size / 8 |
| `--scale_factor` | 0.18215 | VAE latent normalization (diffusers standard) |
| `--num_in_channels` | 4 | VAE latent channels (standard for Stable Diffusion VAE) |
| `--num_out_channels` | 4 | Model output channels (same as input for flow matching) |
| `--lr` | 1e-4 to 2e-4 | Learning rate (lower for large models/datasets) |
| `--ema_decay` | 0.9999 | EMA momentum (if `--use_ema` enabled) |
| `--seed` | 42 (default) | Random seed for reproducibility |
| Time interval [t] | [1.0, 0.0] | Solving from noise (t=1) to data (t=0) |
| `atol`, `rtol` | 1e-5 (default) | ODE solver tolerances |

## References & External Links

- **Paper**: https://arxiv.org/abs/2307.08698
- **Project Page**: https://vinairesearch.github.io/LFM/
- **Dependencies**:
  - torchdiffeq: ODE integration (https://github.com/rtqichen/torchdiffeq)
  - accelerate: Multi-GPU training (https://huggingface.co/docs/accelerate)
  - diffusers: Pretrained VAE models (https://huggingface.co/docs/diffusers)
- **Architecture References**:
  - DiT: https://github.com/facebookresearch/DiT
  - ADM: https://github.com/openai/guided-diffusion
  - EDM: https://github.com/NVlabs/edm