# Architecture: Flow Matching in Latent Space

This document describes the overall system architecture for training and sampling with Conditional Flow Matching (CFM) models operating in autoencoder latent spaces.

## System Overview

```
Training Pipeline:
  Raw Images (32×32×3)
       ↓
  [Frozen AE Encoder] (deterministic)
       ↓
  Latent Space (B, C, 4, 4) where C = latent_dim/16
       ↓
  [FM Model] learns velocity field v(z_t, t)
       ↓
  Checkpoint saved

Sampling Pipeline:
  Random Noise z_1 ~ N(0, I)
       ↓
  Solve ODE: dz/dt = v_θ(z_t, t) from t=1 to t=0
       ↓
  [Frozen AE Decoder]
       ↓
  Generated Images
```

---

## Component 1: Autoencoders (First-Stage Models)

### Purpose
Compress high-dimensional pixel space to low-dimensional latent space, reducing computational cost from O(H×W) to O((H/8)×(W/8)).

### Two AE Backends

#### **Standard Diffusion VAE** (`--ae_type sd_vae`, default)
- **Source**: Hugging Face `diffusers.AutoencoderKL`
- **Encoder**: Images (256×256 or arbitrary size) → latents (32×32 or f=8 compression)
- **Latent channels**: Fixed at 4
- **Pixel convention**: [-1, 1]
- **Determinism**: Uses `.latent_dist.sample()` — **stochastic** (KL sampling)
- **Use case**: Existing large-scale image datasets (CelebA, FFHQ, ImageNet)

#### **Custom ConvAutoencoder** (`--ae_type conv_ae`)
- **Architecture**: ResNet blocks + spatial downsampling (3 levels) / upsampling
- **Encoder**: CIFAR-10 (32×32×3) → latents (4×4 spatial, C channels)
- **Latent channels**: Variable (C = latent_dim / 16)
  - ae_64 → 4 channels
  - ae_128 → 8 channels
  - ae_256 → 16 channels
  - ae_384 → 24 channels
  - ae_512 → 32 channels
  - ae_1024 → 64 channels
- **Pixel convention**: [0, 1]
- **Determinism**: Returns fixed latents — **fully deterministic** (no sampling)
- **Use case**: CIFAR-10 with custom latent dimensions

### Frozen During Training
Both AE types are **frozen** during FM training:
```python
first_stage_model.eval()
for p in first_stage_model.parameters():
    p.requires_grad = False
```
Only the FM model learns; AE weights are fixed.

### Scale Factor
Each latent space is normalized by a scale factor to ensure unit variance:
- **SD-VAE**: scale_factor = 0.18215 (pre-calibrated)
- **ConvAE**: computed per checkpoint via `compute_ae_scale_factor.py`

Application:
```python
z_0 = encode(images) * scale_factor        # Training
images = decode(latents / scale_factor)    # Inference
```

---

## Component 2: FM Model (Velocity Predictor)

### Purpose
Learn a vector field v(z_t, t) that guides latent samples from noise (t=1) to data (t=0).

### Model Types (Backbone Architectures)

#### **DiT (Diffusion Transformer)** — `--model_type DiT-B/2` or `DiT-L/2`
- **Architecture**: Vision Transformer with patch embedding
- **Patch size**: 2 (on 4×4 latent → 2×2 token grid)
- **Input shape**: (B, C, 4, 4) → patchify → (B, 4, 256) tokens
- **Pros**: Modern, parameter-efficient
- **Cons**: Very few tokens (4) on tiny latent grids; not ideal for CIFAR-10
- **Use case**: Larger images / larger latent spaces

#### **EDM UNet (DDPMpp / NCSN++)** — `--model_type ddpm++`, `ncsn++`, `adm`
- **Architecture**: Hierarchical CNN with multi-scale convolutions
- **Downsampling levels**: 3 (4×4 → 2×2 → 1×1)
- **Channel progression**: input_ch → base_channels → base_channels * ch_mult[i]
  - Bottleneck at 1×1 with base_channels * max(ch_mult) channels
- **Self-attention**: Optional at specified resolutions (e.g., `--attn_resolutions 4`)
- **Pros**: Works well on small spatial grids (4×4)
- **Cons**: More parameters than DiT
- **Use case**: CIFAR-10 latent-space FM (recommended)

### Input/Output Specification
```python
# Forward pass
v_pred = model(t, z_t)

# Shapes
t: (B,) — timestep
z_t: (B, num_in_channels, 4, 4) — latent at time t
v_pred: (B, num_out_channels, 4, 4) — velocity prediction

# For FM: num_in_channels == num_out_channels == latent_dim // 16
```

### Bottleneck Capacity (Critical)
The UNet bottleneck (at 1×1 spatial) must have sufficient capacity:
```
bottleneck_channels = nf * max(ch_mult)
```

**Scaling rule** (for ConvAE):
```bash
nf = latent_dim * 8
# Ensures bottleneck = latent_dim * 16 (64× input capacity)

Example:
  ae_256 (16 channels):
    nf = 256 * 8 = 2048
    bottleneck = 2048 * 2 = 4096 channels
    capacity ratio = 4096 / 16 = 256×
```

---

## Component 3: Flow Matching Training

### Probability Path (Linear/Straight)
```
p_t(z) = (1 - t) * p_data(z) + t * p_noise(z)

where:
  t ∈ [0, 1]
  t=0: data distribution
  t=1: noise distribution
```

### Training Loop
1. **Encode images to latents**:
   ```python
   z_0 = AE.encode(images) * scale_factor  # Data latents
   ```

2. **Sample random time and noise**:
   ```python
   t ~ Uniform[0, 1]
   z_1 ~ N(0, I)  # Noise latents
   ```

3. **Interpolate**:
   ```python
   z_t = (1 - t) * z_0 + t * z_1  # Latent at time t
   ```

4. **Compute target velocity**:
   ```python
   u = z_1 - z_0  # Direction from data to noise
   ```

5. **Predict velocity**:
   ```python
   v = model(t, z_t)
   ```

6. **Loss (MSE)**:
   ```python
   loss = || v - u ||²  # L2 distance in latent space
   ```

7. **Backprop and update**:
   ```python
   loss.backward()
   optimizer.step()
   ```

### Key Hyperparameters
- **Learning rate** (`--lr`): 2e-4 (default for CIFAR-10)
- **Batch size** (`--batch_size`): 128
- **Epochs** (`--num_epoch`): 1000 for CIFAR-10
- **EMA decay** (`--ema_decay`): 0.9999 (exponential moving average of weights)
- **Gradient clipping**: Applied via accelerate framework

### Advantages over Diffusion
1. **Fewer function evaluations**: ~20-50 NFE (vs 50-100+ for DDPM)
2. **Direct objective**: Velocity regression (vs iterative denoising)
3. **Flexible paths**: Can use different probability path schedules
4. **Theoretical guarantees**: Wasserstein-2 bound on reconstruction error

---

## Component 4: Sampling / Inference

### ODE Solver
Given a trained FM model, generate images by solving:
```
dz/dt = v_θ(z_t, t), z(1) ~ N(0, I)
```
from t=1 to t=0 using an ODE solver.

### Solver Options

#### **Adaptive Solvers** (default)
- **dopri5**: Runge-Kutta 4th/5th order, variable step size
  - NFE: ~20-50 (adaptive based on error tolerance)
  - Speed: Fast
  - Accuracy: Good
  - **Best for**: Fast sampling with good quality
- **dopri8**: 7th/8th order
  - NFE: ~30-100
  - Accuracy: Excellent
  - **Best for**: High-quality samples, when speed is not critical

#### **Fixed-Step Solvers** (requires `--use_karras_samplers`)
- **euler**: 1st order, deterministic
  - NFE: = --num_steps (e.g., 50)
  - Speed: Very fast
  - Accuracy: Lower
  - **Best for**: Real-time / interactive applications
- **heun**: 2nd order
  - NFE: = --num_steps
  - Speed: Fast
  - Accuracy: Medium
  - **Best for**: Balance of speed and quality
- **rk4**: 4th order
  - NFE: = --num_steps
  - Speed: Medium
  - Accuracy: Good

### Sampling Code
```python
# 1. Sample noise
z_1 = torch.randn(batch_size, num_channels, 4, 4)

# 2. Solve ODE
from torchdiffeq import odeint_adjoint as odeint
z_0 = odeint(model, z_1, t=[1.0, 0.0], method='dopri5')[-1]

# 3. Decode to image space
images = AE.decode(z_0 / scale_factor)
```

### Classifier-Free Guidance (CFG)
For conditional generation (label-based):
```python
v_cond = model(t, z_t, class_embedding)
v_uncond = model(t, z_t, null_embedding)
v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
```
Increases sample diversity vs. condition adherence as cfg_scale increases.

---

## Component 5: Distributed Training (SLURM)

### Job Architecture
```
6 Independent FM Training Jobs (in parallel):
┌─────────────────┬─────────────────┬─────────────────┐
│ latent_64 FM    │ latent_256 FM   │ latent_1024 FM  │
│ (4 channels)    │ (16 channels)   │ (64 channels)   │
│ 1 GPU / 1000 ep │ 1 GPU / 1000 ep │ 1 GPU / 1000 ep │
└─────────────────┴─────────────────┴─────────────────┘
```

Each job:
1. Loads corresponding ae_<dim>.pt checkpoint
2. Computes scale_factor via `compute_ae_scale_factor.py`
3. Trains FM model via `accelerate launch train_flow_latent.py`
4. Saves checkpoints to `saved_info/latent_flow/cifar10/latent_<dim>/`

### SLURM Configuration
- **Compute**: 1 GPU per job (A100 / H100 recommended)
- **Memory**: 30GB (sufficient for batch_size=128, model_channels=nf)
- **Time**: 2 days (1000 epochs on CIFAR-10 ≈ 40-50 hours)
- **Submission**: `bash slurm/run_train_fm.sh`

---

## File Organization

```
FM_Latent_distillation/
├── train_flow_latent.py          # Main training script (accelerate)
├── test_flow_latent.py           # Sampling + FID evaluation (single GPU)
├── test_flow_latent_ddp.py       # Sampling + FID evaluation (multi-GPU)
├── compute_ae_scale_factor.py    # Calibrate scale_factor per AE checkpoint
│
├── models/
│   ├── conv_autoencoder.py       # Custom ConvAutoencoder class
│   ├── ae_backends.py            # Unified AE interface (SD-VAE vs ConvAE)
│   ├── DiT.py                    # Vision Transformer architecture
│   ├── EDM.py                    # U-Net architectures (ddpm++, ncsn++, adm)
│   └── encoder.py                # Text/condition encoders (unused in FM)
│
├── datasets_prep/
│   ├── __init__.py               # Dataset loaders (get_dataset)
│   ├── lmdb_datasets.py          # LMDB dataset interface
│   └── ... (various dataset-specific modules)
│
├── slurm/
│   ├── config.sh                 # Shared SLURM config (paths, venv)
│   ├── run_train_fm.sh           # Submit 6 FM training jobs
│   └── verify_fm_shapes.py       # Validate model shapes before training
│
├── saved_info/
│   └── latent_flow/
│       └── cifar10/
│           ├── latent_64/
│           │   └── model_<epoch>.pth
│           ├── latent_256/
│           └── ...
│
└── checkpoints/
    ├── ae_64.pt
    ├── ae_128.pt
    └── ... (6 AE checkpoints)
```

---

## Key Design Principles

1. **Latent-space efficiency**: 64× speedup vs. pixel-space FM
2. **Deterministic encoding**: ConvAE has no KL sampling; reproducible latents
3. **Bottleneck scaling**: Model capacity grows with latent dimension
4. **Frozen AE**: Only FM learns; AE weights fixed
5. **Modular backends**: Easy to swap AE types (SD-VAE ↔ ConvAE)
6. **Distributed training**: 6 FM models train independently in parallel
7. **Validation**: `verify_fm_shapes.py` ensures all models instantiate correctly

---

## References

- **Paper**: [Flow Matching in Latent Space](https://arxiv.org/abs/2307.08698)
- **Flow Matching**: [Stable Diffusion meets Flow Matching](https://arxiv.org/abs/2210.02747)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **EDM**: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- **torchdiffeq**: [Neural ODE solvers](https://github.com/rtqichen/torchdiffeq)
