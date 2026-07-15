"""
Unified interface for different autoencoder backends (diffusers SD-VAE vs custom ConvAutoencoder).
Abstracts away API differences so training/sampling scripts don't need to know AE type.
"""

import torch


def load_first_stage_model(args, device, dtype):
    """Load the first-stage (autoencoder) model. Handles both diffusers VAE and custom ConvAE.

    Args:
        args: argparse namespace with ae_type, ae_latent_dim, pretrained_autoencoder_ckpt, etc.
        device: torch device
        dtype: torch dtype (e.g., torch.float32)

    Returns:
        Frozen first-stage model (eval mode, no gradients)
    """
    if args.ae_type == "conv_ae":
        from models.conv_autoencoder import ConvAutoencoder
        model = ConvAutoencoder(latent_dim=args.ae_latent_dim).to(device, dtype=dtype)
        ckpt = torch.load(args.pretrained_autoencoder_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:  # "sd_vae" (default, existing behavior)
        from diffusers.models import AutoencoderKL
        model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_to_latent(model, x, args):
    """Encode images to latent space. Handles both AE types.

    Args:
        model: first-stage model (loaded via load_first_stage_model)
        x: input images
            - For conv_ae: shape (B, 3, 32, 32), range [0, 1]
            - For sd_vae: shape (B, 3, H, W), range [-1, 1]
        args: argparse namespace with ae_type, scale_factor

    Returns:
        z: encoded latents, shape (B, latent_channels, 4, 4), scaled by args.scale_factor
    """
    if args.ae_type == "conv_ae":
        z_flat, _, _ = model.encode(x)
        z = z_flat.view(z_flat.shape[0], model.latent_channels, 4, 4)
        return z.mul_(args.scale_factor)
    else:
        return model.encode(x).latent_dist.sample().mul_(args.scale_factor)


def decode_from_latent(model, z, args):
    """Decode latents back to image space. Handles both AE types.

    Args:
        model: first-stage model (loaded via load_first_stage_model)
        z: latent codes, shape (B, latent_channels, 4, 4)
        args: argparse namespace with ae_type, scale_factor

    Returns:
        images: reconstructed images, range [-1, 1] (unified convention across both AE types)
    """
    z_unscaled = z / args.scale_factor

    if args.ae_type == "conv_ae":
        logits = model.decode(z_unscaled)
        recon = torch.sigmoid(logits)
        return recon * 2 - 1
    else:
        return model.decode(z_unscaled).sample
