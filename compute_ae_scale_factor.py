"""
Compute recommended scale_factor for a custom ConvAutoencoder checkpoint.

Usage:
    python compute_ae_scale_factor.py --ckpt <path_to_ae_<dim>.pt> --latent_dim <dim>

This script loads a ConvAutoencoder checkpoint, encodes a few CIFAR-10 batches
(images in [0,1]), and computes the empirical standard deviation of the latents.
The scale_factor is then set to 1 / std_latents, which normalizes the latent
distribution to have unit variance.

Example output:
    Computed scale_factor for latent_dim=256: 1.234567
    Use --scale_factor 1.234567 in your training command.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from models.conv_autoencoder import ConvAutoencoder


def compute_scale_factor(ckpt_path, latent_dim, num_batches=10):
    """Compute recommended scale_factor for a ConvAutoencoder checkpoint.

    Args:
        ckpt_path: path to ae_<dim>.pt checkpoint
        latent_dim: latent dimension
        num_batches: number of CIFAR-10 batches to encode

    Returns:
        scale_factor: 1 / std_latents
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    latents_list = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            z_flat, _, _ = model.encode(images)
            latents_list.append(z_flat.cpu())

    all_latents = torch.cat(latents_list, dim=0)
    std_latents = all_latents.std()
    scale_factor = 1.0 / std_latents

    print(f"\nLatent statistics:")
    print(f"  Shape: {all_latents.shape}")
    print(f"  Mean: {all_latents.mean():.6f}")
    print(f"  Std: {std_latents:.6f}")
    print(f"\n✓ Recommended scale_factor for latent_dim={latent_dim}: {scale_factor:.6f}")
    print(f"\nUse in your training command:")
    print(f"  --scale_factor {scale_factor:.6f}")

    return scale_factor


def main():
    parser = argparse.ArgumentParser(
        description="Compute recommended scale_factor for ConvAutoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to ae_<dim>.pt checkpoint")
    parser.add_argument("--latent_dim", type=int, required=True, choices=[64, 128, 256, 384, 512, 1024],
                        help="Latent dimension")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of CIFAR-10 batches to encode (default: 10)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    compute_scale_factor(ckpt_path, args.latent_dim, args.num_batches)


if __name__ == "__main__":
    main()
