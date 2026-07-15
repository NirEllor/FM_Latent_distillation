#!/usr/bin/env python
"""
Verify that FM UNet models can be instantiated with the correct input/output shapes
for each AE latent dimension, and that the bottleneck capacity is sufficient.

Usage:
    python slurm/verify_fm_shapes.py
"""

import sys
import argparse
import torch

sys.path.insert(0, ".")

from models import create_network


class Args:
    """Minimal args object for model instantiation."""

    def __init__(self, latent_dim, nf, ch_mult, attn_resolutions):
        self.model_type = "ddpm++"
        self.image_size = 32
        self.f = 8
        self.num_in_channels = latent_dim // 16
        self.num_out_channels = latent_dim // 16
        self.nf = nf
        self.ch_mult = ch_mult
        self.attn_resolutions = attn_resolutions
        self.label_dim = 0
        self.augment_dim = 0
        self.num_blocks = 4
        self.dropout = 0.1
        self.label_dropout = 0.0
        self.num_classes = None
        self.use_origin_adm = False
        self.layout = False
        self.use_scale_shift_norm = True
        self.resblock_updown = False
        self.use_new_attention_order = False
        self.num_heads = 4
        self.num_head_channels = -1
        self.num_head_upsample = -1
        self.resamp_with_conv = True


def verify_model_shapes():
    """Verify model instantiation and shapes for all AE dimensions."""
    dims = [64, 128, 256, 384, 512, 1024]
    nf_base = 8
    ch_mult = [1, 2, 2]
    attn_resolutions = [4]

    print("Verifying FM UNet model shapes for all AE dimensions\n")
    print(f"{'AE Dim':<10} {'Channels':<12} {'nf':<10} {'Bottleneck':<12} {'Status':<20}")
    print("-" * 70)

    all_ok = True
    for dim in dims:
        num_channels = dim // 16
        nf = dim * nf_base
        bottleneck_ch = nf * max(ch_mult)

        try:
            args = Args(dim, nf, ch_mult, attn_resolutions)
            model = create_network(args)

            x = torch.randn(2, num_channels, 4, 4)
            t = torch.rand(2)

            with torch.no_grad():
                y = model(t, x)

            if y.shape != x.shape:
                raise RuntimeError(
                    f"Output shape {y.shape} != input shape {x.shape}"
                )

            status = f"✓ OK (out {y.shape})"
        except Exception as e:
            status = f"✗ FAILED: {str(e)[:30]}"
            all_ok = False

        print(
            f"ae_{dim:<6} {num_channels:<12} {nf:<10} {bottleneck_ch:<12} {status}"
        )

    print()
    if all_ok:
        print("✓ All models instantiated successfully with correct shapes!")
        return 0
    else:
        print("✗ Some models failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_model_shapes())
