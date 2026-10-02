"""Wan 2.1-VAE-compatible TAE decode adapter (used by Wan2.2 A14B)."""

import logging

import torch

from . import TAEHV, checkpoint_path


def load_wan_decoder(device, dtype):
    path = checkpoint_path("wan2_2_14b")
    logging.info("Loading Wan TAE decoder from %s", path)
    return TAEHV(checkpoint_path=str(path)).to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def decode_wan_latents(tae, latents):
    """Decode Wan [C,T,H,W] latent items to Wan-style [-1,1] [C,T,H,W] videos."""
    videos = []
    for latent in latents:
        tae_input = latent.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        tae_input = tae_input.to(device=next(tae.parameters()).device,
                                 dtype=next(tae.parameters()).dtype)
        decoded = tae.decode_video(tae_input, parallel=False, show_progress_bar=False)
        videos.append(decoded[0].permute(1, 0, 2, 3).mul(2).sub(1).contiguous())
    return videos
