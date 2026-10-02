"""HunyuanVideo adapter exposing the interface expected by its VAE loader."""

from types import SimpleNamespace

import torch.nn as nn

from .taehv import TAEHV


class TAEHVAdapter(nn.Module):
    expects_5d_latents = True

    def __init__(
        self,
        checkpoint_path,
        repo_path=None,
        parallel=False,
        show_progress_bar=False,
        spatial_compression_ratio=8,
        time_compression_ratio=4,
        latent_channels=16,
    ):
        super().__init__()
        del repo_path

        self.model = TAEHV(checkpoint_path=str(checkpoint_path))
        if self.model.latent_channels != latent_channels:
            raise ValueError(
                f"TAEHV latent channels ({self.model.latent_channels}) do not match expected latent channels ({latent_channels})."
            )

        self.parallel = parallel
        self.show_progress_bar = show_progress_bar
        self.use_tiling = False
        self.config = SimpleNamespace(
            block_out_channels=(64, 64, 64, 64),
            scaling_factor=1.0,
            shift_factor=None,
            spatial_compression_ratio=spatial_compression_ratio,
            time_compression_ratio=time_compression_ratio,
            latent_channels=latent_channels,
        )

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        return self

    def disable_tiling(self):
        self.use_tiling = False
        return self

    def decode(self, latents, return_dict=False, generator=None):
        del generator

        squeeze_temporal = False
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
            squeeze_temporal = True
        elif latents.ndim != 5:
            raise ValueError(
                f"Expected latent tensor with shape (b, c, h, w) or (b, c, t, h, w), but got {latents.shape}."
            )

        video_latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        decoded = self.model.decode_video(
            video_latents,
            parallel=self.parallel,
            show_progress_bar=self.show_progress_bar,
        )
        decoded = decoded.mul(2).sub(1)
        decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()

        if squeeze_temporal:
            decoded = decoded.squeeze(2)

        if return_dict:
            return SimpleNamespace(sample=decoded)
        return (decoded,)
