# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License. You may obtain
# a copy of the License at:
# https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# The Tencent Hunyuan works and any output/results are provided "AS IS", without
# express or implied warranties. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying, or
# distributing the works or outputs and assume all associated risks.

"""HunyuanVideo 1.5 adapter exposing the Diffusers VAE interface."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn

from .taehv import TAEHV


class _DeterministicLatentDistribution:
    def __init__(self, latents: torch.Tensor):
        self._latents = latents

    def sample(self, generator=None):
        del generator
        return self._latents

    def mode(self):
        return self._latents


def _make_block_out_channels(spatial_compression_ratio: int):
    depth = max(1, spatial_compression_ratio.bit_length())
    return tuple(64 for _ in range(depth))


class TAEHVAdapter(nn.Module):
    expects_5d_latents = True

    def __init__(
        self,
        checkpoint_path,
        parallel=False,
        show_progress_bar=False,
        spatial_compression_ratio=16,
        time_compression_ratio=4,
        latent_channels=32,
    ):
        super().__init__()

        self.model = TAEHV(checkpoint_path=str(checkpoint_path))
        if self.model.latent_channels != latent_channels:
            raise ValueError(
                f"TAEHV latent_channels={self.model.latent_channels} does not match expected {latent_channels}."
            )

        self.parallel = parallel
        self.show_progress_bar = show_progress_bar
        self.use_tiling = False
        self._tile_parallelism_enabled = False
        self.config = SimpleNamespace(
            block_out_channels=_make_block_out_channels(spatial_compression_ratio),
            scaling_factor=1.0,
            shift_factor=None,
            spatial_compression_ratio=spatial_compression_ratio,
            time_compression_ratio=time_compression_ratio,
            ffactor_spatial=spatial_compression_ratio,
            ffactor_temporal=time_compression_ratio,
            latent_channels=latent_channels,
        )

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        del sample_size, tile_overlap_factor
        return self

    def enable_tile_parallelism(self):
        self._tile_parallelism_enabled = True
        return self

    def disable_tile_parallelism(self):
        self._tile_parallelism_enabled = False
        return self

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        return self

    def disable_tiling(self):
        self.use_tiling = False
        return self

    @contextmanager
    def memory_efficient_context(self):
        yield

    def encode(self, pixel_values, return_dict=True):
        squeeze_temporal = False
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)
            squeeze_temporal = True
        elif pixel_values.ndim != 5:
            raise ValueError(
                f"Expected pixel_values with shape (b, c, h, w) or (b, c, t, h, w), got {tuple(pixel_values.shape)}."
            )

        video = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        video = video.add(1.0).mul(0.5).clamp_(0.0, 1.0)
        latents = self.model.encode_video(
            video,
            parallel=self.parallel,
            show_progress_bar=self.show_progress_bar,
        )
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()

        if squeeze_temporal:
            latents = latents.squeeze(2)

        output = SimpleNamespace(latent_dist=_DeterministicLatentDistribution(latents))
        if return_dict:
            return output
        return (output.latent_dist,)

    def decode(self, latents, return_dict=True, generator=None):
        del generator

        squeeze_temporal = False
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
            squeeze_temporal = True
        elif latents.ndim != 5:
            raise ValueError(
                f"Expected latents with shape (b, c, h, w) or (b, c, t, h, w), got {tuple(latents.shape)}."
            )

        video_latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        decoded = self.model.decode_video(
            video_latents,
            parallel=self.parallel,
            show_progress_bar=self.show_progress_bar,
        )
        decoded = decoded.mul(2.0).sub(1.0)
        decoded = decoded.permute(0, 2, 1, 3, 4).contiguous()

        if squeeze_temporal:
            decoded = decoded.squeeze(2)

        if return_dict:
            return SimpleNamespace(sample=decoded)
        return (decoded,)
