# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from . import hunyuanvideo_15_vae
from dvg.taehv.hyvideo1_5_adapter import TAEHVAdapter
from dvg.taehv import checkpoint_path

VAE_METADATA: Dict[str, Dict[str, Any]] = {
    "hunyuan": {
        "backend": "hunyuan",
    },
    "taehv1_5": {
        "backend": "taehv",
        "default_filename": "taehv1_5.pth",
        "latent_channels": 32,
        "spatial_compression_ratio": 16,
        "time_compression_ratio": 4,
    },
}

VAE_TYPE_ALIASES = {
    "auto": "auto",
    "hy": "hunyuan",
    "hunyuan": "hunyuan",
    "taehv1.5": "taehv1_5",
    "taehv1_5": "taehv1_5",
}


def _normalize_vae_type(vae_type: str, vae_path: Optional[str]) -> str:
    vae_type = str(vae_type or "auto").strip().lower()
    if vae_type not in VAE_TYPE_ALIASES:
        raise ValueError(
            f"Unsupported vae_type={vae_type!r}. "
            f"Supported values: {sorted(VAE_TYPE_ALIASES)}"
        )

    normalized = VAE_TYPE_ALIASES[vae_type]
    if normalized != "auto":
        return normalized

    if vae_path is None:
        return "hunyuan"

    vae_path_obj = Path(vae_path)
    if vae_path_obj.is_dir():
        return "hunyuan"

    path_name = vae_path_obj.name.lower()
    if "taehv1_5" in path_name:
        return "taehv1_5"
    if "taehv" in path_name:
        return "taehv1_5"

    raise ValueError(
        "Could not infer VAE type from vae_path. "
        "Please set --vae_type explicitly."
    )


def _load_hunyuan_vae(
    vae_path: str,
    torch_dtype: torch.dtype,
    sample_size: Optional[int] = None,
    tile_overlap_factor: Optional[float] = None,
):
    vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    if sample_size is not None and hasattr(vae, "set_tile_sample_min_size"):
        overlap = 0.2 if tile_overlap_factor is None else float(tile_overlap_factor)
        vae.set_tile_sample_min_size(int(sample_size), overlap)
    return vae


def _load_taehv_vae(
    vae_path: str,
    vae_meta: Dict[str, Any],
):
    return TAEHVAdapter(
        checkpoint_path=vae_path,
        latent_channels=int(vae_meta["latent_channels"]),
        spatial_compression_ratio=int(vae_meta["spatial_compression_ratio"]),
        time_compression_ratio=int(vae_meta["time_compression_ratio"]),
    )


def load_vae(
    *,
    pretrained_model_root: Optional[str] = None,
    vae_type: str = "auto",
    vae_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    sample_size: Optional[int] = None,
    tile_overlap_factor: Optional[float] = None,
    device=None,
    logger=None,
):
    normalized_vae_type = _normalize_vae_type(vae_type, vae_path)
    vae_meta = VAE_METADATA[normalized_vae_type]

    if vae_path is None:
        if normalized_vae_type == "hunyuan":
            if pretrained_model_root is None:
                raise ValueError("pretrained_model_root is required when loading the default Hunyuan VAE.")
            vae_path = str(Path(pretrained_model_root).expanduser() / "vae")
        else:
            vae_path = str(checkpoint_path("hunyuan_video_1_5"))

    vae_path = str(Path(vae_path).expanduser())
    if logger is not None:
        logger.info(f"Loading VAE ({normalized_vae_type}) from: {vae_path}")

    if vae_meta["backend"] == "taehv":
        vae = _load_taehv_vae(vae_path, vae_meta)
    else:
        vae = _load_hunyuan_vae(
            vae_path,
            torch_dtype=torch_dtype,
            sample_size=sample_size,
            tile_overlap_factor=tile_overlap_factor,
        )

    if torch_dtype is not None:
        vae = vae.to(dtype=torch_dtype)

    if device is not None:
        vae = vae.to(device)

    vae.requires_grad_(False)
    vae.eval()

    if logger is not None:
        try:
            vae_dtype = vae.dtype
        except AttributeError:
            vae_dtype = next(vae.parameters()).dtype
        try:
            vae_device = vae.device
        except AttributeError:
            vae_device = next(vae.parameters()).device
        logger.info(f"Loaded VAE type={normalized_vae_type} dtype={vae_dtype} device={vae_device}")

    return vae


__all__ = [
    "VAE_METADATA",
    "VAE_TYPE_ALIASES",
    "load_vae",
]
