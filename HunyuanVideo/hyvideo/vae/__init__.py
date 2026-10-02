from pathlib import Path

import torch

from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from dvg.taehv.hyvideo_adapter import TAEHVAdapter
from ..constants import VAE_METADATA, VAE_PATH, PRECISION_TO_TYPE


def _load_hunyuan_vae(vae_path, sample_size=None):
    config = AutoencoderKLCausal3D.load_config(vae_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)

    vae_ckpt = Path(vae_path) / "pytorch_model.pt"
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

    ckpt = torch.load(vae_ckpt, map_location=vae.device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    vae.load_state_dict(ckpt)
    return vae


def _load_taehv_vae(vae_path, vae_meta):
    return TAEHVAdapter(
        checkpoint_path=vae_path,
        latent_channels=vae_meta["latent_channels"],
        spatial_compression_ratio=vae_meta["spatial_compression_ratio"],
        time_compression_ratio=vae_meta["time_compression_ratio"],
    )


def load_vae(vae_type: str="884-16c-hy",
             vae_precision: str=None,
             sample_size: tuple=None,
             vae_path: str=None,
             logger=None,
             device=None
             ):
    """the fucntion to load the 3D VAE model

    Args:
        vae_type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        vae_precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        vae_path (str, optional): the path to vae. Defaults to None.
        logger (_type_, optional): logger. Defaults to None.
        device (_type_, optional): device to load vae. Defaults to None.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    vae_meta = VAE_METADATA[vae_type]
    
    if logger is not None:
        logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")

    if vae_meta["backend"] == "taehv":
        vae = _load_taehv_vae(vae_path, vae_meta)
    else:
        vae = _load_hunyuan_vae(vae_path, sample_size=sample_size)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio
    
    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
