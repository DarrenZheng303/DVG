"""Shared TAEHV implementation and checkpoint registry."""

from pathlib import Path

from .taehv import StreamingTAEHV, TAEHV

UPSTREAM_COMMIT = "011dfc2112197741c540e0bdd5b7b67bcc930771"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Wan2.2 A14B uses the Wan2.1 VAE, so it must use taew2_1 weights.
CHECKPOINTS = {
    "hunyuan_video": "taehv.pth",
    "hunyuan_video_1_5": "taehv1_5.pth",
    "wan2_2_14b": "taew2_1.pth",
}


def checkpoint_path(backbone: str) -> Path:
    try:
        filename = CHECKPOINTS[backbone]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported TAE backbone {backbone!r}; expected one of {sorted(CHECKPOINTS)}"
        ) from exc

    path = WEIGHTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"TAE checkpoint for {backbone} was not found: {path}")
    return path


__all__ = ["CHECKPOINTS", "StreamingTAEHV", "TAEHV", "UPSTREAM_COMMIT", "checkpoint_path"]
