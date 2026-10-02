"""Backbone-independent DVG shape, noise, density and action algorithms."""

from __future__ import annotations

import hashlib
import math
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def rank0_print(*args, **kwargs):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def build_joint_stage_shapes(
    latent_time: int,
    latent_height: int,
    latent_width: int,
    spatial_rate_list: Sequence[float],
    temporal_rate_list: Sequence[float],
) -> List[List[int]]:
    if not spatial_rate_list:
        spatial_rate_list = [1.0]
    if not temporal_rate_list:
        temporal_rate_list = [1.0]
    if len(spatial_rate_list) != len(temporal_rate_list):
        raise ValueError(
            "Joint progressive mode requires spatial_rate_list and temporal_rate_list to have the same length, "
            f"got {len(spatial_rate_list)} and {len(temporal_rate_list)}."
        )

    stage_shapes: List[List[int]] = []
    prev_t = 1
    prev_h = 1
    prev_w = 1

    for idx, (spatial_rate, temporal_rate) in enumerate(zip(spatial_rate_list, temporal_rate_list)):
        if idx == len(spatial_rate_list) - 1:
            stage_t = latent_time
            stage_h = latent_height
            stage_w = latent_width
        else:
            stage_t = max(1, min(latent_time, int(round(latent_time * float(temporal_rate)))))
            stage_h = max(1, min(latent_height, int(round(latent_height * float(spatial_rate)))))
            stage_w = max(1, min(latent_width, int(round(latent_width * float(spatial_rate)))))
            stage_t = max(prev_t, stage_t)
            stage_h = max(prev_h, stage_h)
            stage_w = max(prev_w, stage_w)

        stage_shapes.append([stage_t, stage_h, stage_w])
        prev_t = stage_t
        prev_h = stage_h
        prev_w = stage_w

    return stage_shapes


def smooth_stage2_transition(
    latents: torch.Tensor,
    blend_weight: float = 0.3,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Apply per-frame spatial smoothing to latents at the stage 1->2 transition.

    Each frame is blended with its own avg_pool2d-smoothed version to reduce
    aliasing artifacts caused by spatial anchor restoration on the nested grid.
    The blend_weight controls how much of the smoothed version is mixed in
    (0.0 = no smoothing, 1.0 = fully smoothed).
    """
    batch, channels, target_t, target_h, target_w = latents.shape
    flattened = latents.permute(0, 2, 1, 3, 4).reshape(
        batch * target_t, channels, target_h, target_w
    )
    smoothed = F.avg_pool2d(flattened, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    blended = (1.0 - blend_weight) * flattened + blend_weight * smoothed
    return blended.reshape(batch, target_t, channels, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def validate_config(
    enable_dvg: bool,
    stage_shapes: Sequence[Sequence[int]],
    step_rate_list: Sequence[float],
    scheduler_shift_list: Sequence[float],
) -> None:
    if not stage_shapes:
        raise ValueError("stage_shapes must not be empty.")
    if len(step_rate_list) != len(stage_shapes):
        raise ValueError(
            f"step_rate_list length must match stage count, got {len(step_rate_list)} and {len(stage_shapes)}."
        )
    if any(float(rate) < 0.0 or float(rate) > 1.0 for rate in step_rate_list):
        raise ValueError(f"step_rate_list values must be in [0, 1], got {list(step_rate_list)}.")
    if any(float(step_rate_list[idx]) > float(step_rate_list[idx + 1]) for idx in range(len(step_rate_list) - 1)):
        raise ValueError(f"step_rate_list must be non-decreasing, got {list(step_rate_list)}.")
    if abs(float(step_rate_list[-1]) - 1.0) > 1e-8:
        raise ValueError(f"step_rate_list[-1] must be 1.0, got {step_rate_list[-1]}.")

    prev_shape = None
    for idx, stage_shape in enumerate(stage_shapes):
        if len(stage_shape) != 3:
            raise ValueError(f"Stage {idx} must have 3 dims [T, H, W], got {stage_shape}.")
        if any(int(dim) < 1 for dim in stage_shape):
            raise ValueError(f"Stage {idx} has invalid shape {stage_shape}.")
        if prev_shape is not None:
            for axis_name, prev_dim, stage_dim in zip(("T", "H", "W"), prev_shape, stage_shape):
                if int(stage_dim) < int(prev_dim):
                    raise ValueError(
                        f"Stage {idx} dimension {axis_name}={stage_dim} must be non-decreasing vs previous {axis_name}={prev_dim}."
                    )
        prev_shape = stage_shape

    if not enable_dvg and len(stage_shapes) != 1:
        raise ValueError(f"Non-progressive mode expects exactly 1 stage, got {len(stage_shapes)}.")

    if enable_dvg:
        if len(stage_shapes) != 4:
            raise ValueError(
                "Stage-transition action selection currently expects exactly 4 progressive stages."
            )
        if len(step_rate_list) != 4:
            raise ValueError(
                "Stage-transition action selection currently expects a 4-stage step_rate_list."
            )

    if enable_dvg and len(scheduler_shift_list) != len(stage_shapes):
        raise ValueError(
            f"scheduler_shift_list length must match stage count, got {len(scheduler_shift_list)} and {len(stage_shapes)}."
        )


def compute_runtime_stage_step_counts(
    infer_step_count: int,
    step_rate_list: Sequence[float],
) -> List[int]:
    if infer_step_count <= 0:
        raise ValueError(f"infer_step_count must be positive, got {infer_step_count}.")

    time_step_split = [int(infer_step_count * float(step_rate)) for step_rate in step_rate_list]
    stage_step_counts = [0 for _ in step_rate_list]
    stage_idx = 0
    for step_idx in range(infer_step_count):
        if step_idx in time_step_split[:-1] and stage_idx < len(step_rate_list) - 1:
            stage_idx += 1
        stage_step_counts[stage_idx] += 1
    return stage_step_counts


def compute_density(
    stage_shapes: Sequence[Sequence[int]],
    full_shape: Sequence[int],
    stage_step_counts: Sequence[int],
) -> float:
    if len(stage_shapes) != len(stage_step_counts):
        raise ValueError(
            f"stage_shapes and stage_step_counts must have the same length, "
            f"got {len(stage_shapes)} and {len(stage_step_counts)}."
        )

    full_token_count = math.prod(map(int, full_shape))
    total_steps = sum(map(int, stage_step_counts))
    if full_token_count <= 0 or total_steps <= 0:
        raise ValueError("full_shape and stage_step_counts must describe positive totals.")

    weighted_token_count = sum(
        math.prod(map(int, shape)) * int(step_count)
        for shape, step_count in zip(stage_shapes, stage_step_counts)
    )
    return weighted_token_count / (full_token_count * total_steps)


def _select_nested_subset_indices(superset_indices: Sequence[int], target_size: int) -> List[int]:
    if target_size < 1:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    if target_size > len(superset_indices):
        raise ValueError(
            f"Cannot select {target_size} nested coordinates from superset of size {len(superset_indices)}."
        )
    if target_size == len(superset_indices):
        return list(superset_indices)
    if target_size == 1:
        return [int(superset_indices[0])]

    max_pos = len(superset_indices) - 1
    subset_positions = [int(round(i * max_pos / (target_size - 1))) for i in range(target_size)]
    subset_positions[0] = 0
    subset_positions[-1] = max_pos

    for idx in range(1, target_size):
        min_allowed = subset_positions[idx - 1] + 1
        remaining = target_size - idx - 1
        max_allowed = max_pos - remaining
        subset_positions[idx] = min(max(subset_positions[idx], min_allowed), max_allowed)

    return [int(superset_indices[pos]) for pos in subset_positions]


def build_nested_axis_indices(final_size: int, stage_sizes: Sequence[int]) -> List[List[int]]:
    if final_size < 1:
        raise ValueError(f"final_size must be positive, got {final_size}.")
    if not stage_sizes:
        return []
    if int(stage_sizes[-1]) != final_size:
        raise ValueError(
            f"The last stage size must equal final_size, got last={stage_sizes[-1]}, final={final_size}."
        )

    nested_axes: List[List[int]] = [None] * len(stage_sizes)  # type: ignore[assignment]
    nested_axes[-1] = list(range(final_size))
    for idx in range(len(stage_sizes) - 2, -1, -1):
        nested_axes[idx] = _select_nested_subset_indices(nested_axes[idx + 1], int(stage_sizes[idx]))
    return nested_axes


def build_stage_coordinate_axes(
    stage_shapes: Sequence[Sequence[int]],
) -> List[Tuple[List[int], List[int], List[int]]]:
    if not stage_shapes:
        return []

    final_t, final_h, final_w = map(int, stage_shapes[-1])
    t_axes = build_nested_axis_indices(final_t, [int(shape[0]) for shape in stage_shapes])
    h_axes = build_nested_axis_indices(final_h, [int(shape[1]) for shape in stage_shapes])
    w_axes = build_nested_axis_indices(final_w, [int(shape[2]) for shape in stage_shapes])
    return [(t_axes[idx], h_axes[idx], w_axes[idx]) for idx in range(len(stage_shapes))]


def resize_latents(
    latents: torch.Tensor,
    target_shape: Sequence[int],
    mode: str = "trilinear",
    current_stage_idx: int = 0,
    next_stage_idx: int = 1,
    stage_coordinate_axes: Sequence[Sequence[Sequence[int]]] | None = None,
) -> torch.Tensor:
    target_t = int(target_shape[0])
    target_h = int(target_shape[1])
    target_w = int(target_shape[2])

    _, _, source_t, source_h, source_w = latents.shape

    # Compute anchor indices from coordinate axes
    current_axis_t, current_axis_h, current_axis_w = stage_coordinate_axes[current_stage_idx]
    next_axis_t, next_axis_h, next_axis_w = stage_coordinate_axes[next_stage_idx]

    def _map_anchors(current, next_):
        lookup = {c: i for i, c in enumerate(next_)}
        return [lookup[c] for c in current]

    target_time_anchor_indices = _map_anchors(current_axis_t, next_axis_t)
    target_height_anchor_indices = _map_anchors(current_axis_h, next_axis_h)
    target_width_anchor_indices = _map_anchors(current_axis_w, next_axis_w)

    def _validate(indices: Sequence[int] | None, source_size: int, target_size: int, name: str) -> None:
        if indices is None:
            raise ValueError(f"target_{name}_anchor_indices must not be None.")
        anchors = [int(idx) for idx in indices]
        if len(anchors) != source_size:
            raise ValueError(
                f"target_{name}_anchor_indices length must match source, "
                f"got {len(anchors)} vs {source_size}."
            )
        if any(idx < 0 or idx >= target_size for idx in anchors):
            raise ValueError(
                f"target_{name}_anchor_indices must be within [0, {target_size - 1}], got {anchors}."
            )
        if any(anchors[i] >= anchors[i + 1] for i in range(source_size - 1)):
            raise ValueError(
                f"target_{name}_anchor_indices must be strictly increasing, got {anchors}."
            )

    _validate(target_time_anchor_indices, source_t, target_t, "time")
    _validate(target_height_anchor_indices, source_h, target_h, "height")
    _validate(target_width_anchor_indices, source_w, target_w, "width")

    def _resize_spatial_per_frame(spatial_mode: str) -> torch.Tensor:
        batch, channels, source_t, source_h, source_w = latents.shape
        interpolate_kwargs = {
            "size": (target_h, target_w),
            "mode": spatial_mode,
        }
        if spatial_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            interpolate_kwargs["align_corners"] = False

        spatial_resized = F.interpolate(
            latents.permute(0, 2, 1, 3, 4).reshape(batch * source_t, channels, source_h, source_w),
            **interpolate_kwargs,
        )
        return spatial_resized.reshape(batch, source_t, channels, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()

    def _resolve_spatial_anchor_grids(
        source_h: int,
        source_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height_anchor_indices = [int(idx) for idx in target_height_anchor_indices]
        width_anchor_indices = [int(idx) for idx in target_width_anchor_indices]

        h_idx = torch.tensor(height_anchor_indices, device=device, dtype=torch.long)
        w_idx = torch.tensor(width_anchor_indices, device=device, dtype=torch.long)
        return h_idx[:, None].expand(source_h, source_w), w_idx[None, :].expand(source_h, source_w)

    def _restore_spatial(
        spatial_resized: torch.Tensor,
        source_latents: torch.Tensor,
        source_h: int,
        source_w: int,
    ) -> torch.Tensor:
        h_grid, w_grid = _resolve_spatial_anchor_grids(source_h, source_w, spatial_resized.device)
        spatial_resized[:, :, :, h_grid, w_grid] = source_latents
        return spatial_resized

    def _interp_dim(
        values: torch.Tensor,
        dim: int,
        target_size: int,
        anchor_indices: Sequence[int],
    ) -> torch.Tensor:
        source_size = values.shape[dim]
        if source_size == 1:
            return values.expand(*values.shape[:dim], target_size, *values.shape[dim + 1 :])

        anchor_tensor = torch.tensor(anchor_indices, device=values.device, dtype=torch.long)
        target_positions = torch.arange(target_size, device=values.device, dtype=torch.long)

        left_indices = torch.searchsorted(anchor_tensor, target_positions, right=True) - 1
        left_indices = left_indices.clamp_(0, source_size - 2)
        right_indices = left_indices + 1

        left_anchor_positions = anchor_tensor[left_indices]
        right_anchor_positions = anchor_tensor[right_indices]
        denom = (right_anchor_positions - left_anchor_positions).clamp_min_(1)

        weight = (target_positions - left_anchor_positions).to(torch.float32) / denom.to(torch.float32)
        weight = torch.where(target_positions <= anchor_tensor[0], torch.zeros_like(weight), weight)
        weight = torch.where(target_positions >= anchor_tensor[-1], torch.ones_like(weight), weight)
        weight = weight.to(device=values.device, dtype=values.dtype)

        moved = values.movedim(dim, -1)
        flat = moved.reshape(-1, source_size)
        expand_shape = (flat.shape[0], target_size)
        left_vals = torch.gather(flat, 1, left_indices.view(1, target_size).expand(expand_shape))
        right_vals = torch.gather(flat, 1, right_indices.view(1, target_size).expand(expand_shape))

        blended = torch.lerp(left_vals, right_vals, weight.view(1, target_size))
        return blended.reshape(*moved.shape[:-1], target_size).movedim(-1, dim)

    def _interp_spatial(source_latents: torch.Tensor) -> torch.Tensor:
        height_anchor_indices = [int(idx) for idx in target_height_anchor_indices]
        width_anchor_indices = [int(idx) for idx in target_width_anchor_indices]

        height_resized = _interp_dim(
            source_latents,
            dim=3,
            target_size=target_h,
            anchor_indices=height_anchor_indices,
        )
        return _interp_dim(
            height_resized,
            dim=4,
            target_size=target_w,
            anchor_indices=width_anchor_indices,
        )

    def _interp_temporal(source_frames: torch.Tensor, source_t: int) -> torch.Tensor:
        if source_t == 1:
            return source_frames.expand(-1, -1, target_t, -1, -1)

        return _interp_dim(
            source_frames,
            dim=2,
            target_size=target_t,
            anchor_indices=target_time_anchor_indices,
        )

    if mode == "temporal_interp_spatial_restore":
        # Keep source frames on their spatial anchors, then interpolate missing time points.
        spatial_resized = _resize_spatial_per_frame("area")
        spatial_resized = _restore_spatial(spatial_resized, latents, source_h, source_w)
        result = _interp_temporal(spatial_resized, source_t)

    elif mode == "temporal_interp_spatial_area":
        spatial_resized = _resize_spatial_per_frame("area")
        result = _interp_temporal(spatial_resized, source_t)

    elif mode == "temporal_interp_spatial_interp":
        spatial_resized = _interp_spatial(latents)
        result = _interp_temporal(spatial_resized, source_t)

    else:
        result = F.interpolate(
            latents,
            size=(target_t, target_h, target_w),
            mode=mode,
            align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        )

    return result.contiguous()


def _make_cache_path(
    cache_dir: str,
    stage_idx: int,
    axis_signature: str,
    channels: int,
    seed: int,
) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = (
        f"coord_noise_v2_stage{stage_idx}"
        f"_sig{axis_signature}"
        f"_c{channels}_seed{seed}.pt"
    )
    return cache_path / filename


def load_coordinate_noise_cache(
    cache_dir: str,
    stage_idx: int,
    current_size: tuple[int, int, int],
    axis_signature: str,
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    cache_file = _make_cache_path(cache_dir, stage_idx, axis_signature, channels, seed)
    if not cache_file.exists():
        return None

    cache_data = torch.load(cache_file, map_location=device, weights_only=False)
    if isinstance(cache_data, dict):
        noise = cache_data["noise"]
        saved_shape = tuple(cache_data["shape"])
        expected_shape = (1, channels, current_size[0], current_size[1], current_size[2])
        if saved_shape != expected_shape:
            return None
    else:
        noise = cache_data

    return noise.to(device=device, dtype=dtype)


def save_coordinate_noise_cache(
    noise: torch.Tensor,
    cache_dir: str,
    stage_idx: int,
    axis_signature: str,
    channels: int,
    seed: int,
) -> None:
    cache_file = _make_cache_path(cache_dir, stage_idx, axis_signature, channels, seed)
    cache_data = {
        "noise": noise.detach().cpu(),
        "dtype": str(noise.dtype),
        "shape": tuple(noise.shape),
    }
    torch.save(cache_data, cache_file)


def _hash_mix(x: torch.Tensor) -> torch.Tensor:
    # SplitMix64-style integer mixing for deterministic coordinate hashing.
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x


def _make_axis_signature(
    t_coords: tuple[int, ...],
    h_coords: tuple[int, ...],
    w_coords: tuple[int, ...],
) -> str:
    payload = f"t:{','.join(map(str, t_coords))}|h:{','.join(map(str, h_coords))}|w:{','.join(map(str, w_coords))}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def generate_coordinate_noise(
    axis_t: tuple[int, ...],
    axis_h: tuple[int, ...],
    axis_w: tuple[int, ...],
    channels: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate deterministic stage-transition noise from stable canonical ids.
    """
    t_coords = torch.tensor(axis_t, device=device, dtype=torch.int64)
    h_coords = torch.tensor(axis_h, device=device, dtype=torch.int64)
    w_coords = torch.tensor(axis_w, device=device, dtype=torch.int64)
    c_coords = torch.arange(channels, device=device, dtype=torch.int64)

    tt, hh, ww = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    tt = tt.unsqueeze(0)
    hh = hh.unsqueeze(0)
    ww = ww.unsqueeze(0)
    cc = c_coords.view(channels, 1, 1, 1)

    seed_tensor = torch.full((1,), int(seed), device=device, dtype=torch.int64)

    base = (
        tt * 0x9E3779B185EBCA87
        + hh * 0xC2B2AE3D27D4EB4F
        + ww * 0x165667B19E3779F9
        + cc * 0x85EBCA77C2B2AE63
        + seed_tensor.view(1, 1, 1, 1) * 0x27D4EB2F165667C5
    )

    h1 = _hash_mix(base)
    h2 = _hash_mix(base + 0x9E3779B97F4A7C15)

    denom = float(1 << 53)
    u1 = (((h1 & ((1 << 53) - 1)).to(torch.float64)) + 0.5) / denom
    u2 = (((h2 & ((1 << 53) - 1)).to(torch.float64)) + 0.5) / denom

    eps = 1e-12
    u1 = u1.clamp(min=eps, max=1.0 - eps)
    u2 = u2.clamp(min=eps, max=1.0 - eps)

    noise = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * u2)
    return noise.unsqueeze(0).to(dtype=dtype)


def get_transition_noise(
    batch_size: int,
    channels: int,
    axis_t: tuple[int, ...],
    axis_h: tuple[int, ...],
    axis_w: tuple[int, ...],
    stage_idx: int,
    seed: int,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    current_size = (len(axis_t), len(axis_h), len(axis_w))
    axis_signature = _make_axis_signature(axis_t, axis_h, axis_w)
    shared_noise = load_coordinate_noise_cache(
        cache_dir=cache_dir,
        stage_idx=stage_idx,
        current_size=current_size,
        axis_signature=axis_signature,
        channels=channels,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    if shared_noise is None:
        shared_noise = generate_coordinate_noise(
            axis_t=axis_t,
            axis_h=axis_h,
            axis_w=axis_w,
            channels=channels,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        save_coordinate_noise_cache(
            noise=shared_noise,
            cache_dir=cache_dir,
            stage_idx=stage_idx,
            axis_signature=axis_signature,
            channels=channels,
            seed=seed,
        )

    if batch_size == 1:
        return shared_noise
    return shared_noise.expand(batch_size, -1, -1, -1, -1).contiguous()


def _sobel_kernels(device, dtype):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    return kernel_x, kernel_y


def _project_latents_to_scalar_frames(x0_pred: torch.Tensor) -> torch.Tensor:
    frames = x0_pred.to(torch.float32).mean(dim=1)
    mean = frames.mean(dim=(2, 3), keepdim=True)
    std = frames.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (frames - mean) / std


class LatentMetricCalculator:
    """Compute only the requested latent metrics and cache shared intermediates."""

    TEMPORAL_METRICS = {
        "flow_mag": {"method": "_compute_temporal_flow_magnitude", "needs_norm": True},
        "temporal_grad": {"method": "_compute_temporal_gradient", "needs_norm": True},
        "frame_mse": {"method": "_compute_temporal_frame_mse", "needs_norm": True},
        "frame_lpips": {"method": "_compute_temporal_lpips_proxy", "needs_norm": True},
        "temporal_fft_high_freq": {"method": "_compute_temporal_fft_high_frequency", "needs_norm": False},
    }
    SPATIAL_METRICS = {
        "spatial_edge": {"method": "_compute_spatial_edge", "needs_norm": True},
        "spatial_fft_energy_above_50": {"method": "_compute_spatial_fft_energy_above_50", "needs_norm": False},
        "multi_scale_high_freq": {"method": "_compute_spatial_multi_scale_high_frequency", "needs_norm": True},
        "hf_scale3": {"method": "_compute_spatial_high_frequency_scale3", "needs_norm": True},
        "hf_scale5": {"method": "_compute_spatial_high_frequency_scale5", "needs_norm": True},
        "hf_scale7": {"method": "_compute_spatial_high_frequency_scale7", "needs_norm": True},
        "high_freq": {"method": "_compute_spatial_high_frequency", "needs_norm": True},
    }

    def __init__(self, x0_pred: torch.Tensor):
        if x0_pred.ndim != 5:
            raise ValueError(
                f"Expected dense latent x0 with shape [B, C, T, H, W], got {tuple(x0_pred.shape)}"
            )
        if x0_pred.shape[2] < 2:
            raise ValueError(
                f"Expected at least 2 frames for temporal analysis, got T={x0_pred.shape[2]}"
            )
        self.x0 = x0_pred.to(torch.float32)
        self._metric_cache: Dict[Tuple[str, str], float] = {}

    @classmethod
    def validate_metrics(cls, temporal_metric: str, spatial_metric: str) -> None:
        cls._validate_metric(temporal_metric, "temporal", cls.TEMPORAL_METRICS)
        cls._validate_metric(spatial_metric, "spatial", cls.SPATIAL_METRICS)

    @staticmethod
    def _validate_metric(metric: str, group: str, registry: Dict[str, Dict[str, Any]]) -> None:
        if metric not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unsupported {group} metric '{metric}'. Available metrics: {available}"
            )

    def _compute_metric(self, group: str, metric: str, registry: Dict[str, Dict[str, Any]]) -> float:
        self._validate_metric(metric, group, registry)
        cache_key = (group, metric)
        if cache_key not in self._metric_cache:
            self._metric_cache[cache_key] = float(getattr(self, registry[metric]["method"])())
        return self._metric_cache[cache_key]

    @staticmethod
    def _normalize_metric(raw: float, metric: str, registry: Dict[str, Dict[str, Any]]) -> float:
        return _bounded_norm(raw) if registry[metric]["needs_norm"] else raw

    def compute_temporal(self, metric: str) -> float:
        return self._compute_metric("temporal", metric, self.TEMPORAL_METRICS)

    def compute_spatial(self, metric: str) -> float:
        return self._compute_metric("spatial", metric, self.SPATIAL_METRICS)

    def normalize_temporal(self, metric: str, raw: float) -> float:
        return self._normalize_metric(raw, metric, self.TEMPORAL_METRICS)

    def normalize_spatial(self, metric: str, raw: float) -> float:
        return self._normalize_metric(raw, metric, self.SPATIAL_METRICS)

    @cached_property
    def scalar_frames(self) -> torch.Tensor:
        return _project_latents_to_scalar_frames(self.x0)

    @cached_property
    def frame_differences(self) -> torch.Tensor:
        return self.scalar_frames[:, 1:] - self.scalar_frames[:, :-1]

    @cached_property
    def flat_scalar_frames(self) -> torch.Tensor:
        frames = self.scalar_frames
        return frames.reshape(-1, 1, frames.shape[2], frames.shape[3])

    @cached_property
    def flat_latents(self) -> torch.Tensor:
        return self.x0.permute(0, 2, 1, 3, 4).reshape(
            -1, self.x0.shape[1], self.x0.shape[3], self.x0.shape[4]
        )

    @cached_property
    def sobel_kernels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return _sobel_kernels(self.scalar_frames.device, self.scalar_frames.dtype)

    def _compute_temporal_gradient(self) -> float:
        return float(self.frame_differences.abs().mean().item())

    def _compute_temporal_frame_mse(self) -> float:
        return float(self.frame_differences.pow(2).mean().item())

    def _compute_temporal_flow_magnitude(self) -> float:
        frames = self.scalar_frames
        pair_frames = frames[:, :-1].reshape(-1, 1, frames.shape[2], frames.shape[3])
        temporal_diff = self.frame_differences.reshape_as(pair_frames)
        kernel_x, kernel_y = self.sobel_kernels
        grad_x = F.conv2d(pair_frames, kernel_x, padding=1)
        grad_y = F.conv2d(pair_frames, kernel_y, padding=1)

        a11 = (grad_x * grad_x).sum(dim=(1, 2, 3))
        a22 = (grad_y * grad_y).sum(dim=(1, 2, 3))
        a12 = (grad_x * grad_y).sum(dim=(1, 2, 3))
        b1 = -(grad_x * temporal_diff).sum(dim=(1, 2, 3))
        b2 = -(grad_y * temporal_diff).sum(dim=(1, 2, 3))
        denom = (a11 * a22 - a12 * a12).clamp_min(1e-6)
        flow_u = (b1 * a22 - b2 * a12) / denom
        flow_v = (b2 * a11 - b1 * a12) / denom
        return float(torch.sqrt(flow_u * flow_u + flow_v * flow_v).mean().item())

    def _compute_temporal_lpips_proxy(self) -> float:
        frames = self.scalar_frames
        pair_count = frames.shape[0] * (frames.shape[1] - 1)
        prev = frames[:, :-1].reshape(pair_count, 1, frames.shape[2], frames.shape[3])
        nxt = frames[:, 1:].reshape_as(prev)
        kernel_x, kernel_y = self.sobel_kernels

        distances = []
        for scale in (1, 2, 4):
            if scale > 1:
                prev_s = F.avg_pool2d(prev, kernel_size=scale, stride=scale)
                nxt_s = F.avg_pool2d(nxt, kernel_size=scale, stride=scale)
            else:
                prev_s, nxt_s = prev, nxt

            prev_feat = torch.cat(
                [
                    prev_s,
                    F.conv2d(prev_s, kernel_x, padding=1),
                    F.conv2d(prev_s, kernel_y, padding=1),
                ],
                dim=1,
            )
            nxt_feat = torch.cat(
                [
                    nxt_s,
                    F.conv2d(nxt_s, kernel_x, padding=1),
                    F.conv2d(nxt_s, kernel_y, padding=1),
                ],
                dim=1,
            )
            prev_flat = F.normalize(prev_feat.flatten(1), dim=1, eps=1e-6)
            nxt_flat = F.normalize(nxt_feat.flatten(1), dim=1, eps=1e-6)
            distances.append((1.0 - (prev_flat * nxt_flat).sum(dim=1)).mean())

        return float(torch.stack(distances).mean().item())

    def _compute_temporal_fft_high_frequency(self) -> float:
        frames = self.scalar_frames
        signals = frames.permute(0, 2, 3, 1).reshape(-1, frames.shape[1])
        signals = signals - signals.mean(dim=1, keepdim=True)
        fft_values = torch.fft.rfft(signals, dim=1)
        power = fft_values.real.square() + fft_values.imag.square()
        non_dc = power[:, 1:]
        if non_dc.shape[1] == 0:
            return 0.0
        split = max(1, non_dc.shape[1] // 2)
        return float((non_dc[:, split:].sum() / non_dc.sum().clamp_min(1e-6)).item())

    def _compute_spatial_edge(self) -> float:
        kernel_x, kernel_y = self.sobel_kernels
        grad_x = F.conv2d(self.flat_scalar_frames, kernel_x, padding=1)
        grad_y = F.conv2d(self.flat_scalar_frames, kernel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        return float(grad_mag.mean().item())

    def _compute_spatial_fft_energy_above_50(self) -> float:
        """FFT high-band fraction: H/(1+H) for H = high-band/low-band energy."""
        x0 = self.x0.detach()
        mean = x0.mean(dim=(2, 3, 4), keepdim=True)
        std = x0.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        x0 = (x0 - mean) / std
        height, width = x0.shape[-2:]
        power = torch.fft.fftshift(
            torch.fft.fft2(x0, dim=(-2, -1), norm="ortho").abs().square(),
            dim=(-2, -1),
        )
        fy = torch.linspace(-1.0, 1.0, height, device=x0.device)
        fx = torch.linspace(-1.0, 1.0, width, device=x0.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(2.0)
        return float((power[..., radius >= 0.5].sum() / power.sum().clamp_min(1e-12)).item())

    def _compute_spatial_high_frequency_scale(self, kernel_size: int) -> float:
        blurred = F.avg_pool2d(
            self.flat_latents,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return float((self.flat_latents - blurred).abs().mean().item())

    def _compute_spatial_high_frequency_scale3(self) -> float:
        return self._compute_spatial_high_frequency_scale(3)

    def _compute_spatial_high_frequency_scale5(self) -> float:
        return self._compute_spatial_high_frequency_scale(5)

    def _compute_spatial_high_frequency_scale7(self) -> float:
        return self._compute_spatial_high_frequency_scale(7)

    def _compute_spatial_multi_scale_high_frequency(self) -> float:
        return sum(
            self.compute_spatial(metric)
            for metric in ("hf_scale3", "hf_scale5", "hf_scale7")
        ) / 3.0

    def _compute_spatial_high_frequency(self) -> float:
        # Preserve the original spatial score: temporal FFT detail + spatial detail.
        return 0.5 * self.compute_temporal("temporal_fft_high_freq") + 0.5 * self.compute_spatial(
            "multi_scale_high_freq"
        )


def _bounded_norm(value: float) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"Expected non-negative value for bounded norm, got {value}")
    return float(value / (1.0 + value))


def _log_beta_prior_pdf(x: float, a: float, b: float) -> float:
    x = float(min(max(x, 1e-6), 1.0 - 1e-6))
    return float((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x))


def extract_transition_norm_scores(
    latest: torch.Tensor,
    *,
    m_t_metric: str = "flow_mag",
    m_s_metric: str = "spatial_fft_energy_above_50",
) -> Dict[str, float]:
    calculator = LatentMetricCalculator(latest)
    m_t_raw = calculator.compute_temporal(m_t_metric)
    m_s_raw = calculator.compute_spatial(m_s_metric)

    return {
        "m_t_metric": m_t_metric,
        "m_s_metric": m_s_metric,
        "m_t_raw": m_t_raw,
        "m_t_norm": calculator.normalize_temporal(m_t_metric, m_t_raw),
        "m_s_raw": m_s_raw,
        "m_s_norm": calculator.normalize_spatial(m_s_metric, m_s_raw),
    }


def get_action_list(
    stage1_spatial_rate: float,
    stage1_temporal_rate: float,
    density_budget: float,
    infer_step_count: int,
    latent_hwt: Sequence[int],
    step_rate_list: Sequence[float] = (0.28, 0.50, 0.72, 1.0),
    error: float = 0.02,
    spatial_interval: float = 0.1,
    temporal_interval: int = 1,
    include_stage1_spatial: bool = False,
    max_actions: int | None = 10,
    resolve_shapes: Callable[[Sequence[float], Sequence[float]], Sequence[Sequence[int]]] | None = None,
) -> Sequence[Dict[str, Any]]:
    if infer_step_count <= 0:
        raise ValueError(f"infer_step_count must be positive, got {infer_step_count}")
    if len(latent_hwt) != 3:
        raise ValueError(f"latent_hwt must be (H, W, T), got {latent_hwt}")
    if len(step_rate_list) != 4:
        raise ValueError(f"step_rate_list must contain 4 stage ratios, got {step_rate_list}")
    if spatial_interval <= 0:
        raise ValueError(f"spatial_interval must be positive, got {spatial_interval}")
    if int(temporal_interval) != temporal_interval or temporal_interval <= 0:
        raise ValueError(f"temporal_interval must be positive, got {temporal_interval}")

    stage1_spatial_rate = float(stage1_spatial_rate)
    stage1_temporal_rate = float(stage1_temporal_rate)
    density_budget = float(density_budget)
    step_rate_list = [float(rate) for rate in step_rate_list]
    error = float(error)
    spatial_interval = float(spatial_interval)
    temporal_interval = int(temporal_interval)

    latent_h = int(latent_hwt[0])
    latent_w = int(latent_hwt[1])
    latent_t = int(latent_hwt[2])
    if resolve_shapes is None:
        resolve_shapes = lambda spatial, temporal: build_joint_stage_shapes(
            latent_t, latent_h, latent_w, spatial, temporal
        )
    stage_step_counts = compute_runtime_stage_step_counts(infer_step_count, step_rate_list)

    stage2_spatial_start = min(1.0, stage1_spatial_rate) if include_stage1_spatial else max(stage1_spatial_rate, 0.6)
    if include_stage1_spatial:
        spatial_rates = []
        rate = stage2_spatial_start
        while rate < 1.0 - 1e-9:
            spatial_rates.append(round(rate, 6))
            rate += spatial_interval
    else:
        spatial_start_index = max(1, math.ceil(stage2_spatial_start / spatial_interval - 1e-12))
        spatial_rates = [
            round(index * spatial_interval, 12)
            for index in range(spatial_start_index, math.floor(1.0 / spatial_interval + 1e-12) + 1)
        ]
    if not spatial_rates or not math.isclose(spatial_rates[-1], 1.0):
        spatial_rates.append(1.0)

    stage1_temporal_count = max(1, min(latent_t, int(round(latent_t * stage1_temporal_rate))))
    stage1_temporal_rate = stage1_temporal_count / float(latent_t)
    temporal_counts = list(range(stage1_temporal_count, latent_t + 1, temporal_interval))
    if temporal_counts[-1] != latent_t:
        temporal_counts.append(latent_t)

    actions: List[Dict[str, Any]] = []

    for stage2_spatial_rate in spatial_rates:
        spatial_rate_list = [stage1_spatial_rate, stage2_spatial_rate, 1.0, 1.0]

        for stage23_temporal_count in temporal_counts:
            stage23_temporal_rate = stage23_temporal_count / float(latent_t)
            temporal_rate_list = [
                stage1_temporal_rate,
                stage23_temporal_rate,
                stage23_temporal_rate,
                1.0,
            ]

            resolved = resolve_shapes(spatial_rate_list, temporal_rate_list)
            stage_shapes = [[int(h), int(w), int(t)] for t, h, w in resolved]

            real_density = compute_density(
                resolved,
                (latent_t, latent_h, latent_w),
                stage_step_counts,
            )
            density_gap = real_density - density_budget
            stage2_spatial_label = f"{stage2_spatial_rate:.6f}".rstrip("0")
            if stage2_spatial_label.endswith("."):
                stage2_spatial_label += "0"

            action = {
                "action_id": (
                    f"s1s{stage1_spatial_rate:.3f}_s1t{stage1_temporal_count}_"
                    f"s2s{stage2_spatial_label}_s23t{stage23_temporal_count}"
                ),
                "spatial_rate_list": spatial_rate_list,
                "temporal_rate_list": temporal_rate_list,
                "stage_shapes_hwt": stage_shapes,
                "real_density": real_density,
                "density_gap": density_gap,
                "stage_step_counts": list(stage_step_counts),
                "density_budget": density_budget,
                "infer_step_count": infer_step_count,
            }

            actions.append(action)

    matched_actions = [
        action for action in actions if abs(float(action["density_gap"])) <= error
    ]
    matched_actions.sort(
        key=lambda action: (
            abs(float(action["density_gap"])),
            -float(action["real_density"]),
            -float(action["spatial_rate_list"][1]),
            -float(action["temporal_rate_list"][1]),
        )
    )

    if not matched_actions:
        closest_actions = sorted(
            actions,
            key=lambda action: (
                abs(float(action["density_gap"])),
                -float(action["real_density"]),
            ),
        )[:3]
        rank0_print(
            "[X0-ACTION] No candidate satisfies the density tolerance: "
            f"density_budget={density_budget:.4f}, error={error:.4f}, "
            f"closest_candidates={closest_actions}"
        )
        raise ValueError(
            "No stage-transition action satisfies the density tolerance. "
            f"budget={density_budget:.4f}, tolerance={error:.4f}."
        )

    selected_actions = matched_actions if max_actions is None else matched_actions[:max_actions]
    rank0_print(
        "[X0-ACTION] Generated action list with "
        f"stage1_spatial_rate={stage1_spatial_rate:.4f}, "
        f"stage1_temporal_rate={stage1_temporal_rate:.4f}, "
        f"density_budget={density_budget:.4f}, error={error:.4f}, "
        f"matched_count={len(matched_actions)}, returned_count={len(selected_actions)}, "
        f"step_rate_list={step_rate_list}, stage_step_counts={stage_step_counts}"
    )

    return selected_actions


def score_action_list(
    action_list: Sequence[Dict[str, Any]],
    x0_latent_history: Sequence[Any],
    *,
    latent_t_full: int,
    score_mode: str = "parameter_free_softmax",
    score_input_name: str = "x0_pred",
    m_t_metric: str = "flow_mag",
    m_s_metric: str = "spatial_fft_energy_above_50",
    decision_lambda: float = 0.5,
    demand_alpha: float = 2.6,
    spatial_prior_weight: float = 0.08,
) -> Sequence[Dict[str, Any]]:
    LatentMetricCalculator.validate_metrics(m_t_metric, m_s_metric)
    if not action_list:
        return []

    metric_scores = {
        "m_t_metric": m_t_metric,
        "m_t_raw": 0.0,
        "m_t_norm": 0.0,
        "m_s_metric": m_s_metric,
        "m_s_raw": 0.0,
        "m_s_norm": 0.0,
    }

    if x0_latent_history:
        try:
            latest = x0_latent_history[-1]
            if torch.is_tensor(latest):
                latest = latest.detach().to(torch.float32)
                if latest.ndim == 5 and latest.shape[2] >= 2:
                    metric_scores = extract_transition_norm_scores(
                        latest,
                        m_t_metric=m_t_metric,
                        m_s_metric=m_s_metric,
                    )
        except Exception as exc:
            rank0_print(
                f"[X0-ACTION] Failed to compute selected metrics from {score_input_name}: {exc}"
            )

    m_t_raw = float(metric_scores["m_t_raw"])
    m_t_norm = float(metric_scores["m_t_norm"])
    m_s_raw = float(metric_scores["m_s_raw"])
    m_s_norm = float(metric_scores["m_s_norm"])

    decision_lambda = float(decision_lambda)
    demand_alpha = float(demand_alpha)
    gamma_space = 0.5
    gamma_time = 0.5
    spatial_prior_a = 5.0
    spatial_prior_b = 2.0
    spatial_prior_weight = float(spatial_prior_weight)
    eps = 1e-6
    scored_actions: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    spatial_squares = [float(action["spatial_rate_list"][1]) ** 2 for action in action_list]
    temporal_frames = [
        int(round(float(action["temporal_rate_list"][1]) * max(int(latent_t_full), 1)))
        for action in action_list
    ]
    min_spatial_square, max_spatial_square = min(spatial_squares), max(spatial_squares)
    min_temporal_frame, max_temporal_frame = min(temporal_frames), max(temporal_frames)

    for action_idx, action in enumerate(action_list):
        spatial_rate_list = [float(rate) for rate in action["spatial_rate_list"]]
        temporal_rate_list = [float(rate) for rate in action["temporal_rate_list"]]

        s1 = float(spatial_rate_list[0])
        s2 = spatial_rate_list[1]
        full_frames = max(int(latent_t_full), 1)
        t1_frames = int(round(float(temporal_rate_list[0]) * full_frames))
        t2_frames = int(round(float(temporal_rate_list[1]) * full_frames))
        t3_frames = int(round(float(temporal_rate_list[2]) * full_frames))

        stage_step_counts = action.get("stage_step_counts", [0, 0, 0, 0])
        if len(stage_step_counts) < 4:
            stage_step_counts = list(stage_step_counts) + [0] * (4 - len(stage_step_counts))
        n2 = float(stage_step_counts[1])
        n3 = float(stage_step_counts[2])
        stage23_steps = max(n2 + n3, 1.0)

        init_space_keep = float(s1 * s1)
        init_time_keep = float(t1_frames / full_frames)
        t2_ratio = float(t2_frames / full_frames)
        t3_ratio = float(t3_frames / full_frames)

        space_keep = float((s2 * s2 * n2 + 1.0 * n3) / stage23_steps)
        time_keep = float((t2_ratio * n2 + t3_ratio * n3) / stage23_steps)
        space_gain = float(max((space_keep - init_space_keep) / max(1.0 - init_space_keep, 1e-8), 0.0))
        time_gain = float(max((time_keep - init_time_keep) / max(1.0 - init_time_keep, 1e-8), 0.0))
        spatial_prior = 0.0
        space_mismatch = None
        time_mismatch = None

        if score_mode == "demand_match":
            demand_space = math.exp(demand_alpha * m_s_norm)
            demand_time = math.exp(demand_alpha * m_t_norm)
            demand_denom = demand_space + demand_time + eps
            m_s = float(demand_space / demand_denom)
            m_t = float(demand_time / demand_denom)

            space_mismatch = float((space_gain - m_s) ** 2)
            time_mismatch = float((time_gain - m_t) ** 2)
            space_term = float(-decision_lambda * space_mismatch)
            time_term = float(-(1.0 - decision_lambda) * time_mismatch)
            spatial_prior = float(
                spatial_prior_weight * _log_beta_prior_pdf(s2, spatial_prior_a, spatial_prior_b)
            )
            score = float(space_term + time_term + spatial_prior)
            space_gain_score = space_gain
            time_gain_score = time_gain
        elif score_mode == "linear":
            space_gain_score = float(space_gain ** gamma_space)
            time_gain_score = float(time_gain ** gamma_time)
            space_term = float(decision_lambda * m_s_norm * space_gain_score)
            time_term = float((1.0 - decision_lambda) * m_t_norm * time_gain_score)
            m_s = None
            m_t = None
            score = float(space_term + time_term)
        elif score_mode == "parameter_free_softmax":
            if m_s_metric != "spatial_fft_energy_above_50" or m_t_metric != "flow_mag":
                raise ValueError(
                    "parameter_free_softmax requires m_s_metric='spatial_fft_energy_above_50' "
                    "and m_t_metric='flow_mag'."
                )
            exp_space = math.exp(demand_alpha * m_s_norm)
            exp_time = math.exp(demand_alpha * m_t_norm)
            demand_sum = exp_space + exp_time
            m_s = float(exp_space / demand_sum)
            m_t = float(exp_time / demand_sum)
            space_gain_score = float(
                (s2 * s2 - min_spatial_square)
                / max(max_spatial_square - min_spatial_square, 1e-8)
            )
            time_gain_score = float(
                (t2_frames - min_temporal_frame)
                / max(max_temporal_frame - min_temporal_frame, 1)
            )
            space_mismatch = float((space_gain_score - m_s) ** 2)
            time_mismatch = float((time_gain_score - m_t) ** 2)
            space_term = float(-decision_lambda * space_mismatch)
            time_term = float(-(1.0 - decision_lambda) * time_mismatch)
            score = float(space_term + time_term)
        else:
            raise ValueError(f"Unsupported stage-transition action score_mode: {score_mode}")

        scored_action = dict(action)
        scored_action.update(
            {
                "temporal_frames": [
                    int(round(float(rate) * full_frames)) for rate in temporal_rate_list
                ],
                "space_keep": space_keep,
                "time_keep": time_keep,
                "space_gain_raw": space_gain,
                "time_gain_raw": time_gain,
                "space_gain_score": space_gain_score,
                "time_gain_score": time_gain_score,
                "demand_space": m_s,
                "demand_time": m_t,
                "space_mismatch": space_mismatch,
                "time_mismatch": time_mismatch,
                "space_term": space_term,
                "time_term": time_term,
                "spatial_prior": spatial_prior,
                **metric_scores,
                "score_mode": score_mode,
                "score_input_name": score_input_name,
                "score": score,
            }
        )
        scored_actions.append(scored_action)
        score_rows.append(
            {
                "action_idx": action_idx,
                "action_id": action.get("action_id", f"action_{action_idx}"),
                "score": score,
                "score_mode": score_mode,
                "score_input_name": score_input_name,
                "real_density": float(action.get("real_density", 0.0)),
                "density_gap": float(action.get("density_gap", 0.0)),
                "spatial_rate_list": list(spatial_rate_list),
                "temporal_rate_list": list(temporal_rate_list),
                "temporal_frames": [int(round(float(rate) * full_frames)) for rate in temporal_rate_list],
                "stage_shapes_hwt": action.get("stage_shapes_hwt"),
                "stage_step_counts": list(stage_step_counts),
                "space_gain_raw": space_gain,
                "time_gain_raw": time_gain,
                "space_gain_score": space_gain_score,
                "time_gain_score": time_gain_score,
                "space_keep": space_keep,
                "time_keep": time_keep,
                "space_term": space_term,
                "time_term": time_term,
                "demand_space": m_s,
                "demand_time": m_t,
                "space_mismatch": space_mismatch,
                "time_mismatch": time_mismatch,
                "spatial_prior": spatial_prior,
            }
        )

    scored_actions.sort(key=lambda action: float(action["score"]), reverse=True)

    scoring_context = {
        "score_mode": score_mode,
        "score_input_name": score_input_name,
        **metric_scores,
        "decision_lambda": decision_lambda,
        "demand_alpha": demand_alpha,
        "spatial_prior_beta": (spatial_prior_a, spatial_prior_b),
        "spatial_prior_weight": spatial_prior_weight,
        "gamma_space": gamma_space,
        "gamma_time": gamma_time,
    }
    rank0_print(
        "[X0-ACTION] Selected metric scores and scoring context: "
        f"{scoring_context}"
    )
    rank0_print(
        "[X0-ACTION] Candidate score breakdowns: "
        f"{sorted(score_rows, key=lambda row: row['score'], reverse=True)}"
    )

    return scored_actions


def select_action(
    density_budget: float,
    infer_step_count: int,
    latent_hwt: Sequence[int],
    stage1_spatial_rate: float,
    stage1_temporal_rate: float,
    x0_latent_history: Sequence[Any],
    step_rate_list: Sequence[float] = (0.28, 0.50, 0.72, 1.0),
    density_tolerance: float = 0.02,
    score_mode: str = "parameter_free_softmax",
    score_input_name: str = "x0_pred",
    m_t_metric: str = "flow_mag",
    m_s_metric: str = "spatial_fft_energy_above_50",
    decision_lambda: float = 0.5,
    demand_alpha: float = 2.6,
    spatial_prior_weight: float = 0.08,
    spatial_interval: float = 0.1,
    temporal_interval: int = 1,
    resolve_shapes: Callable[[Sequence[float], Sequence[float]], Sequence[Sequence[int]]] | None = None,
) -> Dict[str, Any]:
    generated_actions = get_action_list(
        stage1_spatial_rate=stage1_spatial_rate,
        stage1_temporal_rate=stage1_temporal_rate,
        density_budget=density_budget,
        infer_step_count=infer_step_count,
        latent_hwt=latent_hwt,
        step_rate_list=tuple(float(rate) for rate in step_rate_list),
        error=float(density_tolerance),
        spatial_interval=spatial_interval,
        temporal_interval=temporal_interval,
        include_stage1_spatial=score_mode == "parameter_free_softmax",
        max_actions=None if score_mode == "parameter_free_softmax" else 10,
        resolve_shapes=resolve_shapes,
    )

    scored_actions = score_action_list(
        generated_actions,
        x0_latent_history,
        latent_t_full=int(latent_hwt[2]),
        score_mode=score_mode,
        score_input_name=score_input_name,
        m_t_metric=m_t_metric,
        m_s_metric=m_s_metric,
        decision_lambda=decision_lambda,
        demand_alpha=demand_alpha,
        spatial_prior_weight=spatial_prior_weight,
    )

    best_action = dict(scored_actions[0])
    best_action["selection_score"] = float(best_action["score"])

    # logging
    preview_rows = [
        {
            "rank": idx + 1,
            "action_id": action.get("action_id"),
            "score": float(action["score"]),
            "real_density": float(action.get("real_density", 0.0)),
            "spatial_rate_list": list(action["spatial_rate_list"]),
            "temporal_rate_list": list(action["temporal_rate_list"]),
            "temporal_frames": list(action.get("temporal_frames", [])),
        }
        for idx, action in enumerate(scored_actions[:3])
    ]
    rank0_print(
        "[X0-ACTION] Selected best action via get_action_list + score_action_list with "
        f"density_budget={density_budget:.4f}, infer_step_count={infer_step_count}, latent_hwt={tuple(latent_hwt)}, "
        f"density_tolerance={float(density_tolerance):.4f}, "
        f"score_mode={score_mode}, "
        f"score_input_name={score_input_name}, "
        f"m_t_metric={m_t_metric}, "
        f"m_s_metric={m_s_metric}, "
        f"decision_lambda={float(decision_lambda):.4f}, "
        f"demand_alpha={float(demand_alpha):.4f}, "
        f"spatial_prior_weight={float(spatial_prior_weight):.4f}, "
        f"stage1_spatial_rate={float(stage1_spatial_rate):.4f}, "
        f"stage1_temporal_rate={float(stage1_temporal_rate):.4f}, "
        f"candidate_count={len(generated_actions)}, best_action_id={best_action.get('action_id')}, "
        f"best_score={best_action['selection_score']:.6f}, "
        f"best_density={float(best_action.get('real_density', 0.0)):.6f}, "
        f"best_spatial={best_action['spatial_rate_list']}, "
        f"best_temporal={best_action['temporal_rate_list']}, "
        f"best_temporal_frames={best_action.get('temporal_frames', [])}, "
        f"top_candidates={preview_rows}"
    )

    return best_action
