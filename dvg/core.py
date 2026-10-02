"""The single implementation of DVG planning and stage transitions."""

from __future__ import annotations

import math
import torch

from . import algorithms as alg


class DVG:
    """Per-pipeline DVG hook. ``prepare`` resets all per-generation state."""

    spatial_prior_weight = 0.08
    density_tolerance = 0.02
    score_mode = "parameter_free_softmax"
    m_t_metric = "flow_mag"
    m_s_metric = "spatial_fft_energy_above_50"
    spatial_interval = 0.1
    temporal_interval = 1
    noise_seed = 12345
    resize_mode = "temporal_interp_spatial_restore"
    noise_cache_dir = "./coordinate_noise_cache"

    def __init__(
        self,
        *,
        net=None,
        transition_rates=(0.28, 0.50, 0.72),
        scheduler_shifts=(7.0, 9.0, 11.0, 11.0),
    ):
        self.net = net
        self.transition_rates = tuple(float(rate) for rate in transition_rates)
        self.scheduler_shifts = tuple(float(shift) for shift in scheduler_shifts)
        self.enabled = False

    @property
    def current_shape(self):
        return self.stage_shapes[self.stage_idx]

    @property
    def current_axes(self):
        return self.stage_axes[self.stage_idx]

    @property
    def n_tokens(self):
        return math.prod(self.token_shape(self.stage_idx))

    def token_shape(self, stage_idx):
        return self.stage_shapes[stage_idx]

    def resolve_shapes(self, spatial_rates, temporal_rates):
        t, h, w = self.full_shape
        return alg.build_joint_stage_shapes(t, h, w, spatial_rates, temporal_rates)

    def action_shapes(self, spatial_rates, temporal_rates):
        return self.resolve_shapes(spatial_rates, temporal_rates)

    def prepare(self, *, enabled, budget, full_shape, num_steps, device, **context):
        self.enabled = bool(enabled)
        self.budget = float(budget)
        if self.enabled and not 0 < self.budget <= 1:
            raise ValueError(f"DVG budget must be in (0, 1], got {self.budget}")
        self.full_shape = tuple(map(int, full_shape))
        self.num_steps = int(num_steps)
        self.device = device
        self.context = context
        self.stage_idx = 0
        self.spatial_rates = [0.5, 1.0, 1.0, 1.0] if self.enabled else [1.0]
        self.temporal_rates = [self.budget * 0.8, 1.0, 1.0, 1.0] if self.enabled else [1.0]
        self.step_rates = [*self.transition_rates, 1.0] if self.enabled else [1.0]
        self.shifts = list(self.scheduler_shifts) if self.enabled else []
        self.stage_shapes = self.resolve_shapes(self.spatial_rates, self.temporal_rates)
        self.stage_axes = alg.build_stage_coordinate_axes(self.stage_shapes)
        alg.validate_config(self.enabled, self.stage_shapes, self.step_rates, self.shifts)
        self.transition_steps = {self.transition_step(rate) for rate in self.transition_rates} if self.enabled else set()
        self.density = self._density(self.spatial_rates, self.temporal_rates)
        return self.initial_values()

    def transition_step(self, rate):
        return int(self.num_steps * rate)

    def _density(self, spatial_rates, temporal_rates):
        shapes = self.action_shapes(spatial_rates, temporal_rates)
        counts = alg.compute_runtime_stage_step_counts(self.num_steps, self.step_rates)
        self.stage_step_counts = counts
        return alg.compute_density(shapes, self.full_shape, counts)

    def initial_values(self):
        return self.current_shape

    def plan(self, x0):
        t, h, w = self.full_shape
        action = alg.select_action(
            density_budget=self.budget,
            infer_step_count=self.num_steps,
            latent_hwt=(h, w, t),
            stage1_spatial_rate=self.spatial_rates[0],
            stage1_temporal_rate=self.temporal_rates[0],
            x0_latent_history=[x0.detach()],
            step_rate_list=self.step_rates,
            density_tolerance=self.density_tolerance,
            score_mode=self.score_mode,
            m_t_metric=self.m_t_metric,
            m_s_metric=self.m_s_metric,
            spatial_prior_weight=self.spatial_prior_weight,
            spatial_interval=self.spatial_interval,
            temporal_interval=self.temporal_interval,
            resolve_shapes=self.action_shapes,
        )
        self.spatial_rates = list(action["spatial_rate_list"])
        self.temporal_rates = list(action["temporal_rate_list"])
        self.stage_shapes = self.resolve_shapes(self.spatial_rates, self.temporal_rates)
        self.stage_axes = alg.build_stage_coordinate_axes(self.stage_shapes)
        self.density = self._density(self.spatial_rates, self.temporal_rates)
        return action

    def tensor(self, latents):
        return latents

    def predict_x0(self, scheduler, noise_pred, t, latents, extra_step_kwargs):
        return scheduler.predict_x0_from_xt(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]

    def reset_scheduler(self, scheduler, next_stage_idx, step_index):
        scheduler.config.shift = self.shifts[next_stage_idx]
        scheduler.set_timesteps(self.num_steps, device=self.device,
                                n_tokens=math.prod(self.token_shape(next_stage_idx)))
        scheduler._step_index = step_index

    def renoise(self, scheduler, resized_x0, stage_noise, next_timestep, target_dtype):
        return scheduler.add_noise_to_step(resized_x0, stage_noise, next_timestep)[0].to(target_dtype)

    def after_transition(self, new_latents, target_dtype, **context):
        return new_latents

    def apply(self, *, scheduler, noise_pred, t, latents, step_index,
              extra_step_kwargs=None, target_dtype=None, **context):
        if not self.enabled or step_index not in self.transition_steps or self.stage_idx >= len(self.stage_shapes) - 1:
            return None
        if step_index + 1 >= self.num_steps:
            raise ValueError("DVG transition cannot occur at the final sampling step")

        x0 = self.predict_x0(scheduler, noise_pred, t, latents, extra_step_kwargs or {})
        if self.stage_idx == 0:
            #planning at scretch stage
            self.plan(x0)
        next_stage = self.stage_idx + 1
        next_shape = self.stage_shapes[next_stage]
        self.reset_scheduler(scheduler, next_stage, step_index)

        resized = alg.resize_latents(
            x0, next_shape, mode=self.resize_mode,
            current_stage_idx=self.stage_idx, next_stage_idx=next_stage,
            stage_coordinate_axes=self.stage_axes,
        ).to(dtype=self.tensor(latents).dtype)
        if next_stage == 1:
            resized = alg.smooth_stage2_transition(resized)
        axis_t, axis_h, axis_w = self.stage_axes[next_stage]
        stage_noise = alg.get_transition_noise(
            batch_size=resized.shape[0], channels=resized.shape[1],
            axis_t=tuple(axis_t), axis_h=tuple(axis_h), axis_w=tuple(axis_w),
            stage_idx=next_stage, seed=self.noise_seed,
            cache_dir=self.noise_cache_dir, device=self.device, dtype=resized.dtype,
        )
        if hasattr(scheduler, "init_noise_sigma"):
            stage_noise = stage_noise * scheduler.init_noise_sigma
        new_latents = self.renoise(
            scheduler, resized, stage_noise, scheduler.timesteps[step_index + 1],
            target_dtype or self.tensor(latents).dtype,
        )
        self.stage_idx = next_stage
        scheduler._step_index = step_index + 1
        return self.after_transition(new_latents, target_dtype or self.tensor(latents).dtype, **context)
