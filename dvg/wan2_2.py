"""Wan T2V stage alignment and FlowMatch scheduler adaptation."""

import math

from . import algorithms as alg
from .core import DVG


class DVG_WanT2V(DVG):
    def __init__(self, *, net, patch_size, sp_size, **kwargs):
        super().__init__(net=net, **kwargs)
        self.patch_size = patch_size
        self.sp_size = sp_size

    def transition_step(self, rate):
        return min(self.num_steps - 1, super().transition_step(rate))

    def resolve_shapes(self, spatial_rates, temporal_rates):
        t, h, w = self.full_shape
        shapes = alg.build_joint_stage_shapes(t, h, w, spatial_rates, temporal_rates)
        previous = [1, min(h, 2), min(w, 2)]
        for shape in shapes[:-1]:
            for axis, multiple in ((1, 2), (2, 2)):
                value = max(previous[axis], (shape[axis] // multiple) * multiple)
                shape[axis] = min(self.full_shape[axis], value)
            previous = shape
        return shapes

    def _seq_len(self):
        t, h, w = self.current_shape
        patches = self.patch_size[1] * self.patch_size[2]
        return math.ceil(h * w * t / patches / self.sp_size) * self.sp_size

    def initial_values(self):
        return self.current_shape, self._seq_len()

    def tensor(self, latents):
        return latents[0].unsqueeze(0)

    def predict_x0(self, scheduler, noise_pred, t, latents, extra_step_kwargs):
        sample = self.tensor(latents)
        idx = scheduler.index_for_timestep(t.to(scheduler.timesteps.device), scheduler.timesteps)
        sigma = scheduler.sigmas[idx].to(device=sample.device, dtype=sample.dtype)
        return sample - sigma * noise_pred.unsqueeze(0)

    def reset_scheduler(self, scheduler, next_stage_idx, step_index):
        scheduler.config.shift = self.shifts[next_stage_idx]
        scheduler.set_timesteps(self.num_steps, device=self.device)
        scheduler._step_index = step_index

    def renoise(self, scheduler, resized_x0, stage_noise, next_timestep, target_dtype):
        return [scheduler.add_noise(resized_x0, stage_noise, next_timestep)[0].to(dtype=target_dtype)]

    def after_transition(self, new_latents, target_dtype, **context):
        return new_latents, self._seq_len()


DVG_Wan2_2 = DVG_WanT2V
