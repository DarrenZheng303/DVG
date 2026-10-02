"""HunyuanVideo-1.5 i2v condition adaptation."""

import torch

from .core import DVG


class DVG_HunyuanVideo1_5(DVG):
    def __init__(self, *, net, host, **kwargs):
        super().__init__(net=net, **kwargs)
        self.host = host

    def initial_values(self):
        task_type = self.context["task_type"]
        return self.current_shape, self.host.get_task_mask(task_type, self.current_shape[0])

    @staticmethod
    def select_cond_latents_by_hw_axes(source_cond_latents, axis_h, axis_w):
        h_idx = torch.tensor(list(map(int, axis_h)), device=source_cond_latents.device, dtype=torch.long)
        w_idx = torch.tensor(list(map(int, axis_w)), device=source_cond_latents.device, dtype=torch.long)
        return source_cond_latents.index_select(3, h_idx).index_select(4, w_idx)

    def get_stage_image_condition_latent(self, task_type, image_cond_cache):
        if task_type == "t2v" or image_cond_cache is None:
            return None

        stage_key = tuple(map(int, self.current_shape[1:]))
        current_image_cond = image_cond_cache.get(stage_key)
        if current_image_cond is None:
            full_cond_latents = image_cond_cache.get("__full__")
            if full_cond_latents is not None:
                _, axis_h, axis_w = self.current_axes
                current_image_cond = self.select_cond_latents_by_hw_axes(full_cond_latents, axis_h, axis_w)
                image_cond_cache[stage_key] = current_image_cond
                return current_image_cond
            available = sorted(key for key in image_cond_cache if key != "__full__")
            raise ValueError(
                "Missing cached i2v condition latent for stage shape "
                f"{stage_key}. Available cached shapes: {available}"
            )
        return current_image_cond

    def after_transition(self, new_latents, target_dtype, *, task_type, image_cond_cache, **context):
        mask = self.host.get_task_mask(task_type, self.current_shape[0])
        image_cond = self.get_stage_image_condition_latent(task_type, image_cond_cache)
        cond_latents = self.host._prepare_cond_latents(task_type, image_cond, new_latents, mask)
        return new_latents, cond_latents
