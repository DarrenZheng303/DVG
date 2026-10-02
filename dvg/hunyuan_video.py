"""HunyuanVideo-specific latent/token geometry and RoPE output."""

from . import algorithms as alg
from .core import DVG


class DVG_HunyuanVideo(DVG):
    spatial_prior_weight = 0.15

    def __init__(self, *, net, rope_theta, **kwargs):
        super().__init__(net=net, **kwargs)
        self.rope_theta = rope_theta

    def action_shapes(self, spatial_rates, temporal_rates):
        t, h, w = self.full_shape
        return alg.build_joint_stage_shapes(t, h, w, spatial_rates, temporal_rates)

    def resolve_shapes(self, spatial_rates, temporal_rates):
        raw = self.action_shapes(spatial_rates, temporal_rates)
        self.token_shapes = [[t, h // 16, w // 16] for t, h, w in raw]
        return [[t, (h // 16) * 2, (w // 16) * 2] for t, h, w in raw]

    def token_shape(self, stage_idx):
        return self.token_shapes[stage_idx]

    def _rope(self, dtype):
        from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed

        t, h, w = self.current_shape
        patch = self.net.patch_size
        cos, sin = get_nd_rotary_pos_embed(
            self.net.rope_dim_list,
            [t // patch[0], h // patch[1], w // patch[2]],
            theta=self.rope_theta, use_real=True, theta_rescale_factor=1,
        )
        return cos.to(device=self.device, dtype=dtype), sin.to(device=self.device, dtype=dtype)

    def initial_values(self):
        return self.current_shape, *self._rope(self.context["dtype"])

    def after_transition(self, new_latents, target_dtype, **context):
        return new_latents.contiguous(), *self._rope(target_dtype)
