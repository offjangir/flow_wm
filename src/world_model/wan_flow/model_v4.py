"""
v4 transformer: ``WanTransformerRenderConditionedV4``.

A thin subclass of ``WanTransformerRenderConditioned`` (model.py) that swaps
the render-conditioning path for ``ActionRenderSelfCrossAdapterV4``
(embodiment_adapter_v3.py) without touching model.py.

v4 builds on v3 (Pathway A — cross-attn transformer blocks) with three
additions targeting the render-encoder failure modes diagnosed by
``scripts/diagnose_egowm_collapse.py``:

  1. Pre-encoder mean subtraction on the VAE render latents (per channel)
     — removes the shared ``f(black canvas)`` baseline so the conv encoder
     learns from arm-signal deviations only.
  2. Temporal mixing in the middle conv (3×3×3 kernel with replicate-padded
     temporal boundary) + 3D positional encoding on the resulting token
     grid + N self-attention transformer blocks so arm-region tokens can
     propagate their pose information to background tokens before the
     cross-attention with actions kicks in.
  3. 1D positional encoding on action tokens before the cross-attention,
     so the cross-attn can reason about temporal alignment instead of
     treating action tokens as an unordered set.

Only ``__init__``, ``forward``, and ``_keep_in_fp32_modules`` are overridden.
Everything else — base Wan DiT blocks, RoPE, patch embed, norm_out,
proj_out, tracks_head — is inherited unchanged.

All adapter layers use standard PyTorch init (kaiming-uniform). No
identity-at-init is used; the pre-norm + residual transformer-block
structure is the stability mechanism.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers

from world_model.wan_flow.model import WanTransformerRenderConditioned
from world_model.wan_flow.embodiment_adapter_v3 import ActionRenderSelfCrossAdapterV4


class WanTransformerRenderConditionedV4(WanTransformerRenderConditioned):
    """Wan DiT with the v4 self-attn + cross-attn render adapter.

    Construction trick (same as v3): the parent ``__init__`` is invoked
    with ``legacy_render_variant="egowm"`` so it builds a (throwaway,
    meta-device during ``from_pretrained``) ``render_encoder``; we
    immediately delete it and install ``self.action_adapter_v4``. Keeps
    model.py untouched.
    """

    _keep_in_fp32_modules = [
        "time_embedder",
        "time_proj",
        "action_adapter_v4",
        "tracks_head",
        "scale_shift_table",
        "norm_out",
    ]

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        render_latent_channels: int = 16,
        render_encoder_kwargs: Optional[Dict[str, Any]] = None,
        tracks_head_kwargs: Optional[Dict[str, Any]] = None,
        use_embodiment_adapter: bool = False,
        embodiment_kwargs: Optional[Dict[str, Any]] = None,
        legacy_render_variant: str = "v4",
        v2_adapter_kwargs: Optional[Dict[str, Any]] = None,
        v3_adapter_kwargs: Optional[Dict[str, Any]] = None,
        v4_adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Build the parent with egowm variant so it constructs a working
        # base Wan DiT + tracks_head. The egowm render_encoder is a
        # throwaway — replaced below.
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            render_latent_channels=render_latent_channels,
            render_encoder_kwargs=render_encoder_kwargs,
            tracks_head_kwargs=tracks_head_kwargs,
            use_embodiment_adapter=False,
            embodiment_kwargs=None,
            legacy_render_variant="egowm",
            v2_adapter_kwargs=None,
        )

        inner_dim = num_attention_heads * attention_head_dim

        # Drop the throwaway egowm render encoder; install the v4 adapter.
        if hasattr(self, "render_encoder"):
            del self.render_encoder
        self.legacy_render_variant = "v4"

        v4_kw = dict(v4_adapter_kwargs or {})
        if "action_dim" not in v4_kw:
            raise ValueError(
                "WanTransformerRenderConditionedV4 requires "
                "v4_adapter_kwargs.action_dim (e.g. 8 for joint+gripper)."
            )
        v4_kw.setdefault("inner_dim", inner_dim)
        v4_kw.setdefault("render_in_channels", render_latent_channels)
        v4_kw.setdefault("hidden_dim", 512)
        v4_kw.setdefault("num_heads", 8)
        v4_kw.setdefault("num_self_blocks", 2)
        v4_kw.setdefault("num_cross_blocks", 2)
        v4_kw.setdefault("spatial_downsample", 2)
        v4_kw.setdefault("ffn_mult", 4)
        v4_kw.setdefault("max_action_frames", 128)
        v4_kw.setdefault("max_t", 32)
        v4_kw.setdefault("max_h", 32)
        v4_kw.setdefault("max_w", 64)
        self.action_adapter_v4 = ActionRenderSelfCrossAdapterV4(**v4_kw)
        self.action_adapter_v4.to(torch.float32)

    # NOTE: no reset_zero_gates override needed. Parent's reset_zero_gates
    # falls through cleanly for lrv="v4" (neither v1 nor v2 branch fires)
    # and still zero-inits tracks_head.mlp[-1].

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        render_latents: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        query_xy: Optional[torch.Tensor] = None,
        track_T: Optional[int] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # No render conditioning → plain Wan I2V forward (skip the v4 path).
        if render_latents is None:
            if query_xy is not None:
                raise ValueError(
                    "query_xy provided but render_latents is None. The tracks "
                    "head requires the render-conditioned forward path."
                )
            return WanTransformerRenderConditioned.__bases__[0].forward(
                self,
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )

        if actions is None:
            raise ValueError(
                "legacy_render_variant='v4' requires `actions` "
                "(B, T_act, action_dim) but got None."
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        tokens_per_frame = post_patch_height * post_patch_width
        num_tokens = post_patch_num_frames * tokens_per_frame

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            timestep = timestep[:, 0]

        # Base time embedding + text/image conditioning (Wan 2.1 path).
        temb_base, _unused_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image,
                timestep_seq_len=None,
            )
        )  # temb_base: (B, inner_dim)

        ce = self.condition_embedder

        # ---- v4 conditioning -------------------------------------------------
        aa_dtype = next(self.action_adapter_v4.parameters()).dtype
        contribution = self.action_adapter_v4(
            render_latents.to(device=hidden_states.device, dtype=aa_dtype),
            actions.to(device=hidden_states.device, dtype=aa_dtype),
        )  # (B, num_tokens, inner_dim)
        if contribution.shape[1] != num_tokens:
            raise RuntimeError(
                f"ActionRenderSelfCrossAdapterV4 produced {contribution.shape[1]} "
                f"tokens but DiT expects {num_tokens}. Check spatial_downsample "
                f"matches Wan's patch_size."
            )
        contribution = contribution.to(dtype=temb_base.dtype)
        temb_per_token = self.action_adapter_v4.combine_with_temb(
            temb_base, contribution
        )  # (B, num_tokens, inner_dim)

        tp_in = ce.act_fn(temb_per_token.to(ce.time_proj.weight.dtype))
        timestep_proj = ce.time_proj(tp_in).to(dtype=temb_per_token.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        temb_for_norm_out = temb_per_token
        # ---------------------------------------------------------------------

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

        # norm_out modulation: per-token.
        shift, scale = (
            self.scale_shift_table.unsqueeze(0) + temb_for_norm_out.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)

        # Token grid for the auxiliary tracks head.
        token_grid = hidden_states.view(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )

        diffusion_hidden = self.proj_out(hidden_states)
        diffusion_hidden = diffusion_hidden.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, -1,
        )
        diffusion_hidden = diffusion_hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = diffusion_hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        pred_tracks: Optional[torch.Tensor] = None
        if query_xy is not None:
            T_video_eff = int(track_T) if track_T is not None else post_patch_num_frames
            pred_tracks = self.tracks_head(token_grid, query_xy, T_video=T_video_eff)

        if not return_dict:
            if pred_tracks is None:
                return (output,)
            return (output, pred_tracks)

        if pred_tracks is None:
            return Transformer2DModelOutput(sample=output)
        out = Transformer2DModelOutput(sample=output)
        out.pred_tracks = pred_tracks
        return out


__all__ = ["WanTransformerRenderConditionedV4"]
