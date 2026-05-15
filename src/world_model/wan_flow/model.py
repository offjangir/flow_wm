"""
Wan 2.1 image-to-video with per-frame render-video conditioning (Diffusers).

At each timestep t, a DrRobot render of the robot executing the action is
VAE-encoded to produce render latents ``(B, C_vae, T_lat, H_lat, W_lat)``.
:class:`RenderLatentEncoder` compresses those to per-latent-frame vectors
``(B, T_lat, inner_dim)``; :class:`WanTransformerRenderConditioned` broadcasts
them to per-token and adds them to the base Wan time embedding ``Z_ts`` before
the AdaLN modulation projection (EgoWM Eq. 5 with ``Z_a = Z_render``).

The target video the DiT denoises is the *real* camera video; the render
latents act as per-frame action conditioning.

Weights layout: Hugging Face Diffusers Wan I2V repo with ``transformer/``,
``vae/``, ``text_encoder/``, ``tokenizer/``, ``image_encoder/``,
``image_processor/``, ``scheduler/``.

Historical note: this module is named ``wan_flow`` for backwards compatibility
with existing configs and CLI invocations. The optical-flow path it once hosted
has been removed in favour of render-video conditioning.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput, AutoencoderKLOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils.import_utils import is_torch_xla_available
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPVisionModel

from world_model.wan_flow.embodiment_adapter import (
    EmbodimentAgnosticConditioning,
    RenderSpatialEncoder,
)
from world_model.wan_flow.embodiment_adapter_v2 import (
    ActionConditionedTembAdapter,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = diffusers_logging.get_logger(__name__)


class WanVAEChunkedEncode(AutoencoderKLWan):
    """Wan VAE encode with temporal chunking (matches common Wan I2V usage)."""

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                chunk = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                chunk = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            out = chunk if out is None else torch.cat([out, chunk], dim=2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc


class RenderLatentEncoder(nn.Module):
    """
    Compress pre-computed render-video VAE latents ``(B, C_vae, T_lat, H_lat, W_lat)``
    to per-latent-frame embeddings ``(B, T_lat, out_dim)``. Two depthwise
    ``Conv3d`` layers (kernel ``1x3x3``) mix spatial features while keeping the
    temporal axis intact, then spatial average pooling + linear projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        if render_latents.ndim != 5:
            raise ValueError(
                f"render_latents must be (B, C, T_lat, H_lat, W_lat); got {tuple(render_latents.shape)}"
            )
        x = self.proj(render_latents)                # (B, H, T, Hl, Wl)
        x = self.pool(x).squeeze(-1).squeeze(-1)     # (B, H, T)
        x = x.transpose(1, 2)                        # (B, T, H)
        return self.out(x)                           # (B, T, out_dim)


def _sinusoidal_posenc(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Sin/cos positional encoding.

    ``x``: ``(..., D_in)`` with values in roughly ``[-1, 1]``.
    Returns ``(..., D_in * 2 * num_freqs)`` with the encoded coordinates
    interleaved as ``[sin(2^0*pi*x), cos(2^0*pi*x), sin(2^1*pi*x), ...]``.
    """
    fr = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * math.pi
    a = x.unsqueeze(-1) * fr                            # (..., D_in, num_freqs)
    return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)


class TracksHead(nn.Module):
    """Predict 2D point tracks from the DiT's post-blocks token grid.

    Given the modulated, post-blocks hidden states from
    :class:`WanTransformerRenderConditioned` (shape ``(B, T_lat, H_p, W_p, D)``,
    captured just before ``proj_out``) and a set of reference-frame query
    points ``query_xy`` (shape ``(B, N, 2)`` in normalized
    ``grid_sample`` coordinates ``[-1, 1]``), this head outputs the predicted
    trajectory of those points across all ``T_video`` output frames:

        pred_xy: (B, T_video, N, 2)  in normalized [-1, 1] xy.

    Architecture:
      1. For each latent frame ``t``, bilinearly sample the token grid at
         ``query_xy`` -> per-point feature ``f_{t,n} \in R^D`` of shape
         ``(B, T_lat, N, D)``.
      2. Concatenate sin/cos positional encodings of ``query_xy`` (broadcast
         across ``T_lat``) and of the latent frame index (broadcast across
         ``N``) to give the MLP both spatial and temporal positioning.
      3. MLP -> per-(latent_frame, point) delta_xy ``(B, T_lat, N, 2)``.
      4. Linear-interpolate temporally from ``T_lat`` to ``T_video`` ->
         delta_xy at video rate ``(B, T_video, N, 2)``.
      5. ``pred_xy = query_xy.unsqueeze(1) + delta_xy``.

    The final MLP layer is zero-initialized so ``delta_xy = 0`` at step 0,
    i.e. the model starts predicting *zero motion* (every frame's xy =
    query_xy). This warm-starts cleanly: only after training does the head
    begin to predict actual trajectories. Combined with the auxiliary loss
    ``smooth_l1(pred_xy, gt_xy) * vis_mask``, this enforces spatial
    consistency on the generated video.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_freqs_xy: int = 8,
        num_freqs_t: int = 6,
    ) -> None:
        super().__init__()
        self.num_freqs_xy = num_freqs_xy
        self.num_freqs_t = num_freqs_t
        xy_pe_dim = 2 * 2 * num_freqs_xy   # x and y, each -> 2*num_freqs_xy
        t_pe_dim = 2 * num_freqs_t         # t (scalar) -> 2*num_freqs_t
        feat_in = in_dim + xy_pe_dim + t_pe_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # Small Kaiming He init on final layer. Zero-init was too weak (no
        # tracks-gradient to penultimate features at step 0); pure Kaiming
        # would give initial pred std ~1.4 -> initial MSE ~2 -> dominates the
        # flow loss when paired with lambda_tracks=50. The 0.1 scale keeps
        # initial pred std ~0.14 -> initial MSE ~0.02 -> contribution ~1.0,
        # comparable to flow.
        nn.init.kaiming_normal_(self.mlp[-1].weight, nonlinearity="relu")
        self.mlp[-1].weight.data.mul_(0.1)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token_grid, query_xy, T_video=None, chunk_size: int = 16384):
        """
        Args:
            token_grid: (B, T_lat, H_p, W_p, D) latent feature grid from transformer blocks.
            query_xy: (B, N, 2) normalized query coordinates in [-1, 1].
            T_video: Final number of output frames (temporally upsampled).
            chunk_size: Max points to process in one pass to avoid OOM.
        Returns:
            trajs: (B, T_video, N, 2) predicted absolute coordinates in [-1, 1].
        """
        B, T_lat, H_p, W_p, D = token_grid.shape
        N = query_xy.shape[1]
        compute_dtype = token_grid.dtype
        T_video = T_video or T_lat

        # Process in chunks of N points if N is large (e.g. dense flow)
        all_d_video = []
        for i in range(0, N, chunk_size):
            q_chunk = query_xy[:, i : i + chunk_size] # (B, Nc, 2)
            Nc = q_chunk.shape[1]

            # ---- 1) bilinear sample features ----
            grid_btd = token_grid.permute(0, 1, 4, 2, 3).reshape(B * T_lat, D, H_p, W_p)
            sampling_grid = (
                q_chunk.unsqueeze(1)
                .expand(B, T_lat, Nc, 2)
                .reshape(B * T_lat, 1, Nc, 2)
            )
            sampled = F.grid_sample(
                grid_btd, sampling_grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            )  # (B*T_lat, D, 1, Nc)
            sampled = sampled.squeeze(2).permute(0, 2, 1)        # (B*T_lat, Nc, D)
            sampled = sampled.reshape(B, T_lat, Nc, D)

            # ---- 2) positional encodings ----
            xy_pe = _sinusoidal_posenc(q_chunk, self.num_freqs_xy)   # (B, Nc, 4*F_xy)
            xy_pe = xy_pe.unsqueeze(1).expand(B, T_lat, Nc, xy_pe.shape[-1])

            t_norm = (
                torch.arange(T_lat, device=token_grid.device, dtype=compute_dtype)
                / max(T_lat - 1, 1) * 2.0 - 1.0
            )
            t_pe = _sinusoidal_posenc(t_norm.unsqueeze(-1), self.num_freqs_t)
            t_pe = t_pe.view(1, T_lat, 1, t_pe.shape[-1]).expand(B, T_lat, Nc, -1)

            feat = torch.cat([sampled, xy_pe, t_pe], dim=-1)         # (B, T_lat, Nc, D')

            # ---- 3) MLP -> delta_xy at latent rate ----
            d_lat = self.mlp(feat)                                    # (B, T_lat, Nc, 2)

            # ---- 4) temporal upsample to T_video frames ----
            if T_video != T_lat:
                d_lat_for_interp = d_lat.permute(0, 2, 3, 1).reshape(B * Nc * 2, 1, T_lat)
                d_video = F.interpolate(
                    d_lat_for_interp, size=T_video, mode="linear", align_corners=True,
                )
                d_video = d_video.reshape(B, Nc, 2, T_video).permute(0, 3, 1, 2)
            else:
                d_video = d_lat

            all_d_video.append(d_video)

        # Recombine chunks
        d_video_all = torch.cat(all_d_video, dim=2) # (B, T_video, N, 2)

        # ---- 5) compute absolute trajs starting from query_xy ----
        trajs = query_xy.unsqueeze(1) + d_video_all
        return trajs


class WanTransformerRenderConditioned(WanTransformer3DModel):
    """Wan DiT with per-latent-frame render-video conditioning injected into the
    timestep pathway, following EgoWM (Bagchi et al., arXiv:2601.15284) Eq. 5::

        P_i^{scale, shift, gate} = F_i(Z_ts + Z_a)

    with ``Z_a = Z_render = render_fuse(render_encoder(render_latents))`` per
    latent frame. Per-frame ``Z_render`` is broadcast to per-token by tiling
    across the ``(p_h, p_w)`` spatial patch grid for each latent frame. This
    activates the Wan ``timestep_seq_len`` (Wan 2.2 TI2V) pathway for
    block-level AdaLN, giving each token frame-specific modulation parameters
    without modifying Diffusers internals.
    """

    _keep_in_fp32_modules = [
        "time_embedder",
        "time_proj",
        "render_encoder",
        "render_fuse",
        "embodiment",
        "action_adapter",
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
        # Selects which legacy variant to instantiate when
        # ``use_embodiment_adapter=False``:
        #   * "egowm"  (default): per-token spatial render encoder, no gate,
        #              no fuse — direct additive to temb. Mirrors EgoWM Eq. 5.
        #              No actions.
        #   * "v1"     (BACKWARD-COMPAT): pooled-per-frame render encoder +
        #              render_fuse Linear + scalar render_gate(=0.1) → adds
        #              to temb. Original variant; kept as an ablation baseline.
        #   * "v2"     (NEW, action-conditioned): ``ActionConditionedTembAdapter``
        #              from ``embodiment_adapter_v2``. Per-token render features
        #              cross-attend over the action trajectory; LayerNorm +
        #              zero-init Linear projects to inner_dim and the result is
        #              ADDED to temb_base BEFORE time_proj. Identity-at-init via
        #              zero-init out_proj (no gates). Requires ``actions`` in
        #              forward().
        legacy_render_variant: str = "egowm",
        # Kwargs forwarded to ``ActionConditionedTembAdapter`` when
        # ``legacy_render_variant="v2"`` (e.g. {"action_dim": 8, "hidden_dim":
        # 512, "num_heads": 8, "spatial_downsample": 2}). ``action_dim`` must
        # match the action-stream feature dim (8 for joint+gripper).
        v2_adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
        )
        inner_dim = num_attention_heads * attention_head_dim
        # ``use_embodiment_adapter`` switches between two conditioning paths:
        #   * False (legacy, default): EgoWM Eq. 5 — pooled per-frame render
        #     summary added to the time-pathway, modulating AdaLN globally
        #     per token. Cheap but loses spatial structure (see
        #     ARCHITECTURE_REVIEW.md §1.1).
        #   * True (recommended for scaling): spatial KV bank +
        #     per-block cross-attn adapters + decoupled AdaLN deltas
        #     (``EmbodimentAgnosticConditioning``). Identity-at-init.
        self.use_embodiment_adapter = bool(use_embodiment_adapter)

        if self.use_embodiment_adapter:
            emb_kw = dict(embodiment_kwargs or {})
            emb_kw.setdefault("inner_dim", inner_dim)
            emb_kw.setdefault("num_blocks", num_layers)
            emb_kw.setdefault("render_in_channels", render_latent_channels)
            self.embodiment = EmbodimentAgnosticConditioning(**emb_kw)
            self.embodiment.to(torch.float32)
        else:
            # Legacy path comes in three variants — see ``legacy_render_variant``
            # docstring on __init__ above.
            if legacy_render_variant not in ("egowm", "v1", "v2"):
                raise ValueError(
                    f"legacy_render_variant must be 'egowm', 'v1', or 'v2'; "
                    f"got {legacy_render_variant!r}"
                )
            self.legacy_render_variant = legacy_render_variant
            re_kw = dict(render_encoder_kwargs or {})
            if legacy_render_variant == "egowm":
                # Per-token spatial encoder + direct additive (no gate, no
                # fuse). Mirrors EgoWM (Bagchi et al. 2026, Eq. 5–6).
                # spatial_pool=2 makes the encoder grid (H_lat/2, W_lat/2)
                # match Wan's post-patch grid, so output (B, T_lat·h·w, D)
                # lines up token-for-token with the DiT.
                re_kw.setdefault("hidden_dim", 512)
                re_kw.setdefault("spatial_pool", 2)
                self.render_encoder = RenderSpatialEncoder(
                    in_channels=render_latent_channels,
                    out_dim=inner_dim,
                    **re_kw,
                )
                self.render_encoder.to(torch.float32)
            elif legacy_render_variant == "v2":
                # NEW: action-conditioned temb adapter. Per-token render
                # features cross-attend over actions, LayerNorm + zero-init
                # Linear → contribution → ADD to temb_base → time_proj.
                # No gates anywhere; identity-at-init via zero-init out_proj.
                v2_kw = dict(v2_adapter_kwargs or {})
                if "action_dim" not in v2_kw:
                    raise ValueError(
                        "legacy_render_variant='v2' requires "
                        "v2_adapter_kwargs.action_dim (e.g. 8 for "
                        "joint_position+gripper)."
                    )
                v2_kw.setdefault("inner_dim", inner_dim)
                v2_kw.setdefault("render_in_channels", render_latent_channels)
                v2_kw.setdefault("hidden_dim", 512)
                v2_kw.setdefault("num_heads", 8)
                v2_kw.setdefault("spatial_downsample", 2)
                self.action_adapter = ActionConditionedTembAdapter(**v2_kw)
                self.action_adapter.to(torch.float32)
            else:
                # v1: original pooled-per-frame encoder + Linear fuse +
                # scalar render_gate. Backward-compat with checkpoints
                # trained before the EgoWM-style refactor.
                self.render_encoder = RenderLatentEncoder(
                    in_channels=render_latent_channels, out_dim=inner_dim, **re_kw
                )
                self.render_fuse = nn.Linear(inner_dim, inner_dim)
                nn.init.zeros_(self.render_fuse.bias)
                # Scalar gate init=0.1 — see SNR notes in earlier comments.
                self.render_gate = nn.Parameter(torch.full((1,), 0.1))
                self.render_encoder.to(torch.float32)
                self.render_fuse.to(torch.float32)

        th_kw = dict(tracks_head_kwargs or {})
        self.tracks_head = TracksHead(in_dim=inner_dim, **th_kw)
        self.tracks_head.to(torch.float32)

    def reset_zero_gates(self) -> None:
        """Re-apply the identity-at-init zero-init on every gated pathway.

        ``_materialize_meta_submodules`` (used by both train.py and train_fsdp.py)
        replaces meta tensors via ``to_empty + reset_parameters``. ``reset_parameters``
        wipes our explicit zero-inits (e.g. ``DecoupledAdaLNHead.up`` →
        kaiming-uniform, ``MultiheadAttention.out_proj`` → xavier). Without
        re-zeroing, "vanilla Wan at step 0" is no longer preserved.

        Safe to call anytime; it only zeros tensors that the architecture
        guarantees should be zero at init.
        """
        with torch.no_grad():
            if self.use_embodiment_adapter:
                # Mirror EmbodimentAgnosticConditioning.__init__: kaiming-init
                # `up.weight` (NOT zero) so gradient can flow into both `up`
                # and the per-channel gate. Zeroing both creates the dual-zero
                # bootstrap deadlock that the constructor explicitly defends
                # against — and which `reset_parameters` (called by
                # `_materialize_meta_submodules`) would also undo. Identity-
                # at-init is preserved because the gate is zero.
                for _head in (self.embodiment.render_to_adaln,
                              self.embodiment.state_to_adaln):
                    nn.init.kaiming_uniform_(_head.up.weight, a=math.sqrt(5))
                    if _head.up.bias is not None:
                        nn.init.zeros_(_head.up.bias)
                self.embodiment.render_adaln_gate.zero_()
                self.embodiment.state_adaln_gate.zero_()
                for adapter in self.embodiment.adapters.values():
                    # KEEP attn.out_proj kaiming (don't zero) — same dual-zero
                    # reasoning as render_to_adaln/state_to_adaln above.
                    # Identity-at-init still holds because gate=0.
                    nn.init.kaiming_uniform_(
                        adapter.attn.out_proj.weight, a=math.sqrt(5)
                    )
                    if adapter.attn.out_proj.bias is not None:
                        nn.init.zeros_(adapter.attn.out_proj.bias)
                    # CRITICAL: nn.MultiheadAttention.in_proj_weight (the
                    # combined Q/K/V projection) is silently zero-init'd by
                    # diffusers `_init_weights` for any module not present in
                    # the loaded HF checkpoint — even though
                    # MultiheadAttention.reset_parameters() would xavier-init
                    # it. Without this re-xavier, attn(q, kv, kv) returns 0
                    # everywhere → residual=0 → ∂L/∂gate=0 → gate never moves.
                    # Verified empirically on 2026-05-06: in_proj_weight.norm
                    # = 0.0 at training step 0. Apply xavier here.
                    nn.init.xavier_uniform_(adapter.attn.in_proj_weight)
                    if adapter.attn.in_proj_bias is not None:
                        nn.init.zeros_(adapter.attn.in_proj_bias)
                    adapter.gate.zero_()
                # Same fix for state_encoder's attention (also flagged as
                # "newly initialized" by HF; in_proj_weight = 0 by default).
                if hasattr(self.embodiment, "state_encoder") and \
                        hasattr(self.embodiment.state_encoder, "attn"):
                    _se_attn = self.embodiment.state_encoder.attn
                    nn.init.xavier_uniform_(_se_attn.in_proj_weight)
                    if _se_attn.in_proj_bias is not None:
                        nn.init.zeros_(_se_attn.in_proj_bias)
                # If action-aware AdaLN uses cross-attention (action_dim != None),
                # its cross_attn also needs the same fix.
                if getattr(self.embodiment, "use_action_aware_adaln", False):
                    _aa = self.embodiment.action_aware_adaln
                    if getattr(_aa, "cross_attn", None) is not None:
                        nn.init.xavier_uniform_(_aa.cross_attn.in_proj_weight)
                        if _aa.cross_attn.in_proj_bias is not None:
                            nn.init.zeros_(_aa.cross_attn.in_proj_bias)
                if getattr(self.embodiment, "use_eq5_residual", False):
                    if hasattr(self.embodiment, "render_fuse_gate"):
                        self.embodiment.render_fuse_gate.zero_()
                if getattr(self.embodiment, "use_action_aware_adaln", False):
                    # Only zero the gate; KEEP `adaln_head.up` healthy
                    # (kaiming-uniform from FSDP's reset_parameters). Zeroing
                    # both creates the dual-zero gradient bootstrap bug —
                    # ∂L/∂gate ∝ head_output and ∂L/∂head ∝ gate, so if both
                    # are zero, no gradient ever flows and the pathway stays
                    # at identity forever.
                    self.embodiment.action_aware_adaln.gate.zero_()
            else:
                # Legacy paths:
                #  - "egowm": no gate, no fuse; nothing to reset
                #  - "v1": match the constructor (gate=0.1, fuse.bias=0)
                #  - "v2": zero-init out_proj (identity-at-init) + xavier on
                #          MHA in_proj_weight (HF/diffusers _init_weights
                #          silently zeros it; same trap as the embodiment
                #          adapter's adapters).
                lrv = getattr(self, "legacy_render_variant", "egowm")
                if lrv == "v1":
                    self.render_gate.fill_(0.1)
                    if self.render_fuse.bias is not None:
                        self.render_fuse.bias.zero_()
                elif lrv == "v2":
                    aa = self.action_adapter
                    # MultiheadAttention.in_proj_weight is silently zero-init'd
                    # by diffusers' `_init_weights` for any module not present
                    # in the loaded HF checkpoint, even though MHA's own
                    # reset_parameters would xavier it. Without this, the
                    # cross-attn returns 0 forever.
                    nn.init.xavier_uniform_(aa.cross_attn.in_proj_weight)
                    if aa.cross_attn.in_proj_bias is not None:
                        nn.init.zeros_(aa.cross_attn.in_proj_bias)
            # Tracks head's final layer: small Kaiming He init (see TracksHead.__init__).
            nn.init.kaiming_normal_(self.tracks_head.mlp[-1].weight, nonlinearity="relu")
            self.tracks_head.mlp[-1].weight.data.mul_(0.1)
            nn.init.zeros_(self.tracks_head.mlp[-1].bias)

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
        # ``query_xy`` (B, N, 2) in [-1, 1]: when provided, the auxiliary
        # ``tracks_head`` runs alongside the diffusion forward and the call
        # returns ``(sample, pred_tracks)`` instead of just ``sample``.
        # ``track_T`` is the number of *output video* frames to predict
        # (defaults to ``num_frames`` of ``hidden_states``).
        if render_latents is None:
            if query_xy is not None:
                raise ValueError(
                    "query_xy provided but render_latents is None. The tracks "
                    "head requires the render-conditioned forward path."
                )
            return super().forward(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

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

        # Base time embedding + text/image conditioning (Wan 2.1 path: scalar t per sample).
        temb_base, _unused_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=None
            )
        )  # temb_base: (B, inner_dim); _unused_proj discarded (re-derived below).

        ce = self.condition_embedder
        kv_bank: Optional[torch.Tensor] = None
        # ``temb_for_norm_out`` is what's added to ``self.scale_shift_table`` for
        # the FINAL norm_out modulation. Per-token in the legacy path (EgoWM
        # Eq. 5) and per-batch in the adapter path (vanilla Wan, since the
        # decoupled AdaLN delta lives inside the block loop).
        if self.use_embodiment_adapter:
            # New path: spatial KV bank + per-block cross-attn adapters +
            # decoupled AdaLN deltas. See embodiment_adapter.py for the
            # full design (and ARCHITECTURE_REVIEW.md §4.1 for rationale).
            mod_in = ce.act_fn(temb_base.to(ce.time_proj.weight.dtype))
            mod_base = ce.time_proj(mod_in).to(dtype=temb_base.dtype)  # (B, 6*D)

            re_dtype = next(self.embodiment.parameters()).dtype
            actions_cast = (
                actions.to(device=hidden_states.device, dtype=re_dtype)
                if actions is not None
                else None
            )
            cond = self.embodiment(
                render_latents.to(device=hidden_states.device, dtype=re_dtype),
                tokens_per_frame=tokens_per_frame,
                num_post_patch_frames=post_patch_num_frames,
                actions=actions_cast,
            )
            timestep_proj = self.embodiment.combine_modulation(mod_base, cond)
            # (B, num_tokens, 6, inner_dim) — at init this equals mod_base
            # broadcast, so vanilla Wan behaviour is preserved bit-exact.
            timestep_proj = timestep_proj.to(dtype=hidden_states.dtype)
            kv_bank = cond["kv_bank"].to(dtype=hidden_states.dtype)
            temb_for_norm_out = temb_base  # (B, D)
        else:
            lrv = getattr(self, "legacy_render_variant", "egowm")
            if lrv == "v2":
                # v2: per-token render features → cross_attn over actions →
                # LayerNorm → zero-init Linear → contribution → ADD to
                # temb_base → time_proj. No gates; identity-at-init via
                # zero-init out_proj. Requires `actions`.
                if actions is None:
                    raise ValueError(
                        "legacy_render_variant='v2' requires `actions` "
                        "(B, T_act, action_dim) but got None."
                    )
                aa_dtype = next(self.action_adapter.parameters()).dtype
                contribution = self.action_adapter(
                    render_latents.to(device=hidden_states.device, dtype=aa_dtype),
                    actions.to(device=hidden_states.device, dtype=aa_dtype),
                )                                            # (B, num_tokens, D)
                if contribution.shape[1] != num_tokens:
                    raise RuntimeError(
                        f"ActionConditionedTembAdapter produced "
                        f"{contribution.shape[1]} tokens but DiT expects "
                        f"{num_tokens}. Check spatial_downsample matches "
                        f"Wan's patch_size."
                    )
                contribution = contribution.to(dtype=temb_base.dtype)
                temb_per_token = self.action_adapter.combine_with_temb(
                    temb_base, contribution
                )                                            # (B, num_tokens, D)
            elif lrv == "egowm":
                # EgoWM-style: per-token spatial encoder → direct additive →
                # time_proj. No gate, no fuse Linear.
                re_dtype = next(self.render_encoder.parameters()).dtype
                r_tokens = self.render_encoder(
                    render_latents.to(device=hidden_states.device, dtype=re_dtype)
                )  # (B, num_tokens, inner_dim) when spatial_pool matches patch grid
                if r_tokens.shape[1] != num_tokens:
                    raise RuntimeError(
                        f"RenderSpatialEncoder produced {r_tokens.shape[1]} tokens but DiT "
                        f"expects {num_tokens}. Check spatial_pool matches Wan's patch_size."
                    )
                r_tokens = r_tokens.to(dtype=temb_base.dtype)
                temb_per_token = (
                    temb_base.unsqueeze(1).expand(-1, num_tokens, -1) + r_tokens
                )                                            # (B, num_tokens, D)
            else:
                # v1: pooled-per-frame encoder + Linear fuse + scalar gate.
                re_dtype = next(self.render_encoder.parameters()).dtype
                r_tokens = self.render_encoder(
                    render_latents.to(device=hidden_states.device, dtype=re_dtype)
                )  # (B, T_lat, inner_dim)
                if r_tokens.shape[1] != post_patch_num_frames:
                    r_tokens = torch.nn.functional.interpolate(
                        r_tokens.transpose(1, 2),
                        size=post_patch_num_frames,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                rf_dtype = self.render_fuse.weight.dtype
                z_render = self.render_fuse(r_tokens.to(dtype=rf_dtype))
                z_render = self.render_gate.to(dtype=z_render.dtype) * z_render
                z_render = z_render.to(dtype=temb_base.dtype)
                z_render_per_token = z_render.repeat_interleave(tokens_per_frame, dim=1)
                temb_per_token = (
                    temb_base.unsqueeze(1).expand(-1, num_tokens, -1)
                    + z_render_per_token
                )                                            # (B, num_tokens, D)

            tp_in = ce.act_fn(temb_per_token.to(ce.time_proj.weight.dtype))
            timestep_proj = ce.time_proj(tp_in).to(dtype=temb_per_token.dtype)
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
            temb_for_norm_out = temb_per_token  # (B, num_tokens, D)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if (
                    self.use_embodiment_adapter
                    and str(i) in self.embodiment.adapters
                    and kv_bank is not None
                ):
                    # NOT wrapping the adapter in `_gradient_checkpointing_func`:
                    # reentrant checkpointing (the diffusers default) silently
                    # zeros gradients to module parameters accessed via
                    # `self.<attr>` inside the checkpointed function. We
                    # observed `adapter.gate.grad` collapse to exactly 0
                    # across all training steps despite ∂L/∂gate being
                    # non-zero analytically. Calling the adapter directly
                    # restores correct gradient flow. Memory cost is
                    # negligible — the cross-attn over a ~5K KV bank is
                    # tiny compared to the WanTransformerBlock that's
                    # already checkpointed.
                    hidden_states = self.embodiment.adapters[str(i)](hidden_states, kv_bank)
        else:
            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                if (
                    self.use_embodiment_adapter
                    and str(i) in self.embodiment.adapters
                    and kv_bank is not None
                ):
                    hidden_states = self.embodiment.adapters[str(i)](hidden_states, kv_bank)

        if temb_for_norm_out.ndim == 3:
            # Per-token (legacy path).
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb_for_norm_out.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # Per-batch scalar (adapter path or vanilla Wan).
            shift, scale = (self.scale_shift_table + temb_for_norm_out.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        # Capture post-modulation, pre-proj_out hidden states as the "token grid"
        # for the auxiliary tracks head. Shape: (B, T_lat, H_p, W_p, inner_dim).
        # We keep this around even when the head is not invoked so the cost is
        # just one reshape (no copy). With p_t=1, post_patch_num_frames == T_lat.
        token_grid = hidden_states.view(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )

        diffusion_hidden = self.proj_out(hidden_states)
        diffusion_hidden = diffusion_hidden.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        diffusion_hidden = diffusion_hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = diffusion_hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        # Auxiliary track prediction (off by default; only when query_xy provided).
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
        # When tracks are requested we also return them; the diffusers
        # Transformer2DModelOutput dataclass doesn't carry a `tracks` field, so
        # callers should use return_dict=False to get the tuple.
        out = Transformer2DModelOutput(sample=output)
        out.pred_tracks = pred_tracks
        return out


class RenderConditionedWanDiffusion(nn.Module):
    def __init__(self, dit: WanTransformerRenderConditioned, vae, scheduler, image_processor, image_encoder):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def freeze_vae_and_image_encoder(self):
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        latent_model_input: torch.Tensor,
        render_latents: torch.Tensor,
        image_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        t: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ):
        return self.dit(
            hidden_states=latent_model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeddings,
            encoder_hidden_states_image=image_embeddings,
            render_latents=render_latents,
            actions=actions,
            return_dict=False,
        )[0]


def _normalize_latents(z: torch.Tensor, vae) -> torch.Tensor:
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    return (z - mean) * inv_std


class RenderConditionedWanI2VPipeline(WanImageToVideoPipeline):
    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
        render_latents=None,
    ):
        super().check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if render_latents is not None:
            if render_latents.ndim != 5:
                raise ValueError(
                    f"`render_latents` must be (B, C, T_lat, H_lat, W_lat), got shape {tuple(render_latents.shape)}"
                )

    def _encode_render_video(
        self,
        render_video,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Preprocess + VAE-encode a render video to normalized latents."""
        vid = self.video_processor.preprocess_video(render_video, height=height, width=width)
        if vid.shape[2] != num_frames:
            import numpy as np
            idxs = np.linspace(0, vid.shape[2] - 1, num_frames).round().astype(int)
            vid = vid[:, :, idxs]
        vid = vid.to(device=device, dtype=self.vae.dtype)
        z = self.vae.encode(vid).latent_dist.sample()
        z = _normalize_latents(z, self.vae)
        return z.to(dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        render_video: Optional[Any] = None,
        render_latents: Optional[torch.Tensor] = None,
        drop_render_conditioning: bool = False,
        actions: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        query_xy: Optional[torch.Tensor] = None,
        track_T: Optional[int] = None,
    ):
        if self.config.expand_timesteps or self.config.boundary_ratio is not None or self.transformer_2 is not None:
            raise NotImplementedError(
                "RenderConditionedWanI2VPipeline supports Wan 2.1-style I2V only "
                "(single transformer, expand_timesteps=False, boundary_ratio=None)."
            )
        if not drop_render_conditioning and render_latents is None and render_video is None:
            raise ValueError(
                "Pass `render_latents` (B, C, T_lat, H_lat, W_lat) or `render_video` (PIL/ndarray frames), "
                "or set `drop_render_conditioning=True` for vanilla I2V (no render branch)."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        eff_bs = batch_size * num_videos_per_prompt
        transformer_dtype = self.transformer.dtype

        if drop_render_conditioning:
            render_latents = None
        elif render_latents is None:
            render_latents = self._encode_render_video(
                render_video, height, width, num_frames, device, transformer_dtype
            )
        else:
            render_latents = render_latents.to(device=device, dtype=transformer_dtype)

        if render_latents is not None:
            if render_latents.shape[0] == 1 and eff_bs > 1:
                render_latents = render_latents.expand(eff_bs, -1, -1, -1, -1)
            elif render_latents.shape[0] != eff_bs:
                raise ValueError(
                    f"render_latents batch {render_latents.shape[0]} != batch_size*num_videos_per_prompt={eff_bs}"
                )

        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
            render_latents,
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Defensive: encode_prompt only auto-expands by num_videos_per_prompt when
        # it computes the embeddings itself. If the caller pre-supplied
        # ``prompt_embeds`` with batch != eff_bs, we'd silently mismatch
        # ``latents`` (which is shaped to eff_bs). Fail early with a clear msg.
        if prompt_embeds.shape[0] != eff_bs:
            raise ValueError(
                f"prompt_embeds batch {prompt_embeds.shape[0]} != eff_bs={eff_bs} "
                f"(batch_size*num_videos_per_prompt). When pre-supplying "
                f"`prompt_embeds`, expand it to batch={eff_bs} before calling."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != eff_bs:
            raise ValueError(
                f"negative_prompt_embeds batch {negative_prompt_embeds.shape[0]} "
                f"!= eff_bs={eff_bs}. When pre-supplying "
                f"`negative_prompt_embeds`, expand it to batch={eff_bs} before calling."
            )

        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            # Match the batch dim to ``eff_bs = batch_size * num_videos_per_prompt``.
            # ``encode_image`` returns batch=1 (or 2 in FLF mode) — the rest of the
            # pipeline (latents, prompt_embeds, render_latents) is shaped against
            # ``eff_bs``, so ``image_embeds`` must agree to avoid a batch mismatch
            # at the encoder_hidden_states concat in the transformer.
            cur = image_embeds.shape[0]
            if cur == 1:
                image_embeds = image_embeds.repeat(eff_bs, 1, 1)
            elif cur != eff_bs:
                if eff_bs % cur != 0:
                    raise ValueError(
                        f"image_embeds batch {cur} cannot be broadcast to "
                        f"eff_bs={eff_bs} (batch_size*num_videos_per_prompt)."
                    )
                image_embeds = image_embeds.repeat(eff_bs // cur, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents, condition = self.prepare_latents(
            image,
            eff_bs,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )

        # Match training (train.py / train_fsdp.py): the first *temporal* latent slice
        # of the 16-channel denoising tensor is always the clean VAE(first_frame) —
        # `noisy_latents[:, :, 0:1] = clean_latents[:, :, 0:1]` before every forward.
        # Official `prepare_latents` leaves slice 0 as noise and never re-pins it after
        # `scheduler.step`, which is out-of-distribution for this finetuning recipe and
        # commonly collapses motion (static / first-frame-like video). Encode first frame
        # + zero tail (same layout as the I2V condition video) and lock slice 0.
        img_t = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
        b_img = img_t.shape[0]
        if b_img == 1 and eff_bs > 1:
            img_t = img_t.expand(eff_bs, -1, -1, -1, -1)
        elif b_img != eff_bs:
            raise ValueError(
                f"preprocessed image batch {b_img} must be 1 (broadcast to eff_bs={eff_bs}) "
                f"or equal to eff_bs."
            )
        vid_pad = torch.cat(
            [img_t, img_t.new_zeros(eff_bs, img_t.shape[1], num_frames - 1, height, width)],
            dim=2,
        )
        z_first = self.vae.encode(vid_pad).latent_dist.sample()
        latents_first = _normalize_latents(z_first, self.vae).to(dtype=latents.dtype)[:, :, 0:1, :, :].contiguous()
        latents[:, :, 0:1] = latents_first

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                model_out = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    render_latents=render_latents,
                    actions=actions,
                    query_xy=query_xy if (query_xy is not None and i == len(timesteps) - 1) else None,
                    track_T=track_T,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )
                noise_pred = model_out[0]
                if i == len(timesteps) - 1 and len(model_out) > 1:
                    pred_tracks = model_out[1]
                else:
                    pred_tracks = None

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        render_latents=render_latents,
                        actions=actions,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                latents[:, :, 0:1] = latents_first

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            if pred_tracks is not None:
                return (video, pred_tracks)
            return (video,)

        out = WanPipelineOutput(frames=video)
        if pred_tracks is not None:
            out.pred_tracks = pred_tracks
        return out


def build_render_conditioned_wan_i2v(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    render_encoder_kwargs: Optional[Dict[str, Any]] = None,
    return_pipeline: bool = False,
) -> Union[RenderConditionedWanDiffusion, Tuple[RenderConditionedWanDiffusion, RenderConditionedWanI2VPipeline]]:
    """
    Load Wan I2V and return ``RenderConditionedWanDiffusion`` for training /
    custom sampling.

    ``model_path``: local directory containing ``transformer/``, ``vae/``, etc.,
    **or** a Hugging Face model id (e.g. ``Wan-AI/Wan2.1-I2V-14B-480P-Diffusers``).

    If ``return_pipeline=True``, returns ``(module, pipeline)`` so you can call
    ``encode_prompt``, ``prepare_latents``, etc. (training scripts).
    """
    root = model_path.rstrip("/")
    re_kw = dict(render_encoder_kwargs or {})
    local_transformer = os.path.isdir(os.path.join(root, "transformer"))
    transformer_kw = dict(torch_dtype=torch_dtype, render_encoder_kwargs=re_kw)
    if local_transformer:
        transformer = WanTransformerRenderConditioned.from_pretrained(
            os.path.join(root, "transformer"), **transformer_kw
        )
    else:
        transformer = WanTransformerRenderConditioned.from_pretrained(
            root, subfolder="transformer", **transformer_kw
        )

    image_encoder = CLIPVisionModel.from_pretrained(
        root, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = WanVAEChunkedEncode.from_pretrained(root, subfolder="vae", torch_dtype=torch_dtype)
    pipe = RenderConditionedWanI2VPipeline.from_pretrained(
        root,
        transformer=transformer,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch_dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
    module = RenderConditionedWanDiffusion(
        dit=transformer,
        vae=pipe.vae,
        scheduler=scheduler,
        image_processor=pipe.image_processor,
        image_encoder=pipe.image_encoder,
    )
    if return_pipeline:
        return module, pipe
    return module
