"""
Action-conditioned embodiment adapter (v3).

Design vs. v2 (embodiment_adapter_v2.py):

v2 was a single bare cross-attention (q_norm/kv_norm → MHA → out_norm →
out_proj) with NO residual connections and NO FFN. That made the gradient
path into the cross-attn QKV depend entirely on the magnitude of out_proj:
zero-init out_proj → rank-1 collapse (QKV welded at xavier); kaiming-init
out_proj → the cross-attn's softmax non-linearity amplified early-training
updates and the loss diverged. See memory:project_v2_adapter_collapse.

v3 fixes this structurally:

1. Two-layer encoders that preserve structure.
   - RenderTokenEncoder: 3 Conv3d layers, GELU. Keeps per-token spatial
     structure (stride-2 on the last conv only, to match Wan's post-patch
     grid). Does NOT pool to a per-frame summary.
   - ActionTokenEncoder: 2 Linear layers, GELU. Per-frame action tokens.
   GELU (not ReLU) so negative activations are not hard-zeroed.

2. TRADITIONAL pre-norm transformer blocks (×N, default 2). Each block is
   ``x = x + cross_attn(LN(x), LN_kv(ctx))`` followed by
   ``x = x + ffn(LN(x))``. The residual connections give the gradient a
   stable path back into the cross-attn QKV that does NOT route through
   the final projection — this is the structural fix for the v2 failure.

3. Two-layer projection to temb (LayerNorm → Linear → GELU → Linear). All
   layers use standard PyTorch init (kaiming-uniform), same as egowm — NO
   small-init / zero-init scaling. egowm trains fine with a ~0.58-std
   contribution at init; v3 is strictly more stable than egowm (pre-norm +
   residual blocks tame the cross-attn gradient), so standard init is
   sufficient and there is no rank-1 trap to engineer around.

4. The per-token contribution is ADDED to ``temb_base`` BEFORE
   ``time_proj`` (same wiring as v2/egowm — ``combine_with_temb``).

No pre-block LayerNorm on the streams: the transformer blocks themselves
are pre-norm (LN inside, before each attention/FFN), so an extra
LayerNorm right after the encoders would be redundant. The blocks'
internal norms already give the attention a per-token-normalized input.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenderTokenEncoder(nn.Module):
    """(B, C, T, H, W) -> (B, T*h*w, hidden), 3 Conv3d layers + GELU.

    ``h = H // spatial_downsample``, ``w = W // spatial_downsample``. Only the
    last conv strides; the first two preserve resolution so spatial structure
    is built up before it is downsampled. spatial_downsample=2 matches Wan's
    patch_size=(1, 2, 2) so output tokens line up 1:1 with the DiT post-patch
    grid.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 512,
        spatial_downsample: int = 2,
    ) -> None:
        super().__init__()
        s = spatial_downsample
        mid = hidden_dim // 2
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, mid, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(mid, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(1, 3, 3),
                stride=(1, s, s),
                padding=(0, 1, 1),
            ),
            nn.GELU(),
        )

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        x = self.proj(render_latents)                          # (B, hidden, T, h, w)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)  # (B, T*h*w, hidden)
        return x


class ActionTokenEncoder(nn.Module):
    """(B, T_act, action_dim) -> (B, T_act, hidden), 2 Linear layers + GELU.

    GELU (not ReLU) so small/negative action features are not hard-zeroed —
    the action stream is low-dimensional (8-d) and we cannot afford to
    destroy half of it at the first layer.
    """

    def __init__(self, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class CrossAttnTransformerBlock(nn.Module):
    """Traditional pre-norm transformer block: residual cross-attention +
    residual FFN.

    Query stream = render tokens; key/value = action tokens. The two
    residual connections are the structural fix for v2's divergence: the
    gradient reaching the cross-attn QKV no longer depends on the magnitude
    of any single downstream projection.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = self.norm_kv(context)
        attn_out, _ = self.cross_attn(
            self.norm_q(x), ctx, ctx, need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ActionRenderCrossAdapterV3(nn.Module):
    """
    render tokens (Q) cross-attend over action tokens (K/V) through N
    traditional transformer blocks, then a 2-layer projection lifts the
    fused per-token features to ``inner_dim`` and the result is ADDED to
    ``temb_base`` before ``time_proj``.

    All layers use standard PyTorch init (kaiming-uniform) — no small-init or
    zero-init scaling. The pre-norm + residual structure of the transformer
    blocks is what keeps the cross-attn gradient stable; identity-at-init is
    not relied upon (egowm doesn't have it either and trains fine).
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 2,
        spatial_downsample: int = 2,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim

        self.render_encoder = RenderTokenEncoder(
            in_channels=render_in_channels,
            hidden_dim=hidden_dim,
            spatial_downsample=spatial_downsample,
        )
        self.action_encoder = ActionTokenEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # No pre-block LayerNorm on either stream: the transformer blocks are
        # themselves pre-norm (norm_q/norm_kv inside), so the per-token
        # normalization needed by attention happens there. An extra pre-block
        # LN here would just be applied twice in a row on the first block's
        # inputs.

        self.blocks = nn.ModuleList([
            CrossAttnTransformerBlock(hidden_dim, num_heads, ffn_mult)
            for _ in range(num_blocks)
        ])

        # 2-layer projection from the fused hidden space up to temb space.
        # Standard PyTorch init throughout — no post-hoc scaling.
        self.to_temb = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, inner_dim),
        )

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: torch.Tensor,
        drop_render_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            contribution: (B, num_tokens, inner_dim), where
            ``num_tokens = T * h * w`` matches the DiT post-patch grid.
            Caller adds it to ``temb_base`` via ``combine_with_temb``.
        """
        r = self.render_encoder(render_latents)        # (B, T*h*w, hidden)
        a = self.action_encoder(actions)               # (B, T_act, hidden)

        for block in self.blocks:
            r = block(r, a)                            # render attends over actions

        contribution = self.to_temb(r)                 # (B, T*h*w, inner_dim)

        if drop_render_mask is not None:
            keep = drop_render_mask.view(-1, 1, 1).to(contribution.dtype)
            contribution = contribution * keep

        return contribution

    @staticmethod
    def combine_with_temb(
        temb_base: torch.Tensor,
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-add the per-token contribution to temb. Broadcasts ``temb_base``
        up to per-token from any of:
          - (B, D)             — scalar temb, broadcast to all tokens
          - (B, 1, D)          — same as above
          - (B, T, D)          — per-frame temb, repeat-interleaved by tokens/frame
          - (B, num_tokens, D) — already per-token, no broadcast

        ``contribution`` is (B, num_tokens, D).
        """
        num_tokens = contribution.shape[1]
        if temb_base.ndim == 2:
            temb_base = temb_base.unsqueeze(1).expand(-1, num_tokens, -1)
        elif temb_base.shape[1] == 1 and num_tokens != 1:
            temb_base = temb_base.expand(-1, num_tokens, -1)
        elif temb_base.shape[1] != num_tokens:
            tokens_per_frame, rem = divmod(num_tokens, temb_base.shape[1])
            if rem != 0:
                raise ValueError(
                    f"contribution has {num_tokens} tokens but temb_base has "
                    f"{temb_base.shape[1]} frames; not an integer multiple."
                )
            temb_base = temb_base.repeat_interleave(tokens_per_frame, dim=1)
        return temb_base + contribution


class SelfAttnTransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + residual + FFN.

    Mirrors CrossAttnTransformerBlock structure but Q, K, V all come from
    the same input. Used inside RenderTokenEncoderWithSelfAttn to let
    spatially-distant render tokens exchange information — critical when
    most tokens are constant-background and only a small fraction carry
    arm-pose information.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm_attn(x)
        attn_out, _ = self.self_attn(x_n, x_n, x_n, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PositionalEncoding3D(nn.Module):
    """Factorized 3D learned positional encoding.

    For a grid of (T, H, W) tokens with feature dim D, returns a
    (T*H*W, D) embedding such that ``pos[t, h, w] = pos_t[t] + pos_h[h] +
    pos_w[w]``. Factorization keeps the parameter cost at
    O(max_t + max_h + max_w) instead of O(max_t · max_h · max_w), and is
    sufficient for our regular cuboid token grid.
    """

    def __init__(self, max_t: int, max_h: int, max_w: int, dim: int) -> None:
        super().__init__()
        self.pos_t = nn.Parameter(torch.zeros(max_t, dim))
        self.pos_h = nn.Parameter(torch.zeros(max_h, dim))
        self.pos_w = nn.Parameter(torch.zeros(max_w, dim))
        nn.init.normal_(self.pos_t, std=0.02)
        nn.init.normal_(self.pos_h, std=0.02)
        nn.init.normal_(self.pos_w, std=0.02)

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        """Returns (T*H*W, dim)."""
        t = self.pos_t[:T].unsqueeze(1).unsqueeze(2)    # (T, 1, 1, dim)
        h = self.pos_h[:H].unsqueeze(0).unsqueeze(2)    # (1, H, 1, dim)
        w = self.pos_w[:W].unsqueeze(0).unsqueeze(1)    # (1, 1, W, dim)
        return (t + h + w).reshape(T * H * W, -1)


class RenderTokenEncoderWithSelfAttn(nn.Module):
    """Render encoder for the v4 (self-attn + cross-attn) pathway.

    Same Conv3d → reshape pipeline as RenderTokenEncoder, with three
    architectural upgrades targeting the diagnosed "mostly-black canvas"
    failure mode:

    1. **Pre-encoder mean subtraction on the VAE latents.** Before the
       first Conv3d, subtract the per-scene per-channel spatial-temporal
       mean from `render_latents`. For DrRobot renders this is
       essentially `f_VAE(empty canvas)` because ~95% of positions are
       black. After subtraction every position represents "deviation
       from the empty baseline" — arm-region positions retain their
       full pose-encoding signal, background positions collapse to ~0.
       The conv encoder then learns from arm-signal-only, not from
       "shared bias + small deviation."

    2. **Temporal mixing in the middle conv.** Middle conv kernel is
       changed from (1, 3, 3) to (3, 3, 3) (padding=(1,1,1)) so each
       spatial position can see 3 adjacent frames of temporal context.
       The first and last convs stay (1, 3, 3) — the first preserves
       the raw VAE temporal structure; the last is strided and handles
       only spatial downsampling.

    3. **Self-attention blocks on per-token features.** N pre-norm
       self-attention transformer blocks (residual + FFN) let every
       render token exchange information with every other token.
       Arm-region tokens can now propagate their pose-encoding
       information to background tokens (mediated by learned attention
       weights). Load-bearing fix: the conv stack alone has a 7×7
       receptive field, but most background tokens are too far from
       arm tokens to see them through convs.

    Output shape matches RenderTokenEncoder for drop-in compatibility.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 512,
        spatial_downsample: int = 2,
        num_self_blocks: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        max_t: int = 32,
        max_h: int = 32,
        max_w: int = 64,
    ) -> None:
        super().__init__()
        s = spatial_downsample
        mid = hidden_dim // 2
        # Individual Conv3d modules (not Sequential) so we can manually pad
        # the temporal dim of the middle conv with replicate-mode instead
        # of zero-mode — zero padding at the temporal boundaries would tell
        # the encoder "nothing existed before/after the clip," replicate
        # tells it "the boundary frame extends."
        self.conv1 = nn.Conv3d(in_channels, mid, (1, 3, 3), padding=(0, 1, 1))
        # Middle conv: kernel (3,3,3). Spatial padding=1 baked in (zeros
        # are fine for renders since "outside the frame" is black anyway);
        # temporal padding is done manually with replicate mode in forward.
        self.conv2 = nn.Conv3d(mid, hidden_dim, (3, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(
            hidden_dim, hidden_dim,
            kernel_size=(1, 3, 3),
            stride=(1, s, s),
            padding=(0, 1, 1),
        )
        # 3D positional encoding added after reshape to per-token sequence.
        # Each (t, h, w) coordinate gets a learned 512-d embedding (factored
        # as pos_t + pos_h + pos_w). Lets the self-attention and downstream
        # cross-attention reason about positions explicitly instead of
        # treating tokens as an unordered set.
        self.pos_enc = PositionalEncoding3D(max_t, max_h, max_w, hidden_dim)
        self.self_attn_blocks = nn.ModuleList([
            SelfAttnTransformerBlock(hidden_dim, num_heads, ffn_mult)
            for _ in range(num_self_blocks)
        ])

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        # Subtract per-scene spatial-temporal mean from the VAE latents
        # themselves, per channel. For DrRobot renders (mostly-black canvas
        # with a small arm region) the spatial mean is ~f_VAE(empty
        # canvas) — subtracting it leaves only the per-scene arm-pose
        # deviation as input. The conv encoder then operates on clean
        # arm-signal from step 0 instead of wasting capacity learning
        # around the shared baseline.
        render_latents = render_latents - render_latents.mean(
            dim=(2, 3, 4), keepdim=True
        )

        # Conv1: spatial-only kernel; zero padding is fine (image is bounded
        # by black anyway).
        x = F.gelu(self.conv1(render_latents))

        # Conv2: temporal mixing. Replicate-pad T by 1 on each side so the
        # boundary frames see their own value as their temporal neighbor
        # instead of a fictitious zero frame. F.pad arg order is reverse-
        # dim: (W_left, W_right, H_left, H_right, T_left, T_right).
        x = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        x = F.gelu(self.conv2(x))

        # Conv3: strided spatial downsample.
        x = F.gelu(self.conv3(x))

        B, C, T, H, W = x.shape                                # T=T_lat, H=h, W=w
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)  # (B, T·h·w, hidden)

        # Add 3D positional encoding. Broadcast over batch dim.
        x = x + self.pos_enc(T, H, W).unsqueeze(0)

        for block in self.self_attn_blocks:
            x = block(x)
        return x


class ActionRenderSelfCrossAdapterV4(nn.Module):
    """v4 adapter: render self-attention BEFORE action cross-attention.

    Building on v3 (cross-attn only), v4 adds two changes targeting the
    diagnostic findings from `scripts/diagnose_egowm_collapse.py` and the
    render-latent analysis (96% of egowm contributions point in the same
    direction; 30% of render-latent positions are constant-black; only
    4% are the genuinely-varying arm region):

    1. **Self-attention + temporal mixing on render tokens.**
       RenderTokenEncoderWithSelfAttn lets the few informative arm tokens
       propagate their signal across all spatial positions BEFORE the
       cross-attention with actions kicks in.

    2. **Pre-encoder mean subtraction on VAE latents.** Inside the render
       encoder, subtract the per-scene per-channel spatial-temporal mean
       from the VAE latents before the first Conv3d. The mean
       approximates `f_VAE(empty canvas)` since ~95% of positions in a
       DrRobot render are black. The conv encoder then learns from
       arm-signal deviations only, not from "shared bias + small
       deviation." Downstream layers (cross-attn, to_temb) keep their
       bias terms free — no late constraint on the output's mean.

    Wiring is identical to v3: contribution is added to ``temb_base``
    before ``time_proj`` via :meth:`combine_with_temb` (delegated to v3).
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_self_blocks: int = 2,
        num_cross_blocks: int = 2,
        spatial_downsample: int = 2,
        ffn_mult: int = 4,
        max_action_frames: int = 128,
        max_t: int = 32,
        max_h: int = 32,
        max_w: int = 64,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim

        self.render_encoder = RenderTokenEncoderWithSelfAttn(
            in_channels=render_in_channels,
            hidden_dim=hidden_dim,
            spatial_downsample=spatial_downsample,
            num_self_blocks=num_self_blocks,
            num_heads=num_heads,
            ffn_mult=ffn_mult,
            max_t=max_t,
            max_h=max_h,
            max_w=max_w,
        )
        self.action_encoder = ActionTokenEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # 1D positional encoding for action tokens. Without this, the cross-
        # attention sees action tokens as an unordered set and cannot
        # distinguish "action at frame 0" from "action at frame 10," so it
        # could degenerate to uniform attention over all actions.
        self.action_pos_emb = nn.Parameter(torch.zeros(max_action_frames, hidden_dim))
        nn.init.normal_(self.action_pos_emb, std=0.02)

        # Cross-attention blocks (render Q, action K/V). Same as v3.
        self.cross_blocks = nn.ModuleList([
            CrossAttnTransformerBlock(hidden_dim, num_heads, ffn_mult)
            for _ in range(num_cross_blocks)
        ])

        # 2-layer projection from the fused hidden space up to temb space.
        # Standard PyTorch init throughout, identical to v3's to_temb.
        self.to_temb = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, inner_dim),
        )

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: torch.Tensor,
        drop_render_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            contribution: (B, num_tokens, inner_dim), zero-mean across the
            spatial-token axis.
        """
        r = self.render_encoder(render_latents)        # (B, T·h·w, hidden) — self-attn'd
        a = self.action_encoder(actions)               # (B, T_act, hidden)

        # Add per-frame positional encoding to action tokens so cross-
        # attention can reason about temporal alignment ("render at frame
        # t attends to action at frame t").
        T_act = a.shape[1]
        a = a + self.action_pos_emb[:T_act].unsqueeze(0)

        for block in self.cross_blocks:
            r = block(r, a)                            # render attends over actions

        contribution = self.to_temb(r)                 # (B, T·h·w, inner_dim)

        if drop_render_mask is not None:
            keep = drop_render_mask.view(-1, 1, 1).to(contribution.dtype)
            contribution = contribution * keep

        return contribution

    @staticmethod
    def combine_with_temb(
        temb_base: torch.Tensor,
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        """Delegates to v3's combine_with_temb — identical wiring."""
        return ActionRenderCrossAdapterV3.combine_with_temb(temb_base, contribution)


__all__ = [
    "RenderTokenEncoder",
    "RenderTokenEncoderWithSelfAttn",
    "ActionTokenEncoder",
    "CrossAttnTransformerBlock",
    "SelfAttnTransformerBlock",
    "PositionalEncoding3D",
    "ActionRenderCrossAdapterV3",
    "ActionRenderSelfCrossAdapterV4",
]
