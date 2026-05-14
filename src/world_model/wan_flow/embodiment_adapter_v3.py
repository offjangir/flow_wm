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

2. LayerNorm on BOTH streams before fusion, so render tokens and action
   tokens live in the same magnitude range going into cross-attention.

3. TRADITIONAL pre-norm transformer blocks (×N, default 2). Each block is
   ``x = x + cross_attn(LN(x), LN_kv(ctx))`` followed by
   ``x = x + ffn(LN(x))``. The residual connections give the gradient a
   stable path back into the cross-attn QKV that does NOT route through
   the final projection — this is the structural fix for the v2 failure.

4. Two-layer projection to temb (LayerNorm → Linear → GELU → Linear). All
   layers use standard PyTorch init (kaiming-uniform), same as egowm — NO
   small-init / zero-init scaling. egowm trains fine with a ~0.58-std
   contribution at init; v3 is strictly more stable than egowm (pre-norm +
   residual blocks tame the cross-attn gradient), so standard init is
   sufficient and there is no rank-1 trap to engineer around.

5. The per-token contribution is ADDED to ``temb_base`` BEFORE
   ``time_proj`` (same wiring as v2/egowm — ``combine_with_temb``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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

        # Bring both streams into the same magnitude range before fusion.
        self.render_norm = nn.LayerNorm(hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)

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

        r = self.render_norm(r)
        a = self.action_norm(a)

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


__all__ = [
    "RenderTokenEncoder",
    "ActionTokenEncoder",
    "CrossAttnTransformerBlock",
    "ActionRenderCrossAdapterV3",
]
