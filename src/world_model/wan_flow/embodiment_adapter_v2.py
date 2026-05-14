"""
Action-conditioned embodiment adapter (v2).

Design vs. embodiment_adapter.py:

1. Per-token, no spatial pooling. The render encoder produces features at
   the DiT's post-patch spatial grid via a stride-2 conv, then each
   (t, h, w) location independently routes through cross-attention over
   actions. The contribution stays at per-token resolution (B, num_tokens, D)
   all the way through — no pool destroys spatial information. The DiT
   ends up with per-token modulation that retains "where the gripper is"
   structure, not a single per-frame summary.

2. Pre-addition to temb. The per-token contribution is added to
   ``temb_base`` *before* ``time_proj``, so the action signal flows through
   the same SiLU + Linear that shapes the timestep into modulation
   parameters. Post-addition straight into the 6*D modulation slot
   (the v1 ``DecoupledAdaLNHead`` path) tends to inject out-of-distribution
   values that destabilize early-training norms. ``combine_with_temb``
   broadcasts ``temb_base`` from per-batch / per-frame up to per-token
   before the add.

3. Magnitude-normalized. LayerNorm before the final projection puts the
   pre-projection features at unit variance; the model is free to learn
   any output scale via ``out_proj.weight``. No hand-set gate — input
   normalization is what bounds gradient magnitude during training.

4. Identity-at-init via zero-init final linear. ``out_proj.weight`` and
   ``out_proj.bias`` are zero, so ``contribution = 0`` on step 0 (vanilla
   Wan behaviour preserved bit-exact). Because LayerNorm produces non-zero
   features upstream, ``∂L/∂out_proj.weight`` is non-zero from step 1 and
   the projection ramps on its own — no gate needed. Standard DiT /
   ControlNet / AdaLN-zero pattern.

Wiring sketch (drop-in for the model forward)::

    contribution = adapter(render_latents, actions)              # (B, num_tokens, D)

    temb_base, _, ehs, ehs_img = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image,
        timestep_seq_len=T_post,    # or num_tokens; combine handles broadcast
    )                                                            # (B, T_post, D)
    temb = adapter.combine_with_temb(temb_base, contribution)    # (B, num_tokens, D)
    timestep_proj = self.condition_embedder.time_proj(
        self.condition_embedder.act_fn(temb)
    )                                                            # (B, num_tokens, 6*D)
    # consumer reshapes (B, num_tokens, 6*D) → (B, num_tokens, 6, D).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenderPerFrameEncoder(nn.Module):
    """(B, C, T, H, W) → (B, T, h*w, D), where ``h = H // spatial_downsample``,
    ``w = W // spatial_downsample``. The last conv has stride
    ``spatial_downsample`` so the encoder grid matches the DiT post-patch grid
    (Wan default ``patch_size=(1, 2, 2)`` ⇒ ``spatial_downsample=2``)."""

    def __init__(
        self,
        in_channels: int = 16,
        out_dim: int = 5120,
        hidden_dim: int = 512,
        spatial_downsample: int = 2,
    ) -> None:
        super().__init__()
        s = spatial_downsample
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(1, 3, 3),
                stride=(1, s, s),
                padding=(0, 1, 1),
            ),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        x = self.proj(render_latents)                          # (B, hidden, T, h, w)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, C)   # (B, T, h*w, hidden)
        return self.out(x)                                     # (B, T, h*w, D)


class ActionEncoder(nn.Module):
    """(B, T_act, action_dim) → (B, T_act, D)."""

    def __init__(
        self,
        action_dim: int,
        out_dim: int = 5120,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class ActionConditionedTembAdapter(nn.Module):
    """
    Per-(t, h, w) render tokens (Q) cross-attend over actions (K/V) →
    LayerNorm → zero-init Linear → pre-added to per-token temb.
    See module docstring for design rationale.
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        spatial_downsample: int = 2,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim

        self.render_encoder = RenderPerFrameEncoder(
            in_channels=render_in_channels,
            out_dim=inner_dim,
            hidden_dim=hidden_dim,
            spatial_downsample=spatial_downsample,
        )
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            out_dim=inner_dim,
            hidden_dim=hidden_dim,
        )

        self.q_norm = nn.LayerNorm(inner_dim)
        self.kv_norm = nn.LayerNorm(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            inner_dim, num_heads=num_heads, batch_first=True,
        )

        self.out_norm = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(inner_dim, inner_dim)

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: torch.Tensor,
        drop_render_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            contribution: (B, num_tokens, D), where ``num_tokens = T * h * w``
            and (h, w) match the DiT post-patch grid. Zero at init via
            zero-init ``out_proj``. Caller adds to ``temb_base`` via
            ``combine_with_temb`` (which broadcasts temb to per-token).
        """
        r = self.render_encoder(render_latents)        # (B, T, h*w, D)
        a = self.action_encoder(actions)               # (B, T_act, D)
        B, T, S, D = r.shape

        # Cross-attn: each (t, h, w) token independently attends over all
        # action frames. No spatial pooling — per-token signal is preserved
        # all the way to the contribution.
        r_flat = r.reshape(B, T * S, D)
        q = self.q_norm(r_flat)
        kv = self.kv_norm(a)
        enriched, _ = self.cross_attn(q, kv, kv, need_weights=False)   # (B, T*S, D)

        contribution = self.out_proj(self.out_norm(enriched))           # (B, T*S, D)

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
          - (B, D)            — scalar temb, broadcast to all tokens
          - (B, 1, D)         — same as above
          - (B, T, D)         — per-frame temb, repeat-interleaved by tokens/frame
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


@torch.no_grad()
def assert_identity_at_init(
    inner_dim: int = 5120,
    action_dim: int = 7,
    batch_size: int = 1,
    t_lat: int = 21,
    h_lat: int = 60,
    w_lat: int = 104,
    t_act: int = 81,
    spatial_downsample: int = 2,
) -> None:
    """Confirm the per-token contribution is exactly zero at step 0 and that
    pre-addition leaves temb bit-identical."""
    adapter = ActionConditionedTembAdapter(
        action_dim=action_dim, inner_dim=inner_dim, hidden_dim=128,
        spatial_downsample=spatial_downsample,
    ).eval()

    render = torch.randn(batch_size, 16, t_lat, h_lat, w_lat)
    actions = torch.randn(batch_size, t_act, action_dim)
    contribution = adapter(render, actions)

    expected_h = h_lat // spatial_downsample
    expected_w = w_lat // spatial_downsample
    expected_tokens = t_lat * expected_h * expected_w
    assert contribution.shape == (batch_size, expected_tokens, inner_dim), \
        f"contribution shape {contribution.shape} != expected " \
        f"({batch_size}, {expected_tokens}, {inner_dim})"
    assert torch.equal(contribution, torch.zeros_like(contribution)), \
        "contribution non-zero at init — out_proj zero-init failed"

    # (B, D) → broadcast to per-token
    temb_scalar = torch.randn(batch_size, inner_dim)
    temb = adapter.combine_with_temb(temb_scalar, contribution)
    expected = temb_scalar.unsqueeze(1).expand(-1, expected_tokens, -1)
    assert torch.equal(temb, expected), "(B, D) broadcast broke temb at init"

    # (B, T, D) → repeat-interleave by tokens/frame
    temb_per_frame = torch.randn(batch_size, t_lat, inner_dim)
    temb = adapter.combine_with_temb(temb_per_frame, contribution)
    expected = temb_per_frame.repeat_interleave(expected_h * expected_w, dim=1)
    assert torch.equal(temb, expected), "(B, T, D) repeat-interleave broke temb at init"

    print(f"OK identity-at-init: contribution shape ({batch_size}, "
          f"{expected_tokens}, {inner_dim}) = 0, temb unchanged.")


__all__ = [
    "RenderPerFrameEncoder",
    "ActionEncoder",
    "ActionConditionedTembAdapter",
    "assert_identity_at_init",
]
