"""
Action-and-render concat adapter (v3).

Design philosophy:
  Cross-attention with actions as V (v2) collapses output diversity into
  span(V·W_v). Because actions are 8-d → 5120-d through an MLP, span(V) is a
  low-rank manifold and out_proj's gradient locks onto a single shared
  direction across all scenes — producing the universal tint we observed
  (cosine 0.99 between per-scene contribution means, R=13 ratio of
  shared/per-scene magnitude).

  v3 fixes this by NEVER routing render through a mixing operation whose
  output is constrained to span(actions). Instead, each per-(t, h, w) token
  is the concatenation of its own render feature and the broadcast action
  feature for its frame, then an MLP fuses them. Each token's output is a
  free function of its own (r, a) pair — no softmax-weighted average, no V-
  span constraint. Render's per-scene/per-token diversity flows through with
  full rank.

Pipeline::

    render_latents  (B, C_vae, T_lat, H_lat, W_lat)
        └─ Conv3d × 3 (last stride=spatial_downsample) → spatial features
        └─ Linear(hidden_dim)                           → r (B, T_lat, h·w, hidden_dim)

    actions         (B, T_act, action_dim)
        └─ MLP(action_dim → hidden_dim)                 → a (B, T_act, hidden_dim)
        └─ avg-pool / interp T_act → T_lat              → a (B, T_lat, hidden_dim)

    a_per_token = a.unsqueeze(2).expand(-1, -1, h·w, -1)  # broadcast per spatial token
    fused = concat([r, a_per_token], dim=-1)               # (B, T_lat, h·w, 2·hidden_dim)
    contribution = fusion_mlp(fused)                        # (B, T_lat·h·w, inner_dim)

  Identity-at-init: ``out_proj.weight`` and ``out_proj.bias`` are zero, so
  contribution = 0 on step 0 — vanilla Wan output is preserved bit-exact.
  Because LayerNorm sits before out_proj and its input is the per-token
  fused vector (always non-zero, varies per scene), ``∂L/∂out_proj.weight``
  is non-zero from step 1 — gradient flows without the dual-zero deadlock.

Why this avoids the v2 bias-trap:
  - For two different scenes, r_features differ (renders differ pixel-wise).
  - Concatenated input ``[r ‖ a]`` therefore differs per scene per token.
  - Fusion MLP literally cannot output the same vector for two different
    scenes unless r matches across scenes — which it doesn't.
  - The single-shared-direction attractor that v2 fell into is structurally
    unreachable here.

Approximate param budget (action_dim=8, hidden_dim=512, inner_dim=5120):
  render_encoder (3 Conv3d + Linear)   ~3.0 M
  action_encoder (3-layer MLP)         ~0.5 M
  fusion_mlp (LN + Linear + Linear)    ~3.7 M
                                       ─────
  total                                ~7.2 M
  (vs v2's 141 M — ~20× smaller; same order of magnitude as egowm's 5 M.)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenderPerFrameEncoderV3(nn.Module):
    """(B, C, T, H, W) → (B, T, h·w, hidden_dim).

    The last conv has stride ``spatial_downsample`` so the encoder grid
    matches the DiT post-patch grid (Wan default ``patch_size=(1, 2, 2)``
    ⇒ ``spatial_downsample=2``). Output is per-(t, h, w) features that
    carry the render's full per-token spatial diversity.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_dim: int = 512,
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
        x = self.proj(render_latents)                       # (B, hidden, T, h, w)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, C)
        return self.out(x)                                  # (B, T, h·w, out_dim)


class ActionEncoderV3(nn.Module):
    """(B, T_act, action_dim) → (B, T_act, out_dim).

    Simple 3-layer MLP with SiLU. action_dim is 8 by default (joint+gripper).
    The MLP expands to ``hidden_dim`` internally; output is ``out_dim`` so
    it matches the render encoder's output for concatenation.
    """

    def __init__(
        self,
        action_dim: int,
        out_dim: int = 512,
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


class ActionRenderConcatAdapter(nn.Module):
    """
    Per-token concat-and-MLP fusion of render + action features. Avoids the
    cross-attention V-span bottleneck that produced v2's universal tint.

    The fusion's final ``out_proj`` is zero-init (identity-at-init); the
    preceding LayerNorm provides non-zero input so gradient flows from step 1.
    """

    def __init__(
        self,
        action_dim: int,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        spatial_downsample: int = 2,
        action_temporal_align: str = "avg_pool",
        out_proj_init: str = "kaiming",
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.hidden_dim = hidden_dim
        if action_temporal_align not in ("avg_pool", "interp"):
            raise ValueError(
                f"action_temporal_align must be 'avg_pool' or 'interp'; "
                f"got {action_temporal_align!r}"
            )
        if out_proj_init not in ("zero", "kaiming"):
            raise ValueError(
                f"out_proj_init must be 'zero' or 'kaiming'; got {out_proj_init!r}"
            )
        self.action_temporal_align = action_temporal_align
        self.out_proj_init = out_proj_init

        self.render_encoder = RenderPerFrameEncoderV3(
            in_channels=render_in_channels,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            spatial_downsample=spatial_downsample,
        )
        self.action_encoder = ActionEncoderV3(
            action_dim=action_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Fusion: [r ‖ a] (2·hidden_dim) → hidden_dim → inner_dim
        # LayerNorm + Linear + SiLU + final Linear (kept named ``out_proj``
        # so existing diagnostics — wandb `action_adapter.out_proj.weight.norm`
        # — work for both v2 and v3 unchanged).
        self.fuse_norm = nn.LayerNorm(2 * hidden_dim)
        self.fuse_hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fuse_act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, inner_dim)
        # Init choice:
        #  - "zero" (legacy): identity-at-init (contribution=0 at step 0), but
        #    creates the dual-zero pattern that traps upstream layers at the
        #    first non-zero direction out_proj picks — the bias attractor.
        #  - "kaiming" (default): standard kaiming-uniform on weight, zero
        #    bias. Contribution is non-trivial at step 0; model has to
        #    immediately learn to use or suppress it. egowm's render_encoder
        #    works the same way and trains fine.
        if out_proj_init == "zero":
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:  # kaiming
            nn.init.kaiming_uniform_(self.out_proj.weight, a=5 ** 0.5)
            nn.init.zeros_(self.out_proj.bias)

    def _align_actions(self, a: torch.Tensor, T_lat: int) -> torch.Tensor:
        """Map (B, T_act, hidden) → (B, T_lat, hidden).

        avg_pool: reshape T_act into groups of T_act/T_lat and average. Matches
                  the Wan VAE's temporal stride (4 video frames → 1 latent frame)
                  when T_act = 4·T_lat (which holds for num_frames=81, T_lat=21
                  modulo the +1 frame at the boundary).
        interp:   linear interpolation across T. Smooth but doesn't reflect
                  the VAE's temporal binning.
        """
        T_act = a.shape[1]
        if T_act == T_lat:
            return a
        if self.action_temporal_align == "avg_pool":
            # Use adaptive avg pool over the temporal axis: handles non-integer
            # ratios cleanly. PyTorch wants (B, C, T) so transpose.
            a_t = a.transpose(1, 2)                          # (B, hidden, T_act)
            a_t = F.adaptive_avg_pool1d(a_t, T_lat)          # (B, hidden, T_lat)
            return a_t.transpose(1, 2)                       # (B, T_lat, hidden)
        else:
            a_t = a.transpose(1, 2)
            a_t = F.interpolate(a_t, size=T_lat, mode="linear", align_corners=False)
            return a_t.transpose(1, 2)

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            render_latents: (B, C_vae, T_lat_in, H_lat, W_lat)
            actions:        (B, T_act, action_dim)
        Returns:
            contribution:   (B, T_lat·h·w, inner_dim)  — per-token contribution
                            ready to be ADD-ed to temb_base before time_proj.
                            Zero at step 0 via zero-init ``out_proj``.
        """
        # Render → (B, T_lat, h·w, hidden_dim)
        r = self.render_encoder(render_latents)
        B, T_lat, S, hid = r.shape

        # Action → (B, T_act, hidden_dim)
        a = self.action_encoder(actions)
        # Align action temporal length to latent temporal length.
        a = self._align_actions(a, T_lat)                    # (B, T_lat, hidden_dim)
        # Broadcast per-frame action to every spatial token in that frame.
        a_per_token = a.unsqueeze(2).expand(-1, -1, S, -1)   # (B, T_lat, h·w, hidden_dim)

        # Concat along feature dim. Render and action keep their full per-
        # token diversity — there's no mixing operation that constrains the
        # output to a low-rank subspace.
        fused = torch.cat([r, a_per_token], dim=-1)          # (B, T_lat, h·w, 2·hidden_dim)
        fused_flat = fused.reshape(B, T_lat * S, 2 * hid)

        # AdaLN-zero fusion: LayerNorm → Linear → SiLU → zero-init Linear.
        h = self.fuse_norm(fused_flat)                       # (B, T*S, 2·hidden_dim)
        h = self.fuse_act(self.fuse_hidden(h))               # (B, T*S, hidden_dim)
        return self.out_proj(h)                              # (B, T*S, inner_dim)

    @staticmethod
    def combine_with_temb(
        temb_base: torch.Tensor,
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        """Same broadcast semantics as v2: tile temb_base to per-token and add."""
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
    "RenderPerFrameEncoderV3",
    "ActionEncoderV3",
    "ActionRenderConcatAdapter",
]
