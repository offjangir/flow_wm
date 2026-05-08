"""
Embodiment-agnostic conditioning for the Wan I2V world model.

This module is a *proposal* file containing drop-in PyTorch components that
extend the current EgoWM Eq. 5 conditioning (Z_ts + Z_a → time_proj → AdaLN)
in three orthogonal, independently-gated layers:

    Layer 1 (Z_s, EgoWM Eq. 6): InitialStateEncoder
        Encodes embodiment identity from the *first* render frame.
        Persistent across the whole sequence.  No discrete robot-type ID
        is needed — the DrRobot render already carries the embodiment
        signal visually (separate DrRobot model per robot kind), so the
        first frame is a complete morphological snapshot.

    Layer 2 (Decoupled AdaLN): DecoupledAdaLNHead
        Render and state contribute *directly* to the (scale, shift, gate)
        AdaLN parameters via a dedicated low-rank head — bypassing the
        scalar `temb` → `time_proj` bottleneck so action features cannot
        be washed out into a single 5,120-d bus.

    Layer 3 (Cross-attention adapters): RenderCrossAttnAdapter
        Per-block residual cross-attention from DiT tokens to a KV bank of
        spatial render tokens + state tokens.  Carries fine, *position-
        aware* action signal (joint positions, gripper state) that AdaLN
        cannot express well.  Inserted at every k-th block.

Every component is **zero-gated at init** (final projection zeroed and/or a
scalar gate set to 0), so the assembled model is *bit-exact* identical to
vanilla Wan 2.1 I2V on step 0.  Each layer can be ablated independently.

Wiring sketch (drop-in for `WanTransformerRenderConditioned.forward`)::

    embodiment = self.embodiment            # EmbodimentAgnosticConditioning
    cond = embodiment(render_latents,
                      tokens_per_frame=tokens_per_frame,
                      num_post_patch_frames=post_patch_num_frames)

    # 1) base time embedding from Wan (scalar timestep only)
    temb_base, _, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(timestep, encoder_hidden_states,
                                encoder_hidden_states_image, timestep_seq_len=None)
    )
    mod_base = self.condition_embedder.time_proj(
        self.condition_embedder.act_fn(temb_base)
    )                                                       # (B, 6*D)
    timestep_proj = embodiment.combine_modulation(
        mod_base=mod_base, cond=cond
    )                                                       # (B, num_tokens, 6, D)

    # 2) DiT block loop with cross-attention adapters
    kv_bank = cond["kv_bank"]                               # (B, K+S, D)
    for i, block in enumerate(self.blocks):
        h = block(h, encoder_hidden_states, timestep_proj, rotary_emb)
        if i in embodiment.adapter_block_ids:
            h = embodiment.adapters[str(i)](h, kv_bank)

    # 3) final norm_out — unchanged; uses temb_base only (vanilla Wan).

The aux ``tracks_head`` is unaffected: it still reads the post-modulation
``token_grid`` and benefits "for free" from the richer h-tokens.

Param budget (relative to a 14B Wan DiT, D=5120, num_layers=40):
    InitialStateEncoder (K=8, hidden=512)         ~10 M
    RenderSpatialEncoder (hidden=512)             ~5 M
    DecoupledAdaLNHead (low-rank=512, 2 paths)    ~36 M
    RenderCrossAttnAdapter × 5 blocks             ~130 M
    -------------------------------------------------------
    Total                                         ~180 M  (~1.3% of 14B)

Notes on dtype / FSDP — these modules should be added to
``WanTransformerRenderConditioned._keep_in_fp32_modules`` so FSDP's
mixed-precision wrapper does not silently bf16-cast the zero-init layers
before they have a chance to receive gradients.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zero_init_(module: nn.Module) -> nn.Module:
    """Zero-initialize a Linear/Conv module's weight and bias."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.zeros_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1a: Initial-state encoder (EgoWM Eq. 6 — Z_s)
# ─────────────────────────────────────────────────────────────────────────────

class InitialStateEncoder(nn.Module):
    """
    Encode the *first frame* of the render-video latents into a sequence of
    state tokens that captures embodiment morphology (robot kind, camera pose,
    initial joint configuration, gripper state).

    Why first frame?  At t=0 the real video and the render are co-registered
    on the same scene and pose.  The first render latent thus contains a
    near-perfect snapshot of the embodiment — far cheaper to encode than the
    full T_lat-frame render and provides a *constant* embodiment anchor that
    the later per-frame action signal Z_a can be interpreted relative to.

    Why K>1 tokens?  A single 5,120-d vector is high capacity but offers no
    way for the DiT to attend to *different aspects* of the embodiment
    (left arm vs. right arm vs. table vs. gripper).  K=8 attention-pooled
    learned-query tokens preserve that decomposition for the cross-attn bank.

    Output is **not zero-init** here — it's the downstream head
    (``state_to_adaln``) and cross-attn ``out_proj`` that must zero-init to
    achieve overall zero-gate behaviour.  Keeping this encoder healthily
    initialized avoids the gradient-bottleneck collapse described in the
    existing ``render_fuse`` comment in model.py.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_dim: int = 5120,
        hidden_dim: int = 512,
        num_state_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        # Learned-query attention pool: K queries attend over the spatial
        # feature map → K state tokens.  Cheaper than flattening the whole
        # spatial grid into the KV bank.
        self.query = nn.Parameter(torch.randn(num_state_tokens, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            render_latents: (B, C_vae, T_lat, H_lat, W_lat).  Only the first
                temporal slice is consumed.

        Returns:
            (B, K, out_dim) state tokens.
        """
        if render_latents.ndim != 5:
            raise ValueError(
                f"render_latents must be 5D; got {tuple(render_latents.shape)}"
            )
        first = render_latents[:, :, 0]                       # (B, C, H, W)
        feat = self.proj(first)                               # (B, hidden, H, W)
        B, H, h, w = feat.shape
        feat_seq = feat.flatten(2).transpose(1, 2)            # (B, h*w, hidden)
        feat_seq = self.norm(feat_seq)
        q = self.query.unsqueeze(0).expand(B, -1, -1)         # (B, K, hidden)
        pooled, _ = self.attn(q, feat_seq, feat_seq, need_weights=False)
        return self.out(pooled)                               # (B, K, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1b: Render encoders — per-frame (for AdaLN) and spatial (for KV bank)
# ─────────────────────────────────────────────────────────────────────────────

class RenderPerFrameEncoder(nn.Module):
    """
    (B, C, T, H, W) → (B, T, D).  Same shape contract as the existing
    ``RenderLatentEncoder`` in model.py — kept here so the embodiment module
    is self-contained.  Re-using model.py's ``RenderLatentEncoder`` is also
    fine; this is just for clarity.
    """

    def __init__(
        self, in_channels: int = 16, out_dim: int = 5120, hidden_dim: int = 512
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        x = self.proj(render_latents)
        x = self.pool(x).squeeze(-1).squeeze(-1).transpose(1, 2)
        return self.out(x)


class RenderSpatialEncoder(nn.Module):
    """
    (B, C, T, H, W) → (B, T*h*w, D) where h, w are downsampled spatial dims.

    Used to build the cross-attention KV bank: keeping spatial structure
    lets the DiT attend to *where* in the render the gripper / end-effector
    is, not just an aggregate per-frame summary.

    Spatial pooling factor is configurable; default targets ~256 tokens per
    frame (so 21 latent frames × 256 ≈ 5,400 KV tokens — comparable to one
    DiT block's self-attention).
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_dim: int = 5120,
        hidden_dim: int = 512,
        spatial_pool: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.pool = nn.AvgPool3d((1, spatial_pool, spatial_pool))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, render_latents: torch.Tensor) -> torch.Tensor:
        x = self.proj(render_latents)                       # (B, H, T, h, w)
        x = self.pool(x)                                    # (B, H, T, h', w')
        B, H, T, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * h * w, H)
        return self.out(x)                                  # (B, T*h*w, D)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Decoupled AdaLN head
# ─────────────────────────────────────────────────────────────────────────────

class DecoupledAdaLNHead(nn.Module):
    """
    Project an arbitrary (B, *, D) conditioning sequence directly to a
    contribution to the 6-way Wan AdaLN parameters (B, *, 6, D), bypassing
    the scalar ``temb`` → ``time_proj`` bottleneck.

    The forward returns a *delta* on top of the base modulation; combination
    happens in :meth:`EmbodimentAgnosticConditioning.combine_modulation`.

    Architecture: LayerNorm → Linear(D, R) → SiLU → Linear(R, 6*D).  The
    last Linear is **zero-initialized**, which (combined with the global
    scalar gate in the orchestrator) guarantees the contribution is exactly
    zero at step 0 — vanilla Wan behaviour preserved bit-exactly.

    Low-rank (R=512 by default) keeps params sane: 5120·512 + 512·6·5120 ≈
    18 M per head, vs. 157 M for a full 5120 → 6·5120 projection.
    """

    def __init__(
        self,
        in_dim: int = 5120,
        out_dim: int = 5120,
        rank: int = 512,
        num_modulations: int = 6,
    ) -> None:
        super().__init__()
        self.num_modulations = num_modulations
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, rank, bias=True)
        self.act = nn.SiLU()
        self.up = nn.Linear(rank, num_modulations * out_dim, bias=True)
        _zero_init_(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, S, D)
        Returns: (B, S, num_modulations, out_dim)
        """
        h = self.up(self.act(self.down(self.norm(x))))
        return h.unflatten(-1, (self.num_modulations, self.out_dim))


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Per-block render cross-attention adapter
# ─────────────────────────────────────────────────────────────────────────────

class RenderCrossAttnAdapter(nn.Module):
    """
    Residual cross-attention from DiT tokens (queries) to the embodiment KV
    bank (state tokens + spatial render tokens).  Inserted *after* a Wan DiT
    block's normal forward, this adapter lets the DiT pull *spatially-
    indexed* action information that AdaLN modulation cannot express.

    Identity-at-init: scalar ``gate`` initialized to 0 (alone). DO NOT
    additionally zero ``out_proj.weight`` — that creates a dual-zero
    bootstrap deadlock: ∂L/∂gate ∝ out_proj(attn(...)) and
    ∂L/∂out_proj.weight ∝ gate, so if both are zero, no gradient ever
    flows and the adapter stays at identity forever.  Keep
    MultiheadAttention's default kaiming-init on out_proj; gate=0 alone
    suffices for identity-at-init and gives a healthy gradient escape.

    Memory: a Wan I2V latent has num_tokens ≈ 21 · 30 · 52 ≈ 33 K and the
    KV bank is ~5 K, so cross-attention is O(33K · 5K · D / heads) ≈ a
    fraction of one block's *self*-attention cost.  Inserted at every k-th
    block to keep the marginal cost small (~5 of 40 blocks = 12% slowdown).
    """

    def __init__(
        self,
        dim: int = 5120,
        num_heads: int = 8,
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kv_dim = kv_dim or dim
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=kv_dim,
            vdim=kv_dim,
            batch_first=True,
        )
        # Per-channel gate (D,). Zero-init → identity at step 0; each channel
        # ramps independently. A scalar gate (size-1) has terrible SNR because
        # its single gradient averages over (B, num_tokens, D)≈10M terms with
        # mixed signs that cancel — empirically scalar gates stay at exactly 0
        # while per-channel gates ramp to ~0.03 norm in the same training time.
        # Don't also zero out_proj — see class docstring on dual-zero deadlock.
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, h_tokens: torch.Tensor, kv_bank: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_tokens: (B, num_tokens, D)
            kv_bank:  (B, S, kv_dim)
        Returns:
            (B, num_tokens, D)  — h_tokens + gate * residual.
        """
        q = self.q_norm(h_tokens)
        kv = self.kv_norm(kv_bank)
        residual, _ = self.attn(q, kv, kv, need_weights=False)
        return h_tokens + self.gate * residual


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: Action-aware spatial AdaLN delta
# ─────────────────────────────────────────────────────────────────────────────

class ActionAwareSpatialAdaLN(nn.Module):
    """Spatial AdaLN delta from render features, optionally enriched by
    cross-attention over an action trajectory.

    Pipeline::

        render_latents (B, C, T, H, W)
            └─ RenderSpatialEncoder(spatial_pool)  → (B, T·h·w, D)   render_tokens
        actions        (B, T_act, action_dim)
            └─ Linear(action_dim → D)              → (B, T_act, D)   action_tokens

        if actions is not None:
            render_tokens (Q) ← cross_attn → action_tokens (K, V)
                                            → (B, T·h·w, D)   enriched
            features = enriched
        else:
            features = render_tokens                ← bypass

        DecoupledAdaLNHead(features)               → (B, T·h·w, 6, D)   mod_delta

    Caller fuses into modulation with a **per-channel** gate (zero-init),
    so each of the D channels learns independently how strongly to inject
    the action-aware spatial signal — much faster gradient escape than a
    scalar gate.

    Choose ``spatial_pool`` so that the encoder's output spatial grid
    ``(h, w) = (H // spatial_pool, W // spatial_pool)`` matches the DiT's
    post-patch grid ``(H_p, W_p) = (H // p_h, W // p_w)``. With Wan's
    default ``patch_size=(1, 2, 2)``, ``spatial_pool=2`` aligns 1:1 and
    no interpolation is needed at fusion time.

    Identity-at-init: ``adaln_head.up`` is zero-init (inherited from
    DecoupledAdaLNHead) and ``gate`` is zero-init, so the delta is exactly
    zero at step 0 and vanilla Wan behaviour is preserved bit-exact.
    """

    def __init__(
        self,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        hidden_dim: int = 512,
        spatial_pool: int = 2,
        action_dim: Optional[int] = None,
        num_heads: int = 8,
        adaln_rank: int = 512,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.spatial_pool = spatial_pool
        self.action_dim = action_dim

        self.render_encoder = RenderSpatialEncoder(
            in_channels=render_in_channels,
            out_dim=inner_dim,
            hidden_dim=hidden_dim,
            spatial_pool=spatial_pool,
        )

        if action_dim is not None:
            self.action_proj = nn.Linear(action_dim, inner_dim)
            self.q_norm = nn.LayerNorm(inner_dim)
            self.kv_norm = nn.LayerNorm(inner_dim)
            self.cross_attn = nn.MultiheadAttention(
                inner_dim, num_heads=num_heads, batch_first=True,
            )
        else:
            self.action_proj = None
            self.cross_attn = None

        self.adaln_head = DecoupledAdaLNHead(
            in_dim=inner_dim, out_dim=inner_dim, rank=adaln_rank,
        )
        # Override DecoupledAdaLNHead's zero-init on `up`: we rely *solely*
        # on the per-channel `gate` (zeroed below) for identity-at-init.
        # If both `gate` and `up.weight` are zero, the contribution
        # `gate * head(features)` has zero gradient w.r.t. *both* — the
        # whole pathway gets stuck at zero forever (dual-zero bootstrap
        # bug). Healthy `up` + zero `gate` keeps identity-at-init exact
        # AND gives a healthy gradient to `gate` from step 1 (since
        # head output is non-zero, ∂L/∂gate flows).
        nn.init.kaiming_uniform_(self.adaln_head.up.weight, a=math.sqrt(5))
        if self.adaln_head.up.bias is not None:
            nn.init.zeros_(self.adaln_head.up.bias)
        # Per-channel gate: (D,). Zero-init → identity at step 0; each
        # channel ramps independently — much better gradient escape than
        # a scalar bottleneck shared across all 5,120 channels.
        self.gate = nn.Parameter(torch.zeros(inner_dim))

    def forward(
        self,
        render_latents: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            render_latents: (B, C_vae, T_lat, H_lat, W_lat).
            actions:        (B, T_act, action_dim) or ``None`` for bypass.

        Returns:
            mod_delta: (B, T_lat·h·w, 6, D).
        """
        r = self.render_encoder(render_latents)              # (B, T·h·w, D)

        if actions is not None and self.cross_attn is not None:
            a = self.action_proj(actions)                    # (B, T_act, D)
            q = self.q_norm(r)
            kv = self.kv_norm(a)
            features, _ = self.cross_attn(q, kv, kv, need_weights=False)
        else:
            features = r                                     # bypass

        return self.adaln_head(features)                     # (B, T·h·w, 6, D)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class EmbodimentConditioning(Dict[str, torch.Tensor]):
    """Lightweight typed dict to keep `forward` outputs readable."""


class EmbodimentAgnosticConditioning(nn.Module):
    """
    Top-level module that produces all conditioning artefacts the modified
    Wan DiT forward needs:

        ``cond["mod_render"]``  (B, num_tokens, 6, D)  — Layer 2 contribution
        ``cond["mod_state"]``   (B, 1,          6, D)  — Layer 2 contribution
        ``cond["kv_bank"]``     (B, K + S,      D)     — Layer 3 KV bank
        ``cond["z_a_per_token"]`` (B, num_tokens, D)   — Layer 1+ Eq. 5 fallback
                                                          (kept for ablation)

    Plus the orchestration helper :meth:`combine_modulation` that fuses the
    Wan vanilla ``time_proj`` output with the decoupled AdaLN deltas under a
    single scalar gate per pathway.

    Usage in `WanTransformerRenderConditioned.__init__`::

        self.embodiment = EmbodimentAgnosticConditioning(
            inner_dim=inner_dim,
            num_blocks=num_layers,
            adapter_every_k=8,
            use_eq5_residual=False,    # turn off legacy additive Z_a
        )

    Drop-in replacement for the existing ``render_encoder`` / ``render_fuse``
    / ``render_gate`` triplet.
    """

    def __init__(
        self,
        inner_dim: int = 5120,
        render_in_channels: int = 16,
        # Layer 1
        num_state_tokens: int = 8,
        state_hidden_dim: int = 512,
        # Layer 2
        adaln_rank: int = 512,
        # Layer 3
        num_blocks: int = 40,
        adapter_every_k: int = 8,        # → 5 adapters across 40 blocks
        adapter_num_heads: int = 8,
        # Spatial render encoder
        spatial_pool: int = 4,
        # Legacy Eq. 5 fallback path (additive Z_a into temb).  Kept so the
        # new code can be ablated against the old.
        use_eq5_residual: bool = False,
        # Layer 4: action-aware spatial AdaLN delta. Bypass-capable: works
        # without ``actions`` (raw spatial render features → AdaLN). With
        # ``action_dim`` set, render tokens cross-attend to the action
        # trajectory before the AdaLN head.
        use_action_aware_adaln: bool = False,
        action_aware_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()

        self.inner_dim = inner_dim
        self.use_eq5_residual = use_eq5_residual

        # ── Layer 1: identity & action encoders ────────────────────────────
        self.state_encoder = InitialStateEncoder(
            in_channels=render_in_channels,
            out_dim=inner_dim,
            hidden_dim=state_hidden_dim,
            num_state_tokens=num_state_tokens,
        )

        self.render_encoder = RenderPerFrameEncoder(
            in_channels=render_in_channels,
            out_dim=inner_dim,
            hidden_dim=state_hidden_dim,
        )
        self.render_spatial_encoder = RenderSpatialEncoder(
            in_channels=render_in_channels,
            out_dim=inner_dim,
            hidden_dim=state_hidden_dim,
            spatial_pool=spatial_pool,
        )

        # ── Layer 2: decoupled AdaLN heads ─────────────────────────────────
        self.render_to_adaln = DecoupledAdaLNHead(
            in_dim=inner_dim, out_dim=inner_dim, rank=adaln_rank
        )
        self.state_to_adaln = DecoupledAdaLNHead(
            in_dim=inner_dim, out_dim=inner_dim, rank=adaln_rank
        )
        # Override DecoupledAdaLNHead's zero-init on `up`: we rely *solely*
        # on the per-channel `*_adaln_gate` (zeroed below) for identity-at-init.
        # If both `gate` and `up.weight` are zero, the contribution
        # `gate * head(features)` has zero gradient w.r.t. *both* — the whole
        # pathway gets stuck at zero forever (dual-zero bootstrap bug).
        # Healthy `up` + zero `gate` keeps identity-at-init exact AND gives a
        # healthy gradient to `gate` from step 1 (since head output is
        # non-zero, ∂L/∂gate flows). Mirrors ActionAwareSpatialAdaLN.
        for _head in (self.render_to_adaln, self.state_to_adaln):
            nn.init.kaiming_uniform_(_head.up.weight, a=math.sqrt(5))
            if _head.up.bias is not None:
                nn.init.zeros_(_head.up.bias)
        # Per-channel gates (D,). Zero-init → identity at step 0; each channel
        # ramps independently — much better gradient escape than a scalar
        # bottleneck shared across all 5,120 channels.
        self.render_adaln_gate = nn.Parameter(torch.zeros(inner_dim))
        self.state_adaln_gate = nn.Parameter(torch.zeros(inner_dim))

        # ── Layer 1 fallback (legacy Eq. 5): additive Z_a into temb ───────
        if use_eq5_residual:
            self.render_fuse = nn.Linear(inner_dim, inner_dim)
            nn.init.zeros_(self.render_fuse.bias)
            self.render_fuse_gate = nn.Parameter(torch.zeros(1))

        # ── Layer 3: cross-attention adapters at every k-th block ─────────
        self.adapter_block_ids: List[int] = list(
            range(adapter_every_k - 1, num_blocks, adapter_every_k)
        )
        self.adapters = nn.ModuleDict({
            str(i): RenderCrossAttnAdapter(
                dim=inner_dim, num_heads=adapter_num_heads
            )
            for i in self.adapter_block_ids
        })

        # ── Layer 4: action-aware spatial AdaLN delta ─────────────────────
        self.use_action_aware_adaln = bool(use_action_aware_adaln)
        if self.use_action_aware_adaln:
            aa_kw = dict(action_aware_kwargs or {})
            aa_kw.setdefault("inner_dim", inner_dim)
            aa_kw.setdefault("render_in_channels", render_in_channels)
            aa_kw.setdefault("spatial_pool", 2)   # match Wan post-patch grid
            self.action_aware_adaln = ActionAwareSpatialAdaLN(**aa_kw)
        else:
            self.action_aware_adaln = None

    # ------------------------------------------------------------------ utils

    def fp32_module_names(self) -> List[str]:
        """Names to add to ``WanTransformerRenderConditioned._keep_in_fp32_modules``."""
        names = [
            "state_encoder",
            "render_encoder",
            "render_spatial_encoder",
            "render_to_adaln",
            "state_to_adaln",
        ]
        names.extend(f"adapters.{i}" for i in self.adapter_block_ids)
        if self.use_action_aware_adaln:
            names.append("action_aware_adaln")
        return [f"embodiment.{n}" for n in names]

    # --------------------------------------------------------------- forward

    def forward(
        self,
        render_latents: torch.Tensor,
        *,
        tokens_per_frame: int,
        num_post_patch_frames: int,
        drop_render_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> EmbodimentConditioning:
        """
        Args:
            render_latents:        (B, C_vae, T_lat, H_lat, W_lat).
            tokens_per_frame:      p_h * p_w of the DiT patch grid.
            num_post_patch_frames: post-patch temporal length of hidden_states.
            drop_render_mask:      (B,) float ∈ {0, 1}; broadcast-multiplied into
                                   the render contributions for CFG / dropout.

        Returns dict with all the derived conditioning artefacts.
        """
        B = render_latents.shape[0]
        D = self.inner_dim

        # ── 1. Identity tokens (Z_s) — embodiment morphology from frame 0 ─
        # The DrRobot render itself encodes which robot this is (separate
        # DrRobot model per robot kind), so the first frame is a complete
        # morphological snapshot.  No discrete robot-type token is needed.
        s_tokens = self.state_encoder(render_latents)        # (B, K, D)

        # ── 2. Per-frame render embedding (for decoupled AdaLN) ───────────
        z_render = self.render_encoder(render_latents)       # (B, T_lat, D)
        if z_render.shape[1] != num_post_patch_frames:
            z_render = F.interpolate(
                z_render.transpose(1, 2),
                size=num_post_patch_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # ── 3. Spatial render bank (for cross-attention) ──────────────────
        spatial_tokens = self.render_spatial_encoder(render_latents)  # (B, S, D)

        # ── 4. CFG / dropout mask ──────────────────────────────────────────
        if drop_render_mask is not None:
            keep = drop_render_mask.view(B, 1, 1).to(z_render.dtype)
            z_render = z_render * keep
            spatial_tokens = spatial_tokens * keep
            s_tokens = s_tokens * keep

        # ── 5. KV bank for cross-attention adapters ───────────────────────
        kv_bank = torch.cat([s_tokens, spatial_tokens], dim=1)        # (B, K+S, D)

        # ── 6. Decoupled AdaLN deltas ─────────────────────────────────────
        mod_render = self.render_to_adaln(z_render)                   # (B, T_lat, 6, D)
        # broadcast per-latent-frame to per-token by repeating spatially
        mod_render = mod_render.repeat_interleave(tokens_per_frame, dim=1)
        # (B, num_tokens, 6, D)

        # State → AdaLN: pool to a single vector (broadcast across all tokens)
        s_pool = s_tokens.mean(dim=1, keepdim=True)                   # (B, 1, D)
        mod_state = self.state_to_adaln(s_pool)                       # (B, 1, 6, D)

        # ── 7. Optional legacy Eq. 5 path (additive Z_a per-token) ────────
        z_a_per_token: Optional[torch.Tensor] = None
        if self.use_eq5_residual:
            za = self.render_fuse_gate * self.render_fuse(z_render)
            z_a_per_token = za.repeat_interleave(tokens_per_frame, dim=1)

        # ── 8. Layer-4 action-aware spatial AdaLN delta (optional) ────────
        mod_action_aware: Optional[torch.Tensor] = None
        if self.use_action_aware_adaln and self.action_aware_adaln is not None:
            mod_action_aware = self.action_aware_adaln(
                render_latents, actions=actions,
            )                                              # (B, T·h·w, 6, D)
            if drop_render_mask is not None:
                keep = drop_render_mask.view(B, 1, 1, 1).to(mod_action_aware.dtype)
                mod_action_aware = mod_action_aware * keep

        return EmbodimentConditioning({
            "s_tokens": s_tokens,
            "z_render": z_render,
            "kv_bank": kv_bank,
            "mod_render": mod_render,
            "mod_state": mod_state,
            "z_a_per_token": z_a_per_token,
            "mod_action_aware": mod_action_aware,
        })

    # ----------------------------------------------------- modulation merge

    def combine_modulation(
        self,
        mod_base: torch.Tensor,
        cond: EmbodimentConditioning,
    ) -> torch.Tensor:
        """
        Combine the vanilla Wan modulation parameters with the decoupled
        AdaLN deltas, gated.

        Args:
            mod_base: (B, 6*D) — output of ``time_proj(SiLU(temb_base))``.
            cond:     dict from :meth:`forward`.

        Returns:
            (B, num_tokens, 6, D) modulation tensor consumable by every Wan
            block (matches the ``timestep_seq_len`` pathway shape).
        """
        D = self.inner_dim
        # base: (B, 6*D) → (B, 1, 6, D)
        mod = mod_base.unflatten(-1, (6, D)).unsqueeze(1)
        # render: (B, num_tokens, 6, D)
        mod = mod + self.render_adaln_gate * cond["mod_render"]
        # state: (B, 1, 6, D) — broadcast to all tokens
        mod = mod + self.state_adaln_gate * cond["mod_state"]
        # action-aware spatial: (B, T·h·w, 6, D) under per-channel gate (D,).
        # When ``spatial_pool=2`` the encoder grid matches the DiT post-patch
        # grid, so T·h·w == num_tokens and broadcast is 1:1 by token.
        mod_aa = cond.get("mod_action_aware")
        if mod_aa is not None and self.action_aware_adaln is not None:
            if mod_aa.shape[1] != mod.shape[1] and mod.shape[1] != 1:
                raise ValueError(
                    f"action-aware AdaLN delta has {mod_aa.shape[1]} tokens but "
                    f"DiT expects {mod.shape[1]}. Set "
                    f"action_aware_kwargs.spatial_pool so render (h, w) matches "
                    f"DiT post-patch (H_p, W_p)."
                )
            mod = mod + self.action_aware_adaln.gate * mod_aa
        return mod                                                    # (B, N, 6, D)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-gate audit (executable sanity check)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def assert_identity_at_init(
    inner_dim: int = 5120,
    num_blocks: int = 40,
    batch_size: int = 1,
    t_lat: int = 21,
    h_lat: int = 60,
    w_lat: int = 104,
    tokens_per_frame: int = 1560,
) -> None:
    """
    Verify every gated pathway is *exactly* zero at init: the embodiment
    module's contributions to AdaLN are 0 and the cross-attention adapters
    return their inputs unchanged.

    Run me with::

        python -c 'from src.world_model.wan_flow.embodiment_adapter import \
            assert_identity_at_init; assert_identity_at_init()'
    """
    mod = EmbodimentAgnosticConditioning(
        inner_dim=inner_dim,
        num_blocks=num_blocks,
        adapter_every_k=8,
        spatial_pool=8,             # smaller for quick sanity test
    ).eval()

    render = torch.randn(batch_size, 16, t_lat, h_lat, w_lat)
    cond = mod(
        render,
        tokens_per_frame=tokens_per_frame,
        num_post_patch_frames=t_lat,
    )

    # 1) AdaLN deltas should be zero after gating.
    base = torch.randn(batch_size, 6 * inner_dim)
    fused = mod.combine_modulation(base, cond)
    expected = base.unflatten(-1, (6, inner_dim)).unsqueeze(1).expand(
        batch_size, t_lat * tokens_per_frame, 6, inner_dim
    )
    assert torch.allclose(fused, expected), "DecoupledAdaLN broke zero-gate!"

    # 2) Each cross-attention adapter should be identity.
    h = torch.randn(batch_size, t_lat * tokens_per_frame, inner_dim)
    for i in mod.adapter_block_ids:
        out = mod.adapters[str(i)](h, cond["kv_bank"])
        assert torch.allclose(out, h), f"adapter {i} broke zero-gate!"

    print("✔ identity-at-init: AdaLN delta = 0  &  adapters return identity.")


__all__ = [
    "InitialStateEncoder",
    "RenderPerFrameEncoder",
    "RenderSpatialEncoder",
    "DecoupledAdaLNHead",
    "RenderCrossAttnAdapter",
    "EmbodimentAgnosticConditioning",
    "assert_identity_at_init",
]
