#!/usr/bin/env python3
"""
Pre-flight gate-gradient test.

Verifies that EVERY gate parameter in the conditioning architecture receives
a non-zero gradient under realistic call patterns. Catches the bug class that
ate ~6 hours of training time on 2026-05-06: per-channel adapter gates of
identical structure to working AdaLN gates were silently stuck at exactly 0
because they were sandwiched between gradient-checkpointed transformer blocks.

The test mocks transformer blocks with simple MLPs and runs the gate-bearing
modules in isolation — no need to load the 14B Wan checkpoint. Each test:
    1. Builds the module(s) at small dims for speed (D=128 vs production 5120).
    2. Synthesizes inputs of realistic shape.
    3. Runs the production call pattern (adapter sandwiched between
       optionally-checkpointed mock blocks).
    4. Backprops a scalar loss.
    5. Asserts every parameter named ``*.gate`` / ``*_gate`` has
       ``param.grad`` non-None and ``param.grad.norm() > 0``.

Run::

    python scripts/test_gate_gradients.py

Exit code 0 if all gates received gradient; 1 otherwise.

Add new tests as new gating mechanisms get added to the architecture.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Make the test runnable from anywhere in the repo.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from world_model.wan_flow.embodiment_adapter import (  # noqa: E402
    EmbodimentAgnosticConditioning,
    RenderCrossAttnAdapter,
)


@dataclass
class GateResult:
    name: str
    has_grad: bool
    grad_norm: float
    passed: bool

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        if not self.has_grad:
            return f"{status}  {self.name:65s}  grad=None"
        return f"{status}  {self.name:65s}  grad_norm={self.grad_norm:.4e}"


def _check_gates(named_modules_or_params, prefix: str = "") -> List[GateResult]:
    """Walk parameters, return a GateResult for every ``*.gate`` / ``*_gate``."""
    results: List[GateResult] = []
    if isinstance(named_modules_or_params, nn.Module):
        iterator = named_modules_or_params.named_parameters()
    else:
        iterator = named_modules_or_params
    for name, p in iterator:
        if not (name.endswith(".gate") or name.endswith("_gate") or name == "gate"):
            continue
        full = f"{prefix}{name}" if prefix else name
        g = p.grad
        if g is None:
            results.append(GateResult(full, False, float("nan"), False))
        else:
            n = float(g.float().norm().item())
            results.append(GateResult(full, True, n, n > 0.0))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: RenderCrossAttnAdapter standalone
#   Question: does adapter.gate get gradient when the adapter is sandwiched
#   between gradient-checkpointed mock blocks?
# ─────────────────────────────────────────────────────────────────────────────

def test_adapter_gate_with_checkpoint_neighbors() -> List[GateResult]:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D, H = 128, 4

    # Mock transformer blocks — wrapped in checkpoint to mimic production.
    pre_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    post_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    adapter = RenderCrossAttnAdapter(dim=D, num_heads=H).to(device)

    h0 = torch.randn(2, 64, D, device=device, requires_grad=False)
    kv_bank = torch.randn(2, 32, D, device=device, requires_grad=False)

    # Production-pattern forward.
    h1 = checkpoint(pre_block, h0, use_reentrant=True)
    h2 = adapter(h1, kv_bank)
    h3 = checkpoint(post_block, h2, use_reentrant=True)

    loss = h3.float().sum()
    loss.backward()

    return _check_gates(adapter, prefix="RenderCrossAttnAdapter[ckpt-sandwiched].")


def test_adapter_gate_without_checkpointing() -> List[GateResult]:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D, H = 128, 4

    pre_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    post_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    adapter = RenderCrossAttnAdapter(dim=D, num_heads=H).to(device)

    h0 = torch.randn(2, 64, D, device=device)
    kv_bank = torch.randn(2, 32, D, device=device)

    h1 = pre_block(h0)
    h2 = adapter(h1, kv_bank)
    h3 = post_block(h2)

    loss = h3.float().sum()
    loss.backward()

    return _check_gates(adapter, prefix="RenderCrossAttnAdapter[no-ckpt].")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: full EmbodimentAgnosticConditioning — every gate should learn.
# ─────────────────────────────────────────────────────────────────────────────

def test_embodiment_gates() -> List[GateResult]:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = 128

    embodiment = EmbodimentAgnosticConditioning(
        inner_dim=D,
        render_in_channels=16,
        adaln_rank=64,
        num_blocks=8,
        adapter_every_k=2,            # → adapters at blocks 1, 3, 5, 7 (4 of them)
        adapter_num_heads=4,
        spatial_pool=2,
        state_hidden_dim=64,
        use_action_aware_adaln=True,
        action_aware_kwargs={
            "spatial_pool": 2,
            "action_dim": None,        # bypass mode (matches production config)
            "hidden_dim": 64,
            "adaln_rank": 64,
            "num_heads": 4,
        },
    ).to(device)

    # Synthetic render latents at small but plausible shape.
    B, C_vae, T_lat, H_lat, W_lat = 1, 16, 4, 16, 16
    render_latents = torch.randn(B, C_vae, T_lat, H_lat, W_lat, device=device)

    # The forward needs (tokens_per_frame, num_post_patch_frames). With Wan's
    # default patch (1, 2, 2): tokens_per_frame = (H_lat/2) * (W_lat/2) = 64.
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    num_post_patch_frames = T_lat
    cond = embodiment(
        render_latents,
        tokens_per_frame=tokens_per_frame,
        num_post_patch_frames=num_post_patch_frames,
    )

    # Use combine_modulation to exercise render_adaln_gate / state_adaln_gate /
    # action_aware_adaln.gate via the AdaLN delta path.
    mod_base = torch.randn(B, 6 * D, device=device)
    timestep_proj = embodiment.combine_modulation(mod_base, cond)

    # Exercise each adapter gate by calling each adapter — sandwich between
    # checkpointed mock blocks so we hit the same call pattern as production.
    pre_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    post_block = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(device)
    h = torch.randn(B, num_post_patch_frames * tokens_per_frame, D, device=device)
    h = checkpoint(pre_block, h, use_reentrant=True)
    for adapter in embodiment.adapters.values():
        h = adapter(h, cond["kv_bank"])
    h = checkpoint(post_block, h, use_reentrant=True)

    loss = (timestep_proj.float().sum() + h.float().sum())
    loss.backward()

    return _check_gates(embodiment, prefix="EmbodimentAgnosticConditioning.")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU "
              "(slower, less representative).")

    print("=" * 80)
    print("Gate-gradient pre-flight test")
    print("=" * 80)

    all_results: List[GateResult] = []
    all_results += test_adapter_gate_without_checkpointing()
    all_results += test_adapter_gate_with_checkpoint_neighbors()
    all_results += test_embodiment_gates()

    # Sort for stable output.
    all_results.sort(key=lambda r: r.name)
    for r in all_results:
        print(r)

    n_pass = sum(1 for r in all_results if r.passed)
    n_fail = sum(1 for r in all_results if not r.passed)
    print()
    print(f"Summary: {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
