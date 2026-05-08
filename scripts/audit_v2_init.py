#!/usr/bin/env python3
"""
Verify the v2 (ActionConditionedTembAdapter) init pipeline is correct.

Three guarantees we want:

  (A) Identity-at-init: ``out_proj.weight`` and ``out_proj.bias`` are
      EXACTLY zero after the full init pipeline (from_pretrained → meta
      materialize → reset_zero_gates).

  (B) Cross-attn is alive: ``cross_attn.in_proj_weight.norm() > 0``.
      A zero in_proj would silently kill the cross-attn output, and the
      LayerNorm-before-zero-init-proj defense against the dual-zero
      deadlock collapses → out_proj.weight gradient becomes 0 → adapter
      stuck at identity forever.

  (C) Forward pass with zero-init out_proj produces an EXACTLY zero
      contribution (so ``temb_per_token == temb_base.broadcast`` at step 0).

Run::

    PYTHONPATH=src python scripts/audit_v2_init.py
"""
from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from world_model.wan_flow.model import WanTransformerRenderConditioned
from world_model.wan_flow.train import _materialize_meta_submodules


def main() -> None:
    model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    print(f"[audit] loading {model_path} with legacy_render_variant='v2' ...")
    dit = WanTransformerRenderConditioned.from_pretrained(
        model_path, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        render_encoder_kwargs={},
        use_embodiment_adapter=False,
        legacy_render_variant="v2",
        v2_adapter_kwargs={
            "action_dim": 8, "hidden_dim": 512,
            "num_heads": 8, "spatial_downsample": 2,
        },
    )
    n_meta = _materialize_meta_submodules(dit)
    print(f"[audit] materialized {n_meta} meta submodules")
    if hasattr(dit, "reset_zero_gates"):
        dit.reset_zero_gates()
        print(f"[audit] called reset_zero_gates")

    aa = dit.action_adapter
    cross_attn_ip_w = aa.cross_attn.in_proj_weight.detach().float()
    cross_attn_op_w = aa.cross_attn.out_proj.weight.detach().float()
    out_proj_w = aa.out_proj.weight.detach().float()
    out_proj_b = aa.out_proj.bias.detach().float()

    print()
    print(f"[audit] === post-init norms (all in fp32) ===")
    print(f"  cross_attn.in_proj_weight :  norm={cross_attn_ip_w.norm():.4f}  "
          f"max_abs={cross_attn_ip_w.abs().max():.4g}")
    print(f"  cross_attn.out_proj.weight:  norm={cross_attn_op_w.norm():.4f}  "
          f"max_abs={cross_attn_op_w.abs().max():.4g}")
    print(f"  cross_attn (q_norm.weight):  norm={aa.q_norm.weight.detach().float().norm():.4f}")
    print(f"  cross_attn (kv_norm.weight): norm={aa.kv_norm.weight.detach().float().norm():.4f}")
    print(f"  out_norm.weight           :  norm={aa.out_norm.weight.detach().float().norm():.4f}")
    print(f"  out_proj.weight (zero?)   :  norm={out_proj_w.norm():.4g}")
    print(f"  out_proj.bias   (zero?)   :  norm={out_proj_b.norm():.4g}")

    print()
    ok = []

    # (A) out_proj zero-init
    if out_proj_w.norm() == 0.0 and out_proj_b.norm() == 0.0:
        print("  PASS (A) out_proj is exactly zero (identity-at-init)")
        ok.append(True)
    else:
        print(f"  FAIL (A) out_proj weight/bias not exactly zero — identity-at-init broken")
        ok.append(False)

    # (B) cross_attn in_proj alive
    if cross_attn_ip_w.norm() > 1.0:
        print(f"  PASS (B) cross_attn.in_proj_weight is xavier-init alive ({cross_attn_ip_w.norm():.2f})")
        ok.append(True)
    else:
        print(f"  FAIL (B) cross_attn.in_proj_weight is too small "
              f"({cross_attn_ip_w.norm():.4g}) — MHA materialization bug. "
              f"Forward will produce 0 → ∂L/∂out_proj=0 → dead pathway.")
        ok.append(False)

    # (C) Forward pass returns exactly zero contribution
    print()
    print(f"[audit] === forward-pass identity-at-init check ===")
    aa_fp32 = aa.to(torch.float32).eval()
    B, T_lat, H_lat, W_lat = 1, 5, 30, 54
    T_act = 17
    render = torch.randn(B, 16, T_lat, H_lat, W_lat)
    actions = torch.randn(B, T_act, 8)
    with torch.no_grad():
        contribution = aa_fp32(render, actions)
    print(f"  contribution shape={tuple(contribution.shape)}  "
          f"max_abs={contribution.abs().max():.4g}  norm={contribution.norm():.4g}")
    if contribution.abs().max() == 0.0:
        print("  PASS (C) contribution is exactly zero — vanilla Wan at step 0")
        ok.append(True)
    else:
        print("  FAIL (C) contribution is non-zero — identity-at-init violated")
        ok.append(False)

    print()
    if all(ok):
        print("[audit] ALL CHECKS PASSED — v2 init pipeline is correct.")
        sys.exit(0)
    else:
        print(f"[audit] {sum(o is False for o in ok)} CHECK(S) FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
