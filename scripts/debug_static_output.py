#!/usr/bin/env python3
"""
Diagnostic: test if VANILLA pretrained Wan I2V produces motion for the same first frame.
Uses the exact same infrastructure as eval_world_model.py to avoid script discrepancies
and guarantee perfect behavior.
"""
import os, sys, torch, numpy as np, time
from PIL import Image
import imageio.v2 as imageio
from contextlib import nullcontext
import argparse

sys.path.insert(0, os.path.abspath("./scripts"))
sys.path.insert(0, os.path.abspath("./src"))
import eval_world_model as eval_wm

def main():
    model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    metadata_csv = "./data_wan/metadata_one_example.csv"
    base = "./data_wan"
    height, width, num_frames = 320, 576, 33
    device = torch.device("cuda")
    dtype = torch.bfloat16

    import pandas as pd
    df = pd.read_csv(metadata_csv)
    row = df.iloc[0]
    vpath = os.path.join(base, row["video"])

    print(f"[diag] Loading first frame from {vpath} ...")
    reader = imageio.get_reader(vpath, "ffmpeg")
    first_frame = Image.fromarray(reader.get_data(0)).convert("RGB").resize((width, height), Image.BILINEAR)
    reader.close()

    print("[diag] Loading pipeline using eval_world_model._build_pipeline ...")
    args = argparse.Namespace()
    args.model_path = model_path
    args.cpu_offload = True
    
    # This automatically instantiates RenderConditionedWanI2VPipeline
    pipe = eval_wm._build_pipeline(args, device, dtype)
    pipe.enable_model_cpu_offload(device=device)

    # We must pass something for render_video to satisfy the pipeline's assert.
    # Since we use the raw un-trained transformer, render_gate initializes to 0.0,
    # meaning the render_video input will be completely ignored, guaranteeing pure Vanilla output.
    render_frames = eval_wm._resize_pils(eval_wm._load_video_frames(vpath, num_frames), height, width)
    
    # Ensure gate is mathematically 0 to bypass conditioning
    with torch.no_grad():
        pipe.transformer.render_gate.zero_()

    gen = torch.Generator(device="cpu").manual_seed(42)
    print("\n[diag] Running VANILLA Wan I2V inference (50 steps)...")
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    
    with torch.inference_mode(), autocast_ctx:
        out = pipe(
            image=first_frame,
            prompt="",
            negative_prompt="",
            render_video=render_frames, 
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=50,
            guidance_scale=5.0,
            max_sequence_length=512,  # Align with true eval parameters
            generator=gen,
            output_type="np",
            return_dict=False,
        )

    frames_u8 = eval_wm._frames_to_uint8(out[0])
    
    # Check frame-to-frame difference
    print(f"\n[diag] Output shape: {frames_u8.shape}")
    diffs = []
    for i in range(1, len(frames_u8)):
        d = np.abs(frames_u8[i].astype(float) - frames_u8[0].astype(float)).mean()
        diffs.append(d)
        
    print(f"[diag] Mean pixel diff from frame 0:")
    for i, d in enumerate(diffs):
        print(f"  frame {i+1}: {d:.2f}")
    
    # Save output
    os.makedirs("./eval_outputs/debug_vanilla", exist_ok=True)
    out_path = "./eval_outputs/debug_vanilla/vanilla_wan_i2v_fixed.mp4"
    writer = imageio.get_writer(out_path, fps=8, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    for f in frames_u8:
        writer.append_data(f)
    writer.close()
    print(f"\n[diag] Saved clean vanilla output to {out_path}")

if __name__ == "__main__":
    main()
