#!/usr/bin/env python3
"""
SeedVR Single GPU Inference Runner

This script provides an easy way to run SeedVR inference on a single GPU
with optimized settings and automatic configuration.
"""

import argparse
import os
import sys
import subprocess
import torch

def check_gpu_availability():
    """Check if CUDA is available and get GPU info."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please install CUDA and PyTorch with CUDA support.")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    print(f"✅ CUDA is available")
    print(f"📊 GPU Count: {gpu_count}")
    print(f"🎮 GPU 0: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    return True

def check_requirements():
    """Check if required files exist."""
    required_files = [
        "pos_emb.pt",
        "neg_emb.pt",
    ]
    
    required_dirs = [
        "ckpts",
        "test_videos",
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    if missing_files or missing_dirs:
        print("❌ Missing required files/directories:")
        for file in missing_files:
            print(f"   - {file}")
        for dir in missing_dirs:
            print(f"   - {dir}/")
        return False
    
    print("✅ All required files and directories found")
    return True

def get_memory_recommendations(model_size, gpu_memory_gb):
    """Get memory-based recommendations for inference settings."""
    if model_size == "3b":
        if gpu_memory_gb >= 24:
            return {"batch_size": 1, "res_h": 720, "res_w": 1280, "note": "Optimal settings"}
        elif gpu_memory_gb >= 16:
            return {"batch_size": 1, "res_h": 512, "res_w": 768, "note": "Reduced resolution for memory"}
        else:
            return {"batch_size": 1, "res_h": 384, "res_w": 512, "note": "Low memory settings"}
    
    elif model_size == "7b":
        if gpu_memory_gb >= 40:
            return {"batch_size": 1, "res_h": 720, "res_w": 1280, "note": "Optimal settings"}
        elif gpu_memory_gb >= 24:
            return {"batch_size": 1, "res_h": 512, "res_w": 768, "note": "Reduced resolution for memory"}
        else:
            return {"batch_size": 1, "res_h": 384, "res_w": 512, "note": "Low memory settings - may still fail"}

def main():
    parser = argparse.ArgumentParser(description="SeedVR Single GPU Inference Runner")
    parser.add_argument("--model", choices=["3b", "7b"], required=True,
                       help="Model size to use (3b or 7b)")
    parser.add_argument("--video_path", type=str, default="./test_videos",
                       help="Path to input videos")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for generated videos")
    parser.add_argument("--cfg_scale", type=float, default=6.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--sample_steps", type=int, default=50,
                       help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=666,
                       help="Random seed")
    parser.add_argument("--res_h", type=int, default=None,
                       help="Output height (auto-determined if not specified)")
    parser.add_argument("--res_w", type=int, default=None,
                       help="Output width (auto-determined if not specified)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (recommended: 1 for single GPU)")
    parser.add_argument("--auto_settings", action="store_true",
                       help="Automatically determine optimal settings based on GPU memory")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show the command that would be run without executing it")
    
    args = parser.parse_args()
    
    print("🚀 SeedVR Single GPU Inference Runner")
    print("=" * 50)
    
    # Check GPU availability
    if not check_gpu_availability():
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 Setup instructions:")
        print("1. Download model checkpoints to ./ckpts/")
        print("2. Download pos_emb.pt and neg_emb.pt")
        print("3. Place test videos in ./test_videos/")
        sys.exit(1)
    
    # Get GPU memory for recommendations
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Auto-determine settings if requested
    if args.auto_settings or args.res_h is None or args.res_w is None:
        recommendations = get_memory_recommendations(args.model, gpu_memory_gb)
        
        if args.res_h is None:
            args.res_h = recommendations["res_h"]
        if args.res_w is None:
            args.res_w = recommendations["res_w"]
        
        print(f"\n🎯 Auto-selected settings ({recommendations['note']}):")
        print(f"   Resolution: {args.res_h}x{args.res_w}")
        print(f"   Batch size: {args.batch_size}")
    
    # Check if checkpoint exists
    checkpoint_path = f"./ckpts/seedvr_ema_{args.model}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        print(f"Please download the {args.model.upper()} model checkpoint")
        sys.exit(1)
    
    # Construct command
    script_name = f"projects/inference_seedvr_{args.model}_single_gpu.py"
    
    if not os.path.exists(script_name):
        print(f"❌ Single GPU inference script not found: {script_name}")
        sys.exit(1)
    
    cmd = [
        sys.executable, script_name,
        "--video_path", args.video_path,
        "--output_dir", args.output_dir,
        "--cfg_scale", str(args.cfg_scale),
        "--sample_steps", str(args.sample_steps),
        "--seed", str(args.seed),
        "--res_h", str(args.res_h),
        "--res_w", str(args.res_w),
        "--batch_size", str(args.batch_size),
    ]
    
    print(f"\n🔧 Running inference with {args.model.upper()} model:")
    print(f"   Input: {args.video_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Resolution: {args.res_h}x{args.res_w}")
    print(f"   CFG Scale: {args.cfg_scale}")
    print(f"   Steps: {args.sample_steps}")
    print(f"   Seed: {args.seed}")
    
    if args.dry_run:
        print(f"\n🔍 Dry run - Command that would be executed:")
        print(" ".join(cmd))
        return
    
    print(f"\n▶️  Starting inference...")
    print("=" * 50)
    
    try:
        # Run the inference script
        result = subprocess.run(cmd, check=True)
        print("=" * 50)
        print("✅ Inference completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except subprocess.CalledProcessError as e:
        print("=" * 50)
        print(f"❌ Inference failed with exit code {e.returncode}")
        print("\n💡 Troubleshooting tips:")
        print("1. Try reducing resolution (--res_h, --res_w)")
        print("2. Ensure sufficient GPU memory")
        print("3. Check that all required files are present")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Inference interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
