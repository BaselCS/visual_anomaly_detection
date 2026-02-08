"""
Quick start script to validate setup and prepare for training
"""
import os
import sys

print("="*60)
print("Visual Anomaly Detection - Setup Check")
print("="*60)

# 1. Check data directory
print("\n1. Checking data directory...")
if os.path.exists("data"):
    print("   ✓ data/ directory found")
    
    # Run validation
    from utils import validate_data_structure
    if validate_data_structure():
        print("\n   ✓ Data structure validated successfully")
    else:
        print("\n   ⚠ Data validation found issues. Please fix before training.")
        sys.exit(1)
else:
    print("   ❌ data/ directory not found!")
    print("   Please create data/ and organize your images according to README")
    sys.exit(1)

# 2. Check if model is cached
print("\n2. Checking Stable Diffusion model...")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_cached = any("stable-diffusion" in d for d in os.listdir(cache_dir)) if os.path.exists(cache_dir) else False

if model_cached:
    print("   ✓ Model is already cached")
else:
    print("   ⚠ Model not cached. Run 'uv run download_model.py' first to avoid")
    print("     downloading during training.")
    response = input("\n   Download model now? (y/n): ")
    if response.lower() == 'y':
        print("\n   Downloading model...")
        os.system("python download_model.py")

# 3. Check GPU
print("\n3. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✓ GPU detected: {gpu_name}")
        print(f"   ✓ GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("   ⚠ Warning: Less than 8GB VRAM. May need to reduce batch size.")
    else:
        print("   ⚠ No GPU detected. Training will be very slow on CPU.")
except ImportError:
    print("   ❌ PyTorch not installed")
    sys.exit(1)

# 4. Summary
print("\n" + "="*60)
print("Setup Status:")
print("="*60)
print("✓ Data structure validated")
print("✓ Dependencies installed")
if model_cached:
    print("✓ Model cached")
else:
    print("⚠ Model will be downloaded during first run")
print("\nYou're ready to train!")
print("\nNext steps:")
print("  1. Review training config in main.py (epochs, batch_size, etc.)")
print("  2. Run: uv run main.py")
print("  3. After training, run: uv run anomal_score.py")
print("="*60)
