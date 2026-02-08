#!/usr/bin/env python3
"""
Quick script to resume training from a specific checkpoint.
Usage: python resume_training.py [epoch_number]
Example: python resume_training.py 40
"""
import sys
import os
import re

def find_latest_checkpoint(output_dir="trained_models"):
    """Find the latest checkpoint in the output directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint_epoch_"):
            match = re.search(r'checkpoint_epoch_(\d+)', item)
            if match:
                epoch_num = int(match.group(1))
                checkpoints.append((epoch_num, os.path.join(output_dir, item)))
    
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0]
    return None

def update_main_py(checkpoint_path):
    """Update main.py to resume from the specified checkpoint"""
    with open("main.py", "r") as f:
        content = f.read()
    
    # Update the resume_from_checkpoint line
    pattern = r'"resume_from_checkpoint":\s*[^,]+,'
    if checkpoint_path:
        replacement = f'"resume_from_checkpoint": "{checkpoint_path}",  # Set to checkpoint path or None'
    else:
        replacement = '"resume_from_checkpoint": None,  # Set to checkpoint path or None'
    
    new_content = re.sub(pattern, replacement, content)
    
    with open("main.py", "w") as f:
        f.write(new_content)
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("Resume Training Helper")
    print("="*60)
    
    # Check if epoch number is provided
    if len(sys.argv) > 1:
        epoch_num = int(sys.argv[1])
        checkpoint_path = f"trained_models/checkpoint_epoch_{epoch_num}"
        
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            print("\nAvailable checkpoints:")
            for item in sorted(os.listdir("trained_models")):
                if item.startswith("checkpoint_epoch_"):
                    print(f"  - {item}")
            sys.exit(1)
    else:
        # Find latest checkpoint
        latest = find_latest_checkpoint()
        if latest:
            epoch_num, checkpoint_path = latest
            print(f"\nFound latest checkpoint: epoch {epoch_num}")
        else:
            print("\n❌ No checkpoints found in trained_models/")
            sys.exit(1)
    
    print(f"\n📂 Checkpoint: {checkpoint_path}")
    print(f"📍 Will resume from epoch {epoch_num + 1}")
    
    # Update main.py
    if update_main_py(checkpoint_path):
        print(f"\n✅ Updated main.py to resume from epoch {epoch_num}")
        print(f"\nTo start training, run:")
        print(f"  uv run main.py")
        print(f"\nTraining will continue from epoch {epoch_num + 1} to 50")
    else:
        print("\n❌ Failed to update main.py")
        sys.exit(1)
