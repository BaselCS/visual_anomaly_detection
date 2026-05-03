#!/usr/bin/env python3
"""
Quick script to resume training from a specific checkpoint.
Usage: python resume_training.py [epoch_number]
Example: python resume_training.py 40
"""
import sys
import os
import re
import argparse


TRAINED_MODELS_DIR = "trained_models"
MAIN_FILE = "main.py"
RESUME_PATTERN = r'"resume_from_checkpoint":\s*[^,]+,'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume training helper")
    parser.add_argument("epoch", nargs="?", type=int, help="Checkpoint epoch to resume from")
    return parser.parse_args()

def find_latest_checkpoint(output_dir=TRAINED_MODELS_DIR):
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


def list_available_checkpoints(output_dir=TRAINED_MODELS_DIR) -> list[str]:
    if not os.path.exists(output_dir):
        return []
    return [item for item in sorted(os.listdir(output_dir)) if item.startswith("checkpoint_epoch_")]

def update_main_py(checkpoint_path):
    """Update main.py to resume from the specified checkpoint"""
    with open(MAIN_FILE, "r") as f:
        content = f.read()

    if checkpoint_path:
        replacement = f'"resume_from_checkpoint": "{checkpoint_path}",  # Set to checkpoint path or None'
    else:
        replacement = '"resume_from_checkpoint": None,  # Set to checkpoint path or None'

    new_content, replacements = re.subn(RESUME_PATTERN, replacement, content)
    if replacements == 0:
        return False

    with open(MAIN_FILE, "w") as f:
        f.write(new_content)

    return True


def resolve_checkpoint_from_args(epoch: int | None) -> tuple[int, str]:
    if epoch is not None:
        checkpoint_path = f"{TRAINED_MODELS_DIR}/checkpoint_epoch_{epoch}"
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ Checkpoint not found: {checkpoint_path}")
            print("\nAvailable checkpoints:")
            for item in list_available_checkpoints():
                print(f"  - {item}")
            sys.exit(1)
        return epoch, checkpoint_path

    latest = find_latest_checkpoint()
    if latest:
        epoch_num, checkpoint_path = latest
        print(f"\nFound latest checkpoint: epoch {epoch_num}")
        return epoch_num, checkpoint_path

    print(f"\n❌ No checkpoints found in {TRAINED_MODELS_DIR}/")
    sys.exit(1)

if __name__ == "__main__":
    args = parse_args()

    print("="*60)
    print("Resume Training Helper")
    print("="*60)

    epoch_num, checkpoint_path = resolve_checkpoint_from_args(args.epoch)
    
    print(f"\n📂 Checkpoint: {checkpoint_path}")
    print(f"📍 Will resume from epoch {epoch_num + 1}")
    
    if update_main_py(checkpoint_path):
        print(f"\n✅ Updated main.py to resume from epoch {epoch_num}")
        print(f"\nTo start training, run:")
        print(f"  uv run main.py")
        print(f"\nTraining will continue from epoch {epoch_num + 1} to 50")
    else:
        print("\n❌ Failed to update main.py (resume_from_checkpoint field not found)")
        sys.exit(1)
