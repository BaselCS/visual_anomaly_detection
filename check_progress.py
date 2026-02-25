#!/usr/bin/env python3
"""
Check training progress without interrupting the training process
Usage: uv run python check_progress.py
"""
import json
import os
from datetime import datetime


def resolve_latest_training_log(base_dir="trained_models"):
    """Return training_log.json path from latest trainX run if available."""
    if not os.path.isdir(base_dir):
        return os.path.join(base_dir, "training_log.json")

    train_dirs = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith("train") and name[5:].isdigit():
            train_dirs.append((int(name[5:]), full_path))

    if not train_dirs:
        return os.path.join(base_dir, "training_log.json")

    train_dirs.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(train_dirs[0][1], "training_log.json")

def format_duration(seconds):
    """Format seconds into human readable duration"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def check_progress():
    log_file = resolve_latest_training_log()
    
    print("="*60)
    print("Training Progress Check")
    print("="*60)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not os.path.exists(log_file):
        print("❌ No training log found. Training may not have started yet.")
        return
    
    try:
        with open(log_file, 'r') as f:
            log = json.load(f)
        
        # Parse times
        start_time = datetime.fromisoformat(log['start_time'])
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        # Get latest epoch info
        if not log['epochs']:
            print("⏳ Training started but no epochs completed yet...")
            print(f"Running for: {format_duration(elapsed)}")
            return
        
        latest = log['epochs'][-1]
        current_epoch = latest['epoch']
        total_epochs = log['config']['epochs']
        progress_pct = (current_epoch / total_epochs) * 100
        
        # Estimate time remaining
        time_per_epoch = elapsed / current_epoch
        remaining_epochs = total_epochs - current_epoch
        est_remaining = time_per_epoch * remaining_epochs
        
        print(f"📊 Progress: Epoch {current_epoch}/{total_epochs} ({progress_pct:.1f}%)")
        print(f"⏱️  Running time: {format_duration(elapsed)}")
        print(f"⏳ Estimated remaining: {format_duration(est_remaining)}")
        eta = datetime.fromtimestamp(current_time.timestamp() + est_remaining)
        print(f"🎯 Expected completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"📉 Current Training Loss: {latest['train_loss']:.4f}")
        print(f"📉 Current Validation Loss: {latest['val_loss']:.4f}")
        print(f"📚 Learning Rate: {latest['learning_rate']:.2e}")
        print()
        
        # Best model info
        if log.get('best_epoch'):
            print(f"⭐ Best Model:")
            print(f"   Epoch: {log['best_epoch']}")
            print(f"   Train Loss: {log['best_train_loss']:.4f}")
            print(f"   Val Loss: {log['best_val_loss']:.4f}")
            print()
        
        # Show last 5 epochs trend
        if len(log['epochs']) >= 5:
            print("📈 Recent 5 Epochs:")
            print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12}")
            print("-" * 32)
            for ep in log['epochs'][-5:]:
                print(f"{ep['epoch']:<8} {ep['train_loss']:<12.4f} {ep['val_loss']:<12.4f}")
        
        # Check for issues
        print()
        if current_epoch > 5:
            recent_val = [e['val_loss'] for e in log['epochs'][-5:]]
            if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
                print("⚠️  WARNING: Validation loss increasing for last 5 epochs!")
                print("   Model may be overfitting. Best model is saved.")
        
    except Exception as e:
        print(f"❌ Error reading log: {e}")

if __name__ == "__main__":
    check_progress()
