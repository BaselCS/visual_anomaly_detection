#!/bin/bash
# Safe training runner with auto-recovery and logging
# Usage: ./run_training.sh

set -e  # Exit on error

LOGFILE="training_output.log"
SCRIPT="main.py"

echo "=========================================="
echo "Visual Anomaly Detection Training Runner"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOGFILE"
echo ""

# Check if training data exists
if [ ! -d "data/train" ]; then
    echo "ERROR: data/train directory not found!"
    exit 1
fi

# Check GPU availability
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || {
    echo "WARNING: nvidia-smi not available or no GPU detected"
}
echo ""

# Run training with uv, redirect output to log file AND terminal
echo "Starting training..."
echo "Press Ctrl+C to stop (model will be saved at last checkpoint)"
echo ""

# Run with unbuffered output and tee to both file and screen
uv run python -u "$SCRIPT" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "Training finished!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Full log saved to: $LOGFILE"
echo "=========================================="

exit $EXIT_CODE
