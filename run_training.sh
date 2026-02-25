#!/bin/bash
# End-to-end unattended runner: synthetic generation -> training -> testing
# Usage: ./run_training.sh

set -euo pipefail

LOGFILE="training_output.log"

# Best-known generation settings from past successful run (2026-02-24).
GEN_RATIO="0.3"
GEN_MAX_PER_CATEGORY="120"
GEN_MIN_BLUR="8.0"
GEN_MAX_TRIES_MULTIPLIER="8"
GEN_IMAGE_SIZE="512"
GEN_NUM_STEPS="30"
GEN_GUIDANCE_SCALE="7.0"
GEN_STRENGTH="0.22"
GEN_SEED="42"

echo "==========================================" | tee -a "$LOGFILE"
echo "Visual Anomaly Detection Full Pipeline" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
echo "Start time: $(date)" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

if [ ! -d "data/train" ]; then
    echo "ERROR: data/train directory not found!" | tee -a "$LOGFILE"
    exit 1
fi

if [ ! -d "data/test" ]; then
    echo "ERROR: data/test directory not found!" | tee -a "$LOGFILE"
    exit 1
fi

echo "Checking GPU..." | tee -a "$LOGFILE"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | tee -a "$LOGFILE" || {
    echo "WARNING: nvidia-smi not available or no GPU detected" | tee -a "$LOGFILE"
}
echo "" | tee -a "$LOGFILE"

echo "[1/3] Generating synthetic normal images..." | tee -a "$LOGFILE"
uv run python -u generate_synthetic_data.py \
    --ratio "$GEN_RATIO" \
    --max-per-category "$GEN_MAX_PER_CATEGORY" \
    --min-blur-score "$GEN_MIN_BLUR" \
    --max-tries-multiplier "$GEN_MAX_TRIES_MULTIPLIER" \
    --image-size "$GEN_IMAGE_SIZE" \
    --num-steps "$GEN_NUM_STEPS" \
    --guidance-scale "$GEN_GUIDANCE_SCALE" \
    --strength "$GEN_STRENGTH" \
    --seed "$GEN_SEED" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "[2/3] Training model..." | tee -a "$LOGFILE"
uv run python -u main.py 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "[3/3] Running anomaly testing..." | tee -a "$LOGFILE"
uv run python -u anomal_score.py 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
echo "Pipeline finished successfully" | tee -a "$LOGFILE"
echo "End time: $(date)" | tee -a "$LOGFILE"
echo "Full log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
