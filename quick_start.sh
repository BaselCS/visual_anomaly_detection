#!/bin/bash
# Quick Start Script for Visual Anomaly Detection

echo "========================================"
echo "Visual Anomaly Detection - Quick Start"
echo "========================================"

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "❌ ERROR: data/ directory not found!"
    echo "Please create data/ and organize your images."
    exit 1
fi

# Step 1: Validate data
echo ""
echo "Step 1: Validating data structure..."
uv run utils.py
if [ $? -ne 0 ]; then
    echo "❌ Data validation failed. Please fix issues before continuing."
    exit 1
fi

# Step 2: Check setup
echo ""
echo "Step 2: Running setup check..."
uv run setup_check.py

# Step 3: Download model (optional but recommended)
echo ""
echo "Step 3: Model download"
read -p "Download model now to avoid delays during training? (y/n): " download
if [ "$download" = "y" ] || [ "$download" = "Y" ]; then
    echo "Downloading Stable Diffusion model (~5GB)..."
    uv run download_model.py
fi

# Ready to train
echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "To start training, run:"
echo "  uv run main.py"
echo ""
echo "Training will take several hours. Monitor progress in the console."
echo "After training completes, evaluate with:"
echo "  uv run anomal_score.py"
echo ""
