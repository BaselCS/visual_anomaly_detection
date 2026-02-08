# Visual Anomaly Detection

This project uses Stable Diffusion with LoRA fine-tuning to detect visual anomalies in manufactured products.

## Setup

1. **Install dependencies:**
```bash
uv pip install torch torchvision diffusers peft pillow transformers accelerate scikit-learn matplotlib tqdm
```

2. **Download the model (one-time setup):**
```bash
uv run download_model.py
```

This will download the Stable Diffusion model (~5GB) and cache it locally.

## Data Structure

Your data should follow this structure:
```
data/
├── bottle/
│   ├── train/good/          # Training images (defect-free)
│   └── test/
│       ├── good/            # Test images (defect-free)
│       ├── broken_large/    # Defect type 1
│       └── contamination/   # Defect type 2
├── capsule/
├── pill/
└── toothbrush/
```

## Usage

### 1. Validate Data
```bash
uv run utils.py
```
This checks your data structure and reports any issues.

### 2. Train the Model
```bash
uv run main.py
```

**Training Configuration:**
- Categories: bottle, capsule, pill, toothbrush
- Epochs: 50 (adjust in main.py)
- Batch size: 2 (optimized for RTX 3060 12GB)
- Effective batch size: 8 (with gradient accumulation)

**Output:**
- `trained_models/final_model/` - Final trained LoRA weights
- `trained_models/checkpoint_epoch_N/` - Intermediate checkpoints
- `trained_models/training_log.json` - Training metrics
- `trained_models/test_generations/` - Sample generated images

### 3. Test Anomaly Detection
```bash
uv run anomal_score.py
```

**Output:**
- `anomaly_detection_results/detection_results.json` - Detailed results
- `anomaly_detection_results/roc_curve.png` - Performance visualization
- Optimal threshold (automatically calculated)
- Performance metrics (AUC-ROC, Accuracy, F1-Score)

## Key Features

### Training (main.py)
- ✅ Multi-category training (single unified model)
- ✅ LoRA fine-tuning (only 0.18% parameters trained)
- ✅ Gradient accumulation for memory efficiency
- ✅ Automatic checkpointing every 10 epochs
- ✅ Progress tracking with tqdm
- ✅ Error handling and recovery
- ✅ Training log with metrics

### Anomaly Detection (anomal_score.py)
- ✅ Automatic threshold detection using ROC curve
- ✅ Multiple scoring metrics (L1, L2, max difference)
- ✅ Per-category performance analysis
- ✅ AUC-ROC, Accuracy, Precision, Recall, F1-Score
- ✅ Visual performance reports (ROC curve)
- ✅ Comprehensive JSON output

## Hardware Requirements

- **GPU:** RTX 3060 12GB (or equivalent)
- **RAM:** 16GB+ recommended
- **Storage:** ~10GB for model + datasets

## Optimization

The code is optimized for RTX 3060 12GB:
- Mixed precision (FP16) training
- Batch size: 2
- Gradient accumulation: 4 steps
- Safety checker disabled to save VRAM
- VAE and text encoder frozen

## Training Tips

1. **First Run:** Train for 50+ epochs to see meaningful results
2. **Monitoring:** Check `training_log.json` for loss progression
3. **Checkpoints:** Resume from `checkpoint_epoch_N` if training interrupted
4. **Data Quality:** Use `utils.py` to validate data before training

## Expected Performance

- **Good performance:** AUC-ROC > 0.8, F1 > 0.7
- **Moderate:** AUC-ROC > 0.6
- **Needs improvement:** AUC-ROC < 0.6

If performance is poor:
- Train for more epochs
- Verify data quality (use utils.py)
- Check that defects are visually distinguishable
- Ensure enough training samples per category

## Files

- `main.py` - Training script
- `anomal_score.py` - Anomaly detection and evaluation
- `download_model.py` - One-time model download
- `utils.py` - Data validation utilities
- `pyproject.toml` - Project dependencies

## Troubleshooting

**Model downloads every time:**
- Run `download_model.py` once to cache the model
- Model cache location: `~/.cache/huggingface/hub/`

**Out of memory:**
- Reduce batch_size in main.py
- Increase gradient_accumulation_steps

**Poor performance:**
- Train for more epochs (50-100)
- Verify data quality with utils.py
- Check that "good" samples are truly defect-free

## License

This project uses:
- Stable Diffusion v1.5 (CreativeML Open RAIL-M)
- Your data should comply with the MVTec AD license terms
