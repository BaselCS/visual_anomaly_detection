## download packages
plz use `uv` for better performance and stability than `pip`

```bash
uv sync
```

if you want to use `pip` instead, you can run:

```bash
pip install -r requirements.txt
```


## File Usage


### 1. `download_model.py`
Pre-downloads the Stable Diffusion v1.5 model weights to your local cache. This avoids long download times during the first training run.
```bash
uv run python download_model.py
```

### 2. `moving_images.py`
Reorganizes raw data from category-specific folders into the flat directory structure (`data/train` and `data/test`) required by the training and evaluation scripts run it only once before first training.
```bash
uv run python moving_images.py
```

### 3. `generate_synthetic_data.py`
Generates additional high-quality synthetic **normal** training images and appends them to `data/train` in the same filename format used by training.
Recommended first run (safe ratio):
```bash
uv run python generate_synthetic_data.py --ratio 0.3 --max-per-category 120
```

If you need to target specific categories only:
```bash
uv run python generate_synthetic_data.py --categories bottle capsule --ratio 0.25
```

### 4. `main.py`
The primary training script. It trains a LoRA model on the provided images for anomaly reconstruction.
```bash
uv run python main.py
```

### 5. `anomal_score.py`
Performs anomaly detection on the test dataset using the trained model. It calculates reconstruction errors, determines optimal thresholds using ROC curves, and saves visualizations.
```bash
uv run python anomal_score.py
```
