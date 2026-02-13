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

### 1. `setup_check.py`
Validates your environment and data setup. Run this before starting to ensure everything is ready.
```bash
uv run python setup_check.py
```

### 2. `download_model.py`
Pre-downloads the Stable Diffusion v1.5 model weights to your local cache. This avoids long download times during the first training run.
```bash
uv run python download_model.py
```

### 3. `moving_images.py`
Reorganizes raw data from category-specific folders into the flat directory structure (`data/train` and `data/test`) required by the training and evaluation scripts run it only once before first training.
```bash
uv run python moving_images.py
```

### 4. `main.py`
The primary training script. It trains a LoRA model on the provided images for anomaly reconstruction.
```bash
uv run python main.py
```

### 5. `check_progress.py`
Monitor training progress in real-time. It provides epoch status, loss trends, and estimated time remaining by reading the training logs.
```bash
uv run python check_progress.py
```

### 6. `resume_training.py`
Helper script to resume training from a specific checkpoint if it was interrupted ( not tested well yet, use with caution). Provide the epoch number to resume from as an argument.
```bash
uv run python resume_training.py [epoch_number]
```

### 7. `anomal_score.py`
Performs anomaly detection on the test dataset using the trained model. It calculates reconstruction errors, determines optimal thresholds using ROC curves, and saves visualizations.
```bash
uv run python anomal_score.py
```
