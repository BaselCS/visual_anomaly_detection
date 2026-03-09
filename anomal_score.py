import os
import torch
from torchvision import transforms
import numpy as np
import random
import hashlib
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch import nn
from peft import PeftModel
import lpips  
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
from datetime import datetime
import logging
import subprocess
import sys
from pytorch_msssim import ms_ssim

warnings.filterwarnings('ignore')


DEFAULT_EVAL_CONFIG = {
    "seed": 999,
    # "categories": ["bottle", "capsule", "pill", "toothbrush"],  # Test all trained categories
    "categories": ["bottle"],  # Test only specific categories (set to None to test all trained categories)
    "calibration_fraction": 0.3,                # Fraction of data used for calibration vs evaluation
    "reconstruction_strength": 0.35,            # Strength for img2img reconstruction (0.0 = perfect copy, 1.0 = full generation)
    "reconstruction_guidance_scale": 5.5,       # Guidance scale for reconstruction (higher = more faithful to prompt, but may reduce diversity)
    "reconstruction_steps": 30,                 # Number of steps for reconstruction (lower = faster but less refined)
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def out(message: str, level: str = "info") -> None:
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)
    print(message)

EVAL_CONFIG = dict(DEFAULT_EVAL_CONFIG)
EVAL_SEED = EVAL_CONFIG["seed"]
CALIBRATION_FRACTION = EVAL_CONFIG["calibration_fraction"]
RECONSTRUCTION_STRENGTH = EVAL_CONFIG["reconstruction_strength"]
RECONSTRUCTION_GUIDANCE_SCALE = EVAL_CONFIG["reconstruction_guidance_scale"]
RECONSTRUCTION_STEPS = EVAL_CONFIG["reconstruction_steps"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_latest_train_run(base_dir: str) -> str:
    """Return latest trainX directory if present, else return base_dir."""
    if not os.path.exists(base_dir):
        return base_dir

    train_dirs = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        if name.startswith("train") and name[5:].isdigit():
            train_dirs.append((int(name[5:]), full_path))

    if not train_dirs:
        return base_dir

    train_dirs.sort(key=lambda x: x[0], reverse=True)
    return train_dirs[0][1]


def resolve_model_path(run_models_dir: str) -> tuple[str | None, str | None]:
    """
    Check for best_model/ and final_model/ first, then look for checkpoint_epoch_XXX_BEST directories.
    """
    best_model_path = os.path.join(run_models_dir, "best_model")
    final_model_path = os.path.join(run_models_dir, "final_model")

    if os.path.exists(best_model_path):
        return best_model_path, "best"
    if os.path.exists(final_model_path):
        return final_model_path, "final"

    # Try BEST checkpoints first
    checkpoints = [d for d in os.listdir(run_models_dir) if d.startswith("checkpoint_epoch_") and "_BEST" in d]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[2]), reverse=True)
        return os.path.join(run_models_dir, checkpoints[0]), "checkpoint_best"

    # Fall back to any checkpoint (use the one with highest epoch number)
    all_checkpoints = [d for d in os.listdir(run_models_dir) if d.startswith("checkpoint_epoch_")]
    if all_checkpoints:
        # Extract epoch number and sort descending
        def get_epoch(name):
            parts = name.replace("_BEST", "").split("_")
            for p in parts:
                if p.isdigit():
                    return int(p)
            return 0
        all_checkpoints.sort(key=get_epoch, reverse=True)
        return os.path.join(run_models_dir, all_checkpoints[0]), "checkpoint_last"

    return None, None

def resolve_best_train_run(base_dir: str) -> str:
    """Select trainX run with best validation loss when available, else latest run."""
    if not os.path.exists(base_dir):
        return base_dir

    candidates = []
    latest = resolve_latest_train_run(base_dir)

    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not (os.path.isdir(full_path) and name.startswith("train") and name[5:].isdigit()):
            continue

        log_path = os.path.join(full_path, "training_log.json")
        best_model = os.path.join(full_path, "best_model")
        if not (os.path.exists(log_path) and os.path.exists(best_model)):
            continue

        try:
            with open(log_path, "r") as f:
                train_log = json.load(f)
            best_val_loss = float(train_log.get("best_val_loss", float("inf")))
            run_num = int(name[5:])
            candidates.append((best_val_loss, run_num, full_path))
        except Exception:
            continue

    if not candidates:
        return latest

    candidates.sort(key=lambda x: (x[0], -x[1]))
    return candidates[0][2]

# Stable integer hashing for consistent seeding based on category and image name
def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

# 1. Auto-detect trained models
set_seed(EVAL_SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
trained_models_dir = "trained_models"
run_models_dir = resolve_best_train_run(trained_models_dir)

out("Detecting trained models...")
if run_models_dir != trained_models_dir:
    out(f"Detected selected training run: {run_models_dir}")
else:
    out(f"Using default model directory: {run_models_dir}")

model_path, model_type = resolve_model_path(run_models_dir)
if model_path and model_type == "best":
    out(f"  ✓ Found best model: {model_path}")
elif model_path and model_type == "final":
    out(f"  ✓ Found final model: {model_path}")
elif model_path and model_type == "checkpoint_best":
    out(f"  ✓ Found best checkpoint: {model_path}")
elif model_path and model_type == "checkpoint_last":
    out(f"  ✓ Found last checkpoint (fallback): {model_path}")

if not model_path:
    out("ERROR: No trained models found!", level="error")
    out("Please run main.py first to train the model.", level="error")
    out(f"Expected: {run_models_dir}/best_model/ or {run_models_dir}/final_model/", level="error")
    raise RuntimeError("No trained models found. Please run main.py first to train the model.")

# Read training config to get the trained categories
training_log_path = os.path.join(run_models_dir, "training_log.json")
if os.path.exists(training_log_path):
    with open(training_log_path, 'r') as f:
        training_log = json.load(f)
        all_trained_categories = training_log.get("config", {}).get("categories", ["bottle"])
        best_epoch = training_log.get("best_epoch", "unknown")
        best_loss = training_log.get("best_val_loss", training_log.get("best_loss", "unknown"))
        out(f"  Training info: Best epoch {best_epoch} with loss {best_loss}")
else:
    all_trained_categories = ["bottle"]  # Default

# Filter categories based on config (if specified)
requested_categories = EVAL_CONFIG.get("categories")
if requested_categories:
    # Only test categories that were both requested AND trained
    trained_categories = [c for c in requested_categories if c in all_trained_categories]
    skipped = [c for c in requested_categories if c not in all_trained_categories]
    if skipped:
        out(f"  Warning: Skipping untrained categories: {', '.join(skipped)}", level="warning")
    if not trained_categories:
        raise RuntimeError(f"None of the requested categories {requested_categories} were trained. Trained: {all_trained_categories}")
else:
    # Test all trained categories
    trained_categories = all_trained_categories

out(f"\n✓ Using {model_type} model")
out(f"Will test categories: {', '.join(trained_categories)}\n")

# Map categories to the model
available_models = {category: model_path for category in trained_categories}

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # used for LPIPS scoring 
# 2. Anomaly Score Function (دالة حساب الخطأ)
def calculate_anomaly_score(original_image, reconstructed_image):
    """Calculate pixel-wise difference between original and reconstructed images by combining L1, L2, and a lightweight SSIM-inspired structural similarity."""
    resize_op = transforms.Resize((512, 512))
    to_tensor = transforms.ToTensor()

    # تحويل الصور إلى Tensors
    org_t = to_tensor(resize_op(original_image)).to(device)
    rec_t = to_tensor(resize_op(reconstructed_image)).to(device)

    # --- حساب المقاييس التقليدية ---
    diff_t = torch.abs(org_t - rec_t)
    l1_score = float(diff_t.mean().item())
    l2_score = float(torch.mean((org_t - rec_t) ** 2).item())
    max_diff = float(diff_t.max().item())

    # --- حساب MS-SSIM ---
    org_batch = org_t.unsqueeze(0)
    rec_batch = rec_t.unsqueeze(0)
    msssim_value = float(ms_ssim(org_batch, rec_batch, data_range=1.0, size_average=True).item())
    msssim_distance = 1.0 - msssim_value

    # --- حساب LPIPS (المقياس الإدراكي الجديد) ---
    # ملاحظة: LPIPS يتطلب أن تكون القيم في نطاق [-1, 1] بدلاً من [0, 1]
    org_lpips = (org_batch * 2) - 1
    rec_lpips = (rec_batch * 2) - 1
    
    with torch.no_grad():
        lpips_distance = float(loss_fn_vgg(org_lpips, rec_lpips).item())

    # --- دمج النتائج بالمعادلة الجديدة ---
    # قمت بتوزيع الأوزان لتعطي LPIPS دوراً محورياً في القرار
    combined_score = (
        0.30 * l1_score +         # التركيز على فروق الألوان
        0.10 * l2_score + 
        0.10 * max_diff + 
        0.25 * msssim_distance +  # التركيز على الهيكل
        0.25 * lpips_distance     # التركيز على التفاصيل الإدراكية (الجديد)
    )

    l1_diff = np.transpose(diff_t.detach().cpu().numpy(), (1, 2, 0))
    return combined_score, l1_diff

# 3. Testing Images & Scoring 
all_results = {}
all_records = []

output_dir = "anomaly_detection_results"
os.makedirs(output_dir, exist_ok=True)


def find_next_results_file(output_dir):
    """Find the next available detection_results file number."""
    base_file = os.path.join(output_dir, "detection_results.json")

    # If base file doesn't exist, use it
    if not os.path.exists(base_file):
        return base_file, 0

    # Find next available number
    counter = 1
    while True:
        numbered_file = os.path.join(output_dir, f"detection_results_{counter}.json")
        if not os.path.exists(numbered_file):
            return numbered_file, counter
        counter += 1


# Reserve output filenames for this run so artifacts share the same run number
results_file, file_number = find_next_results_file(output_dir)

logger.info("\nStarting Anomaly Detection...\n" + "="*50)
print("\nStarting Anomaly Detection...\n" + "="*50)

# Cache loaded pipelines by model path to avoid repeated full model loads
pipeline_cache = {}


def get_or_create_pipeline(model_path: str):
    """
    Load the Stable Diffusion pipeline with the specified LoRA weights, using caching to avoid repeated loads.
    """
    global device

    if model_path in pipeline_cache:
        return pipeline_cache[model_path]

    logger.info("Loading base model...")
    print("Loading base model...")

    def load_pipe(target_device: str):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if target_device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to(target_device)
        return pipe

    def run_model_download() -> None:
        out("Base model not found in local cache. Running download_model.py automatically...", level="warning")
        subprocess.run([sys.executable, "download_model.py"], check=True)
        out("Model download completed. Retrying pipeline load...")

    download_attempted = False
    try:
        pipe = load_pipe(device)
    except Exception as e:
        is_oom = "out of memory" in str(e).lower()
        if device == "cuda" and is_oom:
            logger.warning(f"GPU OOM while loading scoring pipeline ({e}). Falling back to CPU.")
            torch.cuda.empty_cache()
            device = "cpu"
            try:
                pipe = load_pipe(device)
            except Exception as cpu_error:
                if not download_attempted:
                    download_attempted = True
                    try:
                        run_model_download()
                        pipe = load_pipe(device)
                    except Exception as download_error:
                        raise RuntimeError(
                            "Could not load Stable Diffusion model and automatic download failed. "
                            f"Load error: {cpu_error}. Download error: {download_error}"
                        ) from download_error
                else:
                    raise RuntimeError(
                        "Could not load Stable Diffusion model from local cache after retry. "
                        f"Original error: {cpu_error}"
                    ) from cpu_error
        else:
            if not download_attempted:
                download_attempted = True
                try:
                    run_model_download()
                    pipe = load_pipe(device)
                except Exception as download_error:
                    raise RuntimeError(
                        "Could not load Stable Diffusion model and automatic download failed. "
                        f"Load error: {e}. Download error: {download_error}"
                    ) from download_error
            else:
                raise RuntimeError(
                    "Could not load Stable Diffusion model from local cache after retry. "
                    f"Original error: {e}"
                ) from e

    # Attach LoRA once for this model path
    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path)
    pipe.unet.eval()

    # Memory stability options for long evaluation runs
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    pipeline_cache[model_path] = pipe
    return pipe

# Test each category with its own trained model
for category, model_path in available_models.items():
    logger.info(f"\n{'='*70}")
    print(f"\n{'='*70}")
    logger.info(f" TESTING CATEGORY: {category.upper()}")
    print(f" TESTING CATEGORY: {category.upper()}")
    logger.info(f"   Model: {model_path}")
    print(f"   Model: {model_path}")
    logger.info(f"{'='*70}")
    print(f"{'='*70}")
    
    # Load or reuse base+LoRA model for this category
    try:
        pipe = get_or_create_pipeline(model_path)
        logger.info(f" Loaded trained model for {category}\n")
        print(f" Loaded trained model for {category}\n")
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        print(f"Error loading LoRA weights: {e}")
        logger.error("Skipping this category...")
        print("Skipping this category...")
        continue
    test_dir = "data/test"
    
    if not os.path.exists(test_dir):
        logger.warning(f"Warning: {test_dir} not found, skipping testing")
        print(f"Warning: {test_dir} not found, skipping testing")
        continue
    
    category_results = {}
    logger.info(f"\n Processing category: {category.upper()}")
    print(f"\n Processing category: {category.upper()}")
    
    # Get all test images for this category from flat directory
    # Format: category_testtype_001.png
    all_test_files = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith('.png') and f.startswith(f"{category}_")]
    
    # Group by test type
    test_groups = {}
    for img_name in all_test_files:
        # Parse: category_testtype_0001.png -> extract testtype
        parts = img_name.replace('.png', '').split('_')
        if len(parts) >= 3:
            # Join all parts between category and number as test_type
            test_type = '_'.join(parts[1:-1])
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(img_name)
    
    for defect_type, image_files in tqdm(test_groups.items(), desc=f"  Testing {category}"):
        for img_name in image_files:
            img_path = os.path.join(test_dir, img_name)
            
            try:
                original_image = Image.open(img_path).convert("RGB")
                
                prompt = f"a high quality photo of a perfect {category}"
                # reconstruct the image using the model and calculate anomaly score
                image_seed = EVAL_SEED + (stable_int(f"{category}/{img_name}") % 1_000_000)
                generator = torch.Generator(device=device)
                generator.manual_seed(image_seed)
                reconstructed_image = pipe(
                    prompt=prompt, 
                    image=original_image, 
                    strength=RECONSTRUCTION_STRENGTH,
                    guidance_scale=RECONSTRUCTION_GUIDANCE_SCALE,
                    num_inference_steps=RECONSTRUCTION_STEPS,
                    generator=generator,
                ).images[0]
                
              
                score, diff_map = calculate_anomaly_score(original_image, reconstructed_image)
                
                # Store results
                label = 0 if defect_type == "good" else 1
                all_records.append(
                    {
                        "category": category,
                        "image": img_name,
                        "type": defect_type,
                        "label": label,
                        "score": float(score),
                    }
                )
                
                key = f"{category}/{defect_type}/{img_name}"
                category_results[key] = {
                    "type": defect_type,
                    "score": float(score),
                    "is_defect": label == 1
                }
                
            except Exception as e:
                logger.error(f"\n  Error processing {img_path}: {e}")
                print(f"\n  Error processing {img_path}: {e}")
                continue
    
    all_results[category] = category_results
    logger.info(f"  ✓ Processed {len(category_results)} images")
    print(f"  ✓ Processed {len(category_results)} images")
    
    # Keep cached pipeline loaded to avoid expensive and unstable reloads

logger.info("\n" + "="*50)
print("\n" + "="*50)

# 4. Automatic Threshold Detection & Metrics Calculation 
logger.info("\n Calculating automatic threshold and metrics...")
print("\n Calculating automatic threshold and metrics...")

by_category_records = {}
for record in all_records:
    by_category_records.setdefault(record["category"], []).append(record)


def split_calibration_eval(records, category):
    """
    Split records into calibration and evaluation sets while ensuring both sets contain samples from both classes (good and defect) for the given category.
    """
    good = [r for r in records if r["label"] == 0]
    defect = [r for r in records if r["label"] == 1]

    rng = random.Random(EVAL_SEED + (stable_int(category) % 100000))
    rng.shuffle(good)
    rng.shuffle(defect)

    if min(len(good), len(defect)) < 2:
        return records, records, "fallback_insufficient_samples"

    good_calib = max(1, int(round(len(good) * CALIBRATION_FRACTION)))
    defect_calib = max(1, int(round(len(defect) * CALIBRATION_FRACTION)))

    good_calib = min(good_calib, len(good) - 1)
    defect_calib = min(defect_calib, len(defect) - 1)

    calib = good[:good_calib] + defect[:defect_calib]
    evaluation = good[good_calib:] + defect[defect_calib:]
    return calib, evaluation, None


def youden_threshold(labels, oriented_scores):
    """
    Calculate the optimal threshold using Youden's J statistic from the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(labels, oriented_scores)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return thresholds[idx], fpr, tpr, idx


per_category_thresholds = {}
eval_labels = []
eval_oriented_scores = []
eval_predictions = []

calibration_direction_summary = {}

# Determine direction and threshold for each category, then apply to evaluation set
for category, records in by_category_records.items():
    calib_records, eval_records, split_note = split_calibration_eval(records, category)
    calib_labels = [r["label"] for r in calib_records]
    calib_scores = [r["score"] for r in calib_records]

    if len(set(calib_labels)) <= 1:
        logger.warning(f"Category {category}: missing both classes in calibration, using median fallback")
        direction = "higher_is_defect"
        threshold = float(np.median(calib_scores)) if calib_scores else 0.0
        calib_auc = None
    else:
        auc_direct = roc_auc_score(calib_labels, calib_scores)
        auc_inverted = roc_auc_score(calib_labels, [-s for s in calib_scores])

        if auc_inverted > auc_direct:
            direction = "lower_is_defect"
            oriented_calib_scores = [-s for s in calib_scores]
            calib_auc = auc_inverted
        else:
            direction = "higher_is_defect"
            oriented_calib_scores = calib_scores
            calib_auc = auc_direct

        threshold, _, _, _ = youden_threshold(calib_labels, oriented_calib_scores)

    calibration_direction_summary[category] = {
        "direction": direction,
        "calibration_auc": float(calib_auc) if calib_auc is not None else None,
        "split_note": split_note,
        "calibration_count": len(calib_records),
        "evaluation_count": len(eval_records),
    }
    per_category_thresholds[category] = float(threshold)

    for record in eval_records:
        score = record["score"]
        oriented_score = -score if direction == "lower_is_defect" else score
        pred = 1 if oriented_score > threshold else 0

        eval_labels.append(record["label"])
        eval_oriented_scores.append(oriented_score)
        eval_predictions.append(pred)

overall_threshold_proxy = float(np.mean(list(per_category_thresholds.values()))) if per_category_thresholds else 0.0
optimal_threshold = overall_threshold_proxy

# Calculate metrics only if we have both classes in the evaluation set
if len(eval_labels) > 0 and len(set(eval_labels)) > 1:
    auc_score = roc_auc_score(eval_labels, eval_oriented_scores)
    fpr, tpr, thresholds = roc_curve(eval_labels, eval_oriented_scores)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))

    logger.info(f"\n Proxy Threshold (mean of per-category thresholds): {optimal_threshold:.4f}")
    print(f"\n Proxy Threshold (mean of per-category thresholds): {optimal_threshold:.4f}")
    logger.info(f" AUC-ROC Score (evaluation split): {auc_score:.4f}")
    print(f" AUC-ROC Score (evaluation split): {auc_score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        c='red',
        s=100,
        label='Best eval-split operating point',
        zorder=5,
    )
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Detection Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if file_number == 0:
        roc_file = os.path.join(output_dir, "roc_curve.png")
    else:
        roc_file = os.path.join(output_dir, f"roc_curve_{file_number}.png")

    plt.savefig(roc_file, dpi=150)
    logger.info(f" ROC curve saved to {roc_file}")
    print(f" ROC curve saved to {roc_file}")

    tp = sum(1 for y, p in zip(eval_labels, eval_predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(eval_labels, eval_predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(eval_labels, eval_predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(eval_labels, eval_predictions) if y == 1 and p == 0)

    accuracy = (tp + tn) / len(eval_labels) if len(eval_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    tpr_val = recall
    tnr_val = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = 0.5 * (tpr_val + tnr_val)

    logger.info(f"\n Performance Metrics (evaluation split):")
    print(f"\n Performance Metrics (evaluation split):")
    logger.info(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"  Precision:         {precision:.4f}")
    print(f"  Precision:         {precision:.4f}")
    logger.info(f"  Recall:            {recall:.4f}")
    print(f"  Recall:            {recall:.4f}")
    logger.info(f"  F1-Score:          {f1:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
else:
    logger.warning("Warning: Not enough data for evaluation metrics")
    print("Warning: Not enough data for evaluation metrics")
    auc_score = None
    accuracy = None
    balanced_accuracy = None
    precision = None
    recall = None
    f1 = None

# 5. Category-wise Summary 
logger.info(f"\n Results by Category:")
print(f"\n Results by Category:")
for category, results in all_results.items():
    if not results:
        continue
    
    good_scores = [r['score'] for r in results.values() if not r['is_defect']]
    defect_scores = [r['score'] for r in results.values() if r['is_defect']]
    
    if good_scores and defect_scores:
        avg_good = np.mean(good_scores)
        avg_defect = np.mean(defect_scores)
        separation = avg_defect - avg_good
        
        logger.info(f"\n  {category.upper()}:")
        print(f"\n  {category.upper()}:")
        logger.info(f"    Good images:    {len(good_scores):3d} | Avg score: {avg_good:.4f}")
        print(f"    Good images:    {len(good_scores):3d} | Avg score: {avg_good:.4f}")
        logger.info(f"    Defect images:  {len(defect_scores):3d} | Avg score: {avg_defect:.4f}")
        print(f"    Defect images:  {len(defect_scores):3d} | Avg score: {avg_defect:.4f}")
        logger.info(f"    Separation:     {separation:.4f} {'✓' if separation > 0.01 else '⚠'}")
        print(f"    Separation:     {separation:.4f} {'✓' if separation > 0.01 else '⚠'}")

# 6. Save Results 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_run_data = {
    "timestamp": timestamp,
    "run_directory": run_models_dir,
    "model_type": model_type,
    "model_path": model_path,
    "evaluation_seed": EVAL_SEED,
    "calibration_fraction": CALIBRATION_FRACTION,
    "per_category_thresholds": per_category_thresholds,
    "calibration_summary": calibration_direction_summary,
    "optimal_threshold": float(optimal_threshold) if optimal_threshold is not None else None,
    "metrics": {
        "auc_roc": float(auc_score) if auc_score is not None else None,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "balanced_accuracy": float(balanced_accuracy) if balanced_accuracy is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1_score": float(f1) if f1 is not None else None
    },
    "results": all_results
}

# Save current run to the new numbered file
with open(results_file, "w") as f:
    json.dump(current_run_data, f, indent=2)

if file_number == 0:
    logger.info(f"\n✓ Results saved to {results_file}")
    print(f"\n✓ Results saved to {results_file}")
else:
    logger.info(f"\n✓ Results saved to {results_file} (run #{file_number})")
    print(f"\n✓ Results saved to {results_file} (run #{file_number})")

# Count total result files
total_files = len([f for f in os.listdir(output_dir) 
                   if f.startswith("detection_results") and f.endswith(".json")])
logger.info(f" Total result files: {total_files}")
print(f" Total result files: {total_files}")

# Generate training visualization and save with matching number
logger.info("\n Generating training visualization...")
print("\n Generating training visualization...")
try:
    # Load the training log
    training_log_path = os.path.join(run_models_dir, "training_log.json")
    if os.path.exists(training_log_path):
        with open(training_log_path, 'r') as f:
            train_log = json.load(f)
        
        # Extract data
        epochs = [entry['epoch'] for entry in train_log['epochs']]
        train_losses = [entry.get('train_loss', entry.get('avg_loss')) for entry in train_log['epochs']]
        val_losses = [entry.get('val_loss', entry.get('validation_loss')) for entry in train_log['epochs']]
        learning_rates = [entry['learning_rate'] for entry in train_log['epochs']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training and validation loss
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7, label='Training Loss')
        if any(v is not None for v in val_losses):
            plotted_val_losses = [np.nan if v is None else v for v in val_losses]
            ax1.plot(epochs, plotted_val_losses, 'm-', linewidth=2, marker='x', markersize=4, alpha=0.8, label='Validation Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # Add minimum-loss annotations
        valid_train = [(i, v) for i, v in enumerate(train_losses) if v is not None]
        if valid_train:
            train_min_idx, min_train_loss = min(valid_train, key=lambda x: x[1])
            train_min_epoch = epochs[train_min_idx]
            ax1.axhline(
                y=min_train_loss,
                color='b',
                linestyle='--',
                alpha=0.35,
                label=f'Min Train: {min_train_loss:.6f} (Epoch {train_min_epoch})',
            )

        valid_val = [(i, v) for i, v in enumerate(val_losses) if v is not None]
        if valid_val:
            val_min_idx, min_val_loss = min(valid_val, key=lambda x: x[1])
            val_min_epoch = epochs[val_min_idx]
            ax1.axhline(
                y=min_val_loss,
                color='m',
                linestyle='--',
                alpha=0.35,
                label=f'Min Val: {min_val_loss:.6f} (Epoch {val_min_epoch})',
            )
        ax1.legend()
        
        # Plot learning rate
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2, marker='s', markersize=4, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Add training info as text
        info_text = f"Training Configuration:\n"
        info_text += f"Categories: {', '.join(train_log['config']['categories'])}\n"
        info_text += f"Epochs: {train_log['config']['epochs']}\n"
        info_text += f"Batch Size: {train_log['config']['batch_size']}\n"
        info_text += f"Initial LR: {train_log['config']['learning_rate']}\n"
        info_text += f"LR Scheduler: {train_log['config']['lr_scheduler']}"
        
        fig.text(0.98, 0.02, info_text, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        # Save with matching number
        if file_number == 0:
            viz_file = os.path.join(output_dir, "training_visualization.png")
        else:
            viz_file = os.path.join(output_dir, f"training_visualization_{file_number}.png")
        
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f" Training visualization saved to {viz_file}")
        print(f" Training visualization saved to {viz_file}")
    else:
        logger.warning(f"Training log not found at {training_log_path}")
        print(f"Training log not found at {training_log_path}")
except Exception as e:
    logger.error(f"Failed to generate training visualization: {e}")
    print(f"Failed to generate training visualization: {e}")

# 7. Final Assessment 
logger.info("\n" + "="*50)
print("\n" + "="*50)
if auc_score is not None:
    print(f"\nFinal Assessment:")
    logger.info(f"\nFinal Assessment:")
    logger.info(f"  AUC-ROC Score (evaluation split): {auc_score:.4f}")
    print(f"  AUC-ROC Score (evaluation split): {auc_score:.4f}")
    logger.info(f"  Proxy Threshold (mean per-category): {optimal_threshold:.4f}")
    print(f"  Proxy Threshold (mean per-category): {optimal_threshold:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    print(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    print(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    print(f"  F1-Score: {f1:.4f}")
else:
    logger.warning(" INSUFFICIENT DATA: Need both good and defect samples to evaluate.")
    print(" INSUFFICIENT DATA: Need both good and defect samples to evaluate.")
logger.info("="*50)
print("="*50)
