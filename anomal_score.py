import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch import nn
from peft import PeftModel
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

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

# 1. Auto-detect trained models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
trained_models_dir = "trained_models"

logger.info("Detecting trained models...")
print("Detecting trained models...")

# Check for best model first, then final model, then checkpoints
best_model_path = os.path.join(trained_models_dir, "best_model")
final_model_path = os.path.join(trained_models_dir, "final_model")

model_path = None
model_type = None

if os.path.exists(best_model_path):
    model_path = best_model_path
    model_type = "best"
    logger.info(f"  ✓ Found best model: {best_model_path}")
    print(f"  ✓ Found best model: {best_model_path}")
elif os.path.exists(final_model_path):
    model_path = final_model_path
    model_type = "final"
    logger.info(f"  ✓ Found final model: {final_model_path}")
    print(f"  ✓ Found final model: {final_model_path}")
else:
    # Check for checkpoint with _BEST suffix
    checkpoints = [d for d in os.listdir(trained_models_dir) 
                   if d.startswith("checkpoint_epoch_") and "_BEST" in d]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_")[2]), reverse=True)
        model_path = os.path.join(trained_models_dir, checkpoints[0])
        model_type = "checkpoint"
        logger.info(f"  ✓ Found best checkpoint: {model_path}")
        print(f"  ✓ Found best checkpoint: {model_path}")

if not model_path:
    logger.error("ERROR: No trained models found!")
    print("ERROR: No trained models found!")
    logger.error("Please run main.py first to train the model.")
    print("Please run main.py first to train the model.")
    logger.error(f"Expected: {trained_models_dir}/best_model/ or {trained_models_dir}/final_model/")
    print(f"Expected: {trained_models_dir}/best_model/ or {trained_models_dir}/final_model/")
    exit(1)

# Read training config to get the trained categories
training_log_path = os.path.join(trained_models_dir, "training_log.json")
if os.path.exists(training_log_path):
    with open(training_log_path, 'r') as f:
        training_log = json.load(f)
        trained_categories = training_log.get("config", {}).get("categories", ["bottle"])
        best_epoch = training_log.get("best_epoch", "unknown")
        best_loss = training_log.get("best_loss", "unknown")
        logger.info(f"  Training info: Best epoch {best_epoch} with loss {best_loss}")
        print(f"  Training info: Best epoch {best_epoch} with loss {best_loss}")
else:
    trained_categories = ["bottle"]  # Default

logger.info(f"\n✓ Using {model_type} model")
print(f"\n✓ Using {model_type} model")
logger.info(f"Will test categories: {', '.join(trained_categories)}\n")
print(f"Will test categories: {', '.join(trained_categories)}\n")

# Map categories to the model
available_models = {category: model_path for category in trained_categories}

# 2. Anomaly Score Function (دالة حساب الخطأ)
def calculate_anomaly_score(original_image, reconstructed_image):
    """Calculate pixel-wise difference between original and reconstructed images"""
    # تحويل الصور إلى مصفوفات NumPy
    org_np = np.array(original_image.resize((512, 512))).astype(np.float32) / 255.0
    rec_np = np.array(reconstructed_image.resize((512, 512))).astype(np.float32) / 255.0
    
    # حساب الفرق (Multiple metrics)
    l1_diff = np.abs(org_np - rec_np)
    l2_diff = (org_np - rec_np) ** 2
    
    # Aggregate scores
    l1_score = np.mean(l1_diff)
    l2_score = np.mean(l2_diff)
    max_diff = np.max(l1_diff)
    
    # Combine scores (weighted average)
    combined_score = 0.6 * l1_score + 0.3 * l2_score + 0.1 * max_diff
    
    return combined_score, l1_diff

# 3. Process Test Images (معالجة صور الاختبار)
all_results = {}
all_scores = []
all_labels = []  # 0 for good, 1 for defect

output_dir = "anomaly_detection_results"
os.makedirs(output_dir, exist_ok=True)

logger.info("\nStarting Anomaly Detection...\n" + "="*50)
print("\nStarting Anomaly Detection...\n" + "="*50)

# Test each category with its own trained model
for category, model_path in available_models.items():
    logger.info(f"\n{'='*70}")
    print(f"\n{'='*70}")
    logger.info(f"📦 TESTING CATEGORY: {category.upper()}")
    print(f"📦 TESTING CATEGORY: {category.upper()}")
    logger.info(f"   Model: {model_path}")
    print(f"   Model: {model_path}")
    logger.info(f"{'='*70}")
    print(f"{'='*70}")
    
    # Load the base model
    logger.info("Loading base model...")
    print("Loading base model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Load the trained LoRA weights for this category
    try:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path)
        logger.info(f"✓ Loaded trained model for {category}\n")
        print(f"✓ Loaded trained model for {category}\n")
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
    logger.info(f"\n📦 Processing category: {category.upper()}")
    print(f"\n📦 Processing category: {category.upper()}")
    
    # Get all test images for this category from flat directory
    # Format: category_testtype_0001.png
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
                # أ. قراءة الصورة الأصلية
                original_image = Image.open(img_path).convert("RGB")
                
                # ب. إعادة البناء (Reconstruction)
                prompt = f"a high quality photo of a perfect {category}"
                reconstructed_image = pipe(
                    prompt=prompt, 
                    image=original_image, 
                    strength=0.4,  # REDUCED to preserve more structure and defects
                    guidance_scale=6.5,  # Slightly reduced for less aggressive reconstruction
                    num_inference_steps=30
                ).images[0]
                
                # ج. حساب السكور
                score, diff_map = calculate_anomaly_score(original_image, reconstructed_image)
                
                # Store results
                label = 0 if defect_type == "good" else 1
                all_scores.append(score)
                all_labels.append(label)
                
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
    
    # Clean up GPU memory before next category
    del pipe
    torch.cuda.empty_cache()

logger.info("\n" + "="*50)
print("\n" + "="*50)

# 4. Automatic Threshold Detection & Metrics (حساب العتبة التلقائية والمقاييس)
logger.info("\n📊 Calculating automatic threshold and metrics...")
print("\n📊 Calculating automatic threshold and metrics...")

# Calculate optimal threshold using ROC curve
if len(set(all_labels)) > 1:  # Make sure we have both classes
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate AUC-ROC
    auc_score = roc_auc_score(all_labels, all_scores)
    
    logger.info(f"\n✓ Optimal Threshold: {optimal_threshold:.4f}")
    print(f"\n✓ Optimal Threshold: {optimal_threshold:.4f}")
    logger.info(f"✓ AUC-ROC Score: {auc_score:.4f}")
    print(f"✓ AUC-ROC Score: {auc_score:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=100, 
                label=f'Optimal Threshold = {optimal_threshold:.4f}', zorder=5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Anomaly Detection Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    logger.info(f"✓ ROC curve saved to {output_dir}/roc_curve.png")
    print(f"✓ ROC curve saved to {output_dir}/roc_curve.png")
    
    # Calculate confusion matrix metrics
    predictions = [1 if score > optimal_threshold else 0 for score in all_scores]
    tp = sum([1 for i in range(len(all_labels)) if all_labels[i] == 1 and predictions[i] == 1])
    tn = sum([1 for i in range(len(all_labels)) if all_labels[i] == 0 and predictions[i] == 0])
    fp = sum([1 for i in range(len(all_labels)) if all_labels[i] == 0 and predictions[i] == 1])
    fn = sum([1 for i in range(len(all_labels)) if all_labels[i] == 1 and predictions[i] == 0])
    
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\n📈 Performance Metrics:")
    print(f"\n📈 Performance Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    print(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    print(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
else:
    logger.warning("Warning: Not enough data for threshold calculation")
    print("Warning: Not enough data for threshold calculation")
    optimal_threshold = np.median(all_scores)
    auc_score = None
    accuracy = None
    precision = None
    recall = None
    f1 = None

# 5. Category-wise Summary (ملخص حسب الفئات)
logger.info(f"\n📋 Results by Category:")
print(f"\n📋 Results by Category:")
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

# 6. Save Results (حفظ النتائج)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_run_data = {
    "timestamp": timestamp,
    "model_type": model_type,
    "model_path": model_path,
    "optimal_threshold": float(optimal_threshold) if optimal_threshold is not None else None,
    "metrics": {
        "auc_roc": float(auc_score) if auc_score is not None else None,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "f1_score": float(f1) if f1 is not None else None
    },
    "results": all_results
}

# Find next available numbered file
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

# Get next available file
results_file, file_number = find_next_results_file(output_dir)

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
logger.info(f"✓ Total result files: {total_files}")
print(f"✓ Total result files: {total_files}")

# Generate training visualization and save with matching number
logger.info("\n📊 Generating training visualization...")
print("\n📊 Generating training visualization...")
try:
    # Load the training log
    training_log_path = os.path.join(trained_models_dir, "training_log.json")
    if os.path.exists(training_log_path):
        with open(training_log_path, 'r') as f:
            train_log = json.load(f)
        
        # Extract data
        epochs = [entry['epoch'] for entry in train_log['epochs']]
        losses = [entry['avg_loss'] for entry in train_log['epochs']]
        learning_rates = [entry['learning_rate'] for entry in train_log['epochs']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training loss
        ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Average Loss', fontsize=12)
        ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # Add min loss annotation
        min_loss = min(losses)
        min_epoch = epochs[losses.index(min_loss)]
        ax1.axhline(y=min_loss, color='r', linestyle='--', alpha=0.5, 
                    label=f'Min Loss: {min_loss:.6f} (Epoch {min_epoch})')
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
        
        logger.info(f"✓ Training visualization saved to {viz_file}")
        print(f"✓ Training visualization saved to {viz_file}")
    else:
        logger.warning(f"Training log not found at {training_log_path}")
        print(f"Training log not found at {training_log_path}")
except Exception as e:
    logger.error(f"Failed to generate training visualization: {e}")
    print(f"Failed to generate training visualization: {e}")

# 7. Final Assessment (التقييم النهائي)
logger.info("\n" + "="*50)
print("\n" + "="*50)
if auc_score is not None:
    if auc_score > 0.8 and f1 > 0.7:
        logger.info("🎉 SUCCESS: The model performs well at detecting anomalies!")
        print("🎉 SUCCESS: The model performs well at detecting anomalies!")
    elif auc_score > 0.6:
        logger.warning("⚠ MODERATE: The model shows some ability to detect defects.")
        print("⚠ MODERATE: The model shows some ability to detect defects.")
        logger.info("   Consider training for more epochs or adjusting parameters.")
        print("   Consider training for more epochs or adjusting parameters.")
    else:
        logger.warning("❌ POOR: The model needs significant improvement.")
        print("❌ POOR: The model needs significant improvement.")
        logger.info("   Recommendations:")
        print("   Recommendations:")
        logger.info("   - Train for more epochs (current: check training_log.json)")
        print("   - Train for more epochs (current: check training_log.json)")
        logger.info("   - Increase batch size if GPU memory allows")
        print("   - Increase batch size if GPU memory allows")
        logger.info("   - Verify data quality and variety")
        print("   - Verify data quality and variety")
else:
    logger.warning("⚠ INSUFFICIENT DATA: Need both good and defect samples to evaluate.")
    print("⚠ INSUFFICIENT DATA: Need both good and defect samples to evaluate.")
logger.info("="*50)
print("="*50)
