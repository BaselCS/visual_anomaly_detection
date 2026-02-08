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
warnings.filterwarnings('ignore')

# 1. Auto-detect trained models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
trained_models_dir = "trained_models"

print("Detecting trained models...")
all_categories = ["bottle", "capsule", "pill", "toothbrush"]
available_models = {}

for category in all_categories:
    # Check for best model first, then final model
    best_model = os.path.join(trained_models_dir, f"{category}_best_model")
    final_model = os.path.join(trained_models_dir, f"{category}_final_model")
    
    if os.path.exists(best_model):
        available_models[category] = best_model
        print(f"  ✓ Found {category} model (best): {best_model}")
    elif os.path.exists(final_model):
        available_models[category] = final_model
        print(f"  ✓ Found {category} model (final): {final_model}")

if not available_models:
    print("ERROR: No trained models found!")
    print("Please run main.py first to train the models.")
    print(f"Expected format: {trained_models_dir}/<category>_best_model/")
    exit(1)

print(f"\n✓ Found {len(available_models)} trained models")
print(f"Will test categories: {', '.join(available_models.keys())}\n")

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

print("\nStarting Anomaly Detection...\n" + "="*50)

# Test each category with its own trained model
for category, model_path in available_models.items():
    print(f"\n{'='*70}")
    print(f"📦 TESTING CATEGORY: {category.upper()}")
    print(f"   Model: {model_path}")
    print(f"{'='*70}")
    
    # Load the base model
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
        print(f"✓ Loaded trained model for {category}\n")
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Skipping this category...")
        continue
    test_dir = f"data/{category}/test"
    
    if not os.path.exists(test_dir):
        print(f"Warning: {test_dir} not found, skipping {category}")
        continue
    
    category_results = {}
    print(f"\n📦 Processing category: {category.upper()}")
    
    defect_types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    for defect_type in tqdm(defect_types, desc=f"  Testing {category}"):
        defect_path = os.path.join(test_dir, defect_type)
        
        image_files = [f for f in os.listdir(defect_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            img_path = os.path.join(defect_path, img_name)
            
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
                print(f"\n  Error processing {img_path}: {e}")
                continue
    
    all_results[category] = category_results
    print(f"  ✓ Processed {len(category_results)} images")
    
    # Clean up GPU memory before next category
    del pipe
    torch.cuda.empty_cache()

print("\n" + "="*50)

# 4. Automatic Threshold Detection & Metrics (حساب العتبة التلقائية والمقاييس)
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
    
    print(f"\n✓ Optimal Threshold: {optimal_threshold:.4f}")
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
    
    print(f"\n📈 Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
else:
    print("Warning: Not enough data for threshold calculation")
    optimal_threshold = np.median(all_scores)
    auc_score = None
    accuracy = None
    precision = None
    recall = None
    f1 = None

# 5. Category-wise Summary (ملخص حسب الفئات)
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
        
        print(f"\n  {category.upper()}:")
        print(f"    Good images:    {len(good_scores):3d} | Avg score: {avg_good:.4f}")
        print(f"    Defect images:  {len(defect_scores):3d} | Avg score: {avg_defect:.4f}")
        print(f"    Separation:     {separation:.4f} {'✓' if separation > 0.01 else '⚠'}")

# 6. Save Results (حفظ النتائج)
results_file = os.path.join(output_dir, "detection_results.json")
with open(results_file, "w") as f:
    json.dump({
        "optimal_threshold": float(optimal_threshold) if optimal_threshold is not None else None,
        "metrics": {
            "auc_roc": float(auc_score) if auc_score is not None else None,
            "accuracy": float(accuracy) if accuracy is not None else None,
            "precision": float(precision) if precision is not None else None,
            "recall": float(recall) if recall is not None else None,
            "f1_score": float(f1) if f1 is not None else None
        },
        "results": all_results
    }, f, indent=2)

print(f"\n✓ Results saved to {results_file}")

# 7. Final Assessment (التقييم النهائي)
print("\n" + "="*50)
if auc_score is not None:
    if auc_score > 0.8 and f1 > 0.7:
        print("🎉 SUCCESS: The model performs well at detecting anomalies!")
    elif auc_score > 0.6:
        print("⚠ MODERATE: The model shows some ability to detect defects.")
        print("   Consider training for more epochs or adjusting parameters.")
    else:
        print("❌ POOR: The model needs significant improvement.")
        print("   Recommendations:")
        print("   - Train for more epochs (current: check training_log.json)")
        print("   - Increase batch size if GPU memory allows")
        print("   - Verify data quality and variety")
else:
    print("⚠ INSUFFICIENT DATA: Need both good and defect samples to evaluate.")
print("="*50)



.
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   ├── broken_small
│   │   └── contamination
│   ├── license.txt
│   ├── readme.txt
│   ├── test
│   │   ├── broken_large
│   │   ├── broken_small
│   │   ├── contamination
│   │   └── good
│   └── train
│       └── good
├── capsule
│   ├── ground_truth
│   │   ├── crack
│   │   ├── faulty_imprint
│   │   ├── poke