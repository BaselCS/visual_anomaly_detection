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

# 1. Setup & Load Model (تحميل النموذج المدرب)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
model_path = "trained_models/final_model"  # Path to your trained LoRA model

print("Loading trained model...")
if not os.path.exists(model_path):
    print(f"ERROR: Trained model not found at {model_path}")
    print("Please run main.py first to train the model.")
    exit(1)

# Load the base model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

# Load the trained LoRA weights
try:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path)
    print("✓ Trained model loaded successfully")
except Exception as e:
    print(f"Error loading LoRA weights: {e}")
    print("Using base model without fine-tuning...")


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
categories = ["bottle", "capsule", "pill", "toothbrush"]
all_results = {}
all_scores = []
all_labels = []  # 0 for good, 1 for defect

output_dir = "anomaly_detection_results"
os.makedirs(output_dir, exist_ok=True)

print("\nStarting Anomaly Detection...\n" + "="*50)

for category in categories:
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
                    strength=0.6,  # Increased for better reconstruction
                    guidance_scale=7.5,
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

print("\n" + "="*50)

# 4. Display Summary (ملخص النتائج)
print("\n--- Final Results ---")
avg_good_score = np.mean([r['score'] for r in results.values() if r['type'] == 'good'])
avg_bad_score = np.mean([r['score'] for r in results.values() if r['type'] != 'good'])

print(f"Average Anomaly Score for GOOD images: {avg_good_score:.4f}")
print(f"Average Anomaly Score for DEFECTIVE images: {avg_bad_score:.4f}")

if avg_bad_score > avg_good_score:
    print("SUCCESS: The model successfully distinguishes defects!")
else:
    print("WARNING: The model needs more training or better threshold adjustment.")