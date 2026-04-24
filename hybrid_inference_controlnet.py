import os
import torch
import json # مكتبة قراءة الملفات
from PIL import Image
import numpy as np
import cv2  
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from peft import PeftModel
import xgboost as xgb
from tqdm import tqdm
import warnings

from metrics_factory import MetricsFactory

warnings.filterwarnings('ignore')

# ==========================================
# الإعدادات (Configurations)
# ==========================================
STRENGTH = 0.42
GUIDANCE = 5.5
CONTROLNET_CONDITIONING_SCALE = 0.8 
CATEGORY = "bottle"

def get_latest_train_folder(base_dir="trained_models"):
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train") and os.path.isdir(os.path.join(base_dir, folder_name)):
            try:
                num = int(folder_name.replace("train", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if latest_folder is None:
        raise FileNotFoundError(f"No train directories found in {base_dir}")
    
    best_model_path = os.path.join(base_dir, latest_folder, "best_model")
    if os.path.exists(best_model_path):
        return latest_folder, best_model_path
    return latest_folder, os.path.join(base_dir, latest_folder, "final_model")

latest_folder_name, SD_MODEL_PATH = get_latest_train_folder()
latest_dir = os.path.join("trained_models", latest_folder_name)

# مسارات النماذج والبيانات الوصفية
XGB_MODEL_PATH = os.path.join(latest_dir, f"xgboost_hybrid_S{STRENGTH}_G{GUIDANCE}.json")
METADATA_PATH = os.path.join(latest_dir, f"xgboost_metadata_S{STRENGTH}_G{GUIDANCE}.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- 🏭 QASSAS HYBRID INFERENCE SYSTEM (ControlNet Edition) ---")

# ==========================================
# 1. تحميل النماذج والعتبة التلقائية
# ==========================================
# قراءة العتبة (Threshold) بشكل آلي
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        OPTIMAL_THRESHOLD = metadata['optimal_threshold']
    print(f"✅ Auto-Loaded Optimal Threshold: {OPTIMAL_THRESHOLD:.4f}")
except Exception as e:
    raise FileNotFoundError(f"⚠️ Could not load threshold metadata from {METADATA_PATH}. Please run XGBoost training first! Error: {e}")

print("Loading ControlNet (Canny Edge)...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
).to(device)

print(f"Loading Stable Diffusion with ControlNet...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

print(f"Applying LoRA Weights from: {SD_MODEL_PATH}")
pipe.unet = PeftModel.from_pretrained(pipe.unet, SD_MODEL_PATH)

metrics_gen = MetricsFactory(device=device)

try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    print(f"Loading XGBoost Classifier from: {XGB_MODEL_PATH}")
except Exception as e:
    raise FileNotFoundError(f"⚠️ Could not load XGBoost model. Please run train_hybrid_xgboost.py first! Error: {e}")

# ==========================================
# دالة استخراج الحواف
# ==========================================
def get_canny_image(image):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_cv, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

# ==========================================
# 2. الفحص المباشر (Live Inspection)
# ==========================================
test_dir = "data/test"
if os.path.exists(os.path.join(test_dir, CATEGORY)):
    test_dir = os.path.join(test_dir, CATEGORY)
    test_files = os.listdir(test_dir)
else:
    test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{CATEGORY}")]

print(f"\nStarting live inspection with ControlNet on {len(test_files)} images...")

correct_predictions = 0
total_images = len(test_files)

for img_name in tqdm(test_files, desc="Inspecting"):
    img_path = os.path.join(test_dir, img_name)
    true_label = 0 if "good" in img_name.lower() else 1
    
    original_image = Image.open(img_path).convert("RGB").resize((512, 512))
    control_image = get_canny_image(original_image)
    generator = torch.Generator(device=device).manual_seed(999)

    with torch.no_grad():
        reconstructed_image = pipe(
            prompt=f"a high quality photo of a perfect {CATEGORY}",
            image=original_image,            
            control_image=control_image,     
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE, 
            num_inference_steps=30,
            generator=generator,
        ).images[0]

    scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
    features_array = np.array([[scores['L1'], scores['L2'], scores['MS_SSIM'], scores['LPIPS'], scores['Max_Patch']]])
    
    anomaly_probability = xgb_model.predict_proba(features_array)[0, 1]
    predicted_label = 1 if anomaly_probability >= OPTIMAL_THRESHOLD else 0
    
    if predicted_label == true_label:
        correct_predictions += 1
    else:
        true_status = "Good" if true_label == 0 else "Anomaly"
        pred_status = "Anomaly" if predicted_label == 1 else "Good"
        print(f"\n⚠️ Mismatch on {img_name}: True: {true_status} | Predicted: {pred_status} (Prob: {anomaly_probability:.4f})")

print("\n" + "="*40)
print("--- 📊 CONTROLNET LIVE INFERENCE REPORT ---")
print(f"Total Images: {total_images}")
print(f"Live Accuracy: {(correct_predictions/total_images)*100:.2f}%")
print("========================================")