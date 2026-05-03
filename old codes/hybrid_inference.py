import os
import torch
import json
from PIL import Image
import numpy as np
import glob
import re
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
import xgboost as xgb
from tqdm import tqdm
import warnings
import gc

from metrics_factory import MetricsFactory
import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

CATEGORY = "bottle"
PROMPT = "a high quality photo of a perfect bottle"


def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def get_latest_oft_dir(base_dir="trained_models"):
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_oft_"):
            try:
                num = int(folder_name.replace("train_oft_", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if not latest_folder: raise FileNotFoundError("No OFT training found.")
    return os.path.join(base_dir, latest_folder)

def generate_anomaly_map(original_image, reconstructed_image, save_path=None, threshold_value=70):
    # 1. تحويل الصور إلى مصفوفات رياضية
    orig_np = np.array(original_image).astype(np.float32)
    recon_np = np.array(reconstructed_image).astype(np.float32)

    # 2. حساب الفرق المطلق بين الأصلية والمصلحة
    diff = np.abs(orig_np - recon_np)

    # 3. تحويل إلى قناة أحادية
    diff_gray = np.mean(diff, axis=2)

    # 4. تنعيم الصورة لتقليل ضوضاء الكاميرا الخفيفة
    diff_smoothed = cv2.GaussianBlur(diff_gray, (15, 15), 0)

    # 5. تطبيع الأرقام لتوسيع نطاق الاختلافات (0-255)
    diff_normalized = cv2.normalize(diff_smoothed, None, 0, 255, cv2.NORM_MINMAX)
    diff_8u = np.uint8(diff_normalized)

    # 6. التصفية الثنائية (Binary Thresholding) - السر هنا!
    # أي بيكسل يتجاوز الـ 70 سيصبح أبيض (255)، والباقي أسود (0)
    _, binary_mask = cv2.threshold(diff_8u, threshold_value, 255, cv2.THRESH_BINARY)

    # 7. تحويل المصفوفة إلى صورة أبيض وأسود صريحة
    bw_pil = Image.fromarray(binary_mask, mode='L')
    
    if save_path:
        bw_pil.save(save_path)
        
    return bw_pil


latest_dir = get_latest_oft_dir()

metadata_search_pattern = os.path.join(latest_dir, "xgboost_metadata_*.json")
metadata_files = glob.glob(metadata_search_pattern)

if not metadata_files:
    raise FileNotFoundError(f"Could not find metadata file in {latest_dir}.")

METADATA_PATH = max(metadata_files, key=os.path.getctime)
match = re.search(r"S([\d\.]+)_G([\d\.]+)\.json", os.path.basename(METADATA_PATH))

if match:
    STRENGTH = float(match.group(1))
    GUIDANCE = float(match.group(2))
else:
    raise ValueError("Could not extract Strength and Guidance.")

XGB_MODEL_PATH = METADATA_PATH.replace("xgboost_metadata_", "xgboost_hybrid_")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- QASSAS HYBRID INFERENCE SYSTEM (OFT Edition) ---")
print(f"Auto-Detected Config - Strength: {STRENGTH}, Guidance: {GUIDANCE}")

try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        OPTIMAL_THRESHOLD = metadata['optimal_threshold']
    print(f"Auto-Loaded Optimal Threshold: {OPTIMAL_THRESHOLD:.4f}")
except Exception as e:
    raise FileNotFoundError(f"Could not load threshold. Error: {e}")

print("Loading Base Pipeline (Safety Checker DISABLED)...")
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, 
        local_files_only=True,
        safety_checker=None, requires_safety_checker=False
    ).to(device)
except:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    ).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

# اختراق الحماية داخلياً
def dummy_checker(image, device, dtype): return image, [False] * len(image)
pipe.run_safety_checker = dummy_checker

print(f"Applying OFT Weights to U-Net from: {latest_dir}")
pipe.unet = PeftModel.from_pretrained(pipe.unet, latest_dir)
pipe.unet.to(device, dtype=torch.float16)

metrics_gen = MetricsFactory(device=device)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)
print(f"Loading XGBoost Classifier from: {XGB_MODEL_PATH}")

test_dir = "data/test"
if os.path.exists(os.path.join(test_dir, CATEGORY)):
    test_dir = os.path.join(test_dir, CATEGORY)
    test_files = os.listdir(test_dir)
else:
    test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{CATEGORY}")]

print(f"\nStarting live inspection on {len(test_files)} images...")

correct_predictions = 0
total_images = len(test_files)

for img_name in tqdm(test_files, desc="Inspecting"):
    img_path = os.path.join(test_dir, img_name)
    true_label = 0 if "good" in img_name.lower() else 1
    
    original_image = Image.open(img_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(999)

    with torch.no_grad():
        reconstructed_image = pipe(
            prompt=PROMPT,
            image=original_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

    scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
    features_array = np.array([[scores['L1'], scores['L2'], scores['MS_SSIM'], scores['LPIPS'], scores['Max_Patch']]])
    
    anomaly_probability = xgb_model.predict_proba(features_array)[0, 1]
    predicted_label = 1 if anomaly_probability >= OPTIMAL_THRESHOLD else 0
    
    if predicted_label == 1: # إذا اكتشف XGBoost وجود شذوذ
        # إنشاء مجلد لحفظ الخرائط إذا لم يكن موجوداً
        os.makedirs("anomaly_maps_output", exist_ok=True)
        map_path = os.path.join("anomaly_maps_output", f"map_{img_name}")
        
        # استدعاء الدالة لتوليد وحفظ الخريطة
        generate_anomaly_map(original_image, reconstructed_image, save_path=map_path)
        
    if predicted_label == true_label:
        correct_predictions += 1
    else:
        true_status = "Good" if true_label == 0 else "Anomaly"
        pred_status = "Anomaly" if predicted_label == 1 else "Good"
        print(f"\nMismatch on {img_name}: True: {true_status} | Predicted: {pred_status} (Prob: {anomaly_probability:.4f})")
        
    clean_vram()

print("\n" + "="*40)
print("--- OFT LIVE INFERENCE REPORT ---")
print(f"Total Images: {total_images}")
print(f"Live Accuracy: {(correct_predictions/total_images)*100:.2f}%")
print("========================================")