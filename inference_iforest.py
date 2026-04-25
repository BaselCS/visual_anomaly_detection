import os
import torch
import numpy as np
from PIL import Image
import cv2
import joblib
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from tqdm import tqdm
import warnings
import gc

from metrics_factory import MetricsFactory

warnings.filterwarnings('ignore')

# 1. إعدادات النظام (يجب أن تطابق ما استخرجنا به الخصائص)
CATEGORY = "bottle"
PROMPT = "a high quality photo of a perfect bottle"
TARGET_STRENGTH = 0.40
TARGET_GUIDANCE = 6.5

# العتبة المثالية التي استخرجتها من مرحلة المعايرة
OPTIMAL_THRESHOLD = 0.0578  

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

# دالة توليد خريطة الشذوذ بالأبيض والأسود
def generate_anomaly_map(original_image, reconstructed_image, save_path=None, threshold_value=70):
    orig_np = np.array(original_image).astype(np.float32)
    recon_np = np.array(reconstructed_image).astype(np.float32)

    diff = np.abs(orig_np - recon_np)
    diff_gray = np.mean(diff, axis=2)

    diff_smoothed = cv2.GaussianBlur(diff_gray, (15, 15), 0)
    diff_normalized = cv2.normalize(diff_smoothed, None, 0, 255, cv2.NORM_MINMAX)
    diff_8u = np.uint8(diff_normalized)

    _, binary_mask = cv2.threshold(diff_8u, threshold_value, 255, cv2.THRESH_BINARY)
    bw_pil = Image.fromarray(binary_mask, mode='L')
    
    if save_path:
        bw_pil.save(save_path)
        
    return bw_pil

latest_dir = get_latest_oft_dir()
IFOREST_MODEL_PATH = os.path.join(latest_dir, 'iforest_bulletproof_model.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- QASSAS LIVE INFERENCE (ISOLATION FOREST EDITION) ---")
print(f"Locked Threshold: {OPTIMAL_THRESHOLD}")

# 2. تحميل النماذج
print("Loading Stable Diffusion Pipeline...")
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True, safety_checker=None, requires_safety_checker=False
    ).to(device)
except:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(device)

try: pipe.enable_xformers_memory_efficient_attention()
except: pass

def dummy_checker(image, device, dtype): return image, [False] * len(image)
pipe.run_safety_checker = dummy_checker

print("Applying OFT Weights...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, latest_dir)
pipe.unet.to(device, dtype=torch.float16)

metrics_gen = MetricsFactory(device=device)

# تحميل نموذج غابة العزل
if_model = joblib.load(IFOREST_MODEL_PATH)
print("Isolation Forest Model Loaded Successfully.")

# 3. إعداد مجلد الاختبار
test_dir = "data/test"
if os.path.exists(os.path.join(test_dir, CATEGORY)):
    test_dir = os.path.join(test_dir, CATEGORY)
test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{CATEGORY}_")]

output_maps_dir = "anomaly_maps_output"
os.makedirs(output_maps_dir, exist_ok=True)

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
            strength=TARGET_STRENGTH,
            guidance_scale=TARGET_GUIDANCE,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

    scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
    features_array = np.array([[scores['L1'], scores['L2'], scores['MS_SSIM'], scores['LPIPS'], scores['Max_Patch']]])
    
    # حساب درجة الشذوذ (مع عكس الإشارة كما فعلنا في التدريب)
    anomaly_score = -if_model.decision_function(features_array)[0]
    
    # اتخاذ القرار بناء على العتبة المثالية
    predicted_label = 1 if anomaly_score >= OPTIMAL_THRESHOLD else 0
    
    # إذا تم اكتشاف عيب، قم برسم الخريطة
    if predicted_label == 1:
        map_path = os.path.join(output_maps_dir, f"map_{img_name}")
        generate_anomaly_map(original_image, reconstructed_image, save_path=map_path)
        
    if predicted_label == true_label:
        correct_predictions += 1
    else:
        true_status = "Good" if true_label == 0 else "Anomaly"
        pred_status = "Anomaly" if predicted_label == 1 else "Good"
        print(f"\nMismatch on {img_name}: True: {true_status} | Predicted: {pred_status} (Score: {anomaly_score:.4f})")
        
    clean_vram()

print("\n========================================")
print("--- LIVE INFERENCE REPORT ---")
print(f"Total Images: {total_images}")
print(f"Live Accuracy: {(correct_predictions/total_images)*100:.2f}%")
print(f"Anomaly maps saved to: {output_maps_dir}")
print("========================================")