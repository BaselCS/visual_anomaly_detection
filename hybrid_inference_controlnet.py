import os
import torch
import json
from PIL import Image
import numpy as np
import glob
import re
from diffusers import StableDiffusionImg2ImgPipeline
import xgboost as xgb
from tqdm import tqdm
import warnings

from metrics_factory import MetricsFactory

warnings.filterwarnings('ignore')

CATEGORY = "bottle"
PLACEHOLDER = "<perfect-bottle>"

def get_latest_ti_dir(base_dir="trained_models"):
    """البحث عن أحدث مجلد خاص بتدريب الانعكاس النصي فقط"""
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_ti_"):
            try:
                num = int(folder_name.replace("train_ti_", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if not latest_folder: raise FileNotFoundError("No Textual Inversion training found.")
    return os.path.join(base_dir, latest_folder)

latest_dir = get_latest_ti_dir()

# ==========================================
# اكتشاف المتغيرات تلقائيا
# ==========================================
metadata_search_pattern = os.path.join(latest_dir, "xgboost_metadata_*.json")
metadata_files = glob.glob(metadata_search_pattern)

if not metadata_files:
    raise FileNotFoundError(f"Could not find any metadata file in {latest_dir}. Did you run train_hybrid_xgboost.py?")

METADATA_PATH = max(metadata_files, key=os.path.getctime)

match = re.search(r"S([\d\.]+)_G([\d\.]+)\.json", os.path.basename(METADATA_PATH))

if match:
    STRENGTH = float(match.group(1))
    GUIDANCE = float(match.group(2))
else:
    raise ValueError("Could not extract Strength and Guidance from metadata filename.")

XGB_MODEL_PATH = METADATA_PATH.replace("xgboost_metadata_", "xgboost_hybrid_")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- QASSAS HYBRID INFERENCE SYSTEM (Textual Inversion Edition) ---")
print(f"Auto-Detected Configurations - Strength: {STRENGTH}, Guidance: {GUIDANCE}")

# ==========================================
# 1. تحميل النماذج والعتبة التلقائية
# ==========================================
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        OPTIMAL_THRESHOLD = metadata['optimal_threshold']
    print(f"Auto-Loaded Optimal Threshold: {OPTIMAL_THRESHOLD:.4f}")
except Exception as e:
    raise FileNotFoundError(f"Could not load threshold. Error: {e}")

print("Loading Base Pipeline...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

print(f"Loading Textual Inversion Token from {latest_dir}...")
pipe.load_textual_inversion(latest_dir, weight_name="learned_embeds.bin")

metrics_gen = MetricsFactory(device=device)

try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    print(f"Loading XGBoost Classifier from: {XGB_MODEL_PATH}")
except Exception as e:
    raise FileNotFoundError(f"Could not load XGBoost model. Error: {e}")

# ==========================================
# 2. الفحص المباشر 
# ==========================================
test_dir = "data/test"
if os.path.exists(os.path.join(test_dir, CATEGORY)):
    test_dir = os.path.join(test_dir, CATEGORY)
    test_files = os.listdir(test_dir)
else:
    test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{CATEGORY}")]

print(f"\nStarting live inspection with Textual Inversion on {len(test_files)} images...")

correct_predictions = 0
total_images = len(test_files)

for img_name in tqdm(test_files, desc="Inspecting"):
    img_path = os.path.join(test_dir, img_name)
    true_label = 0 if "good" in img_name.lower() else 1
    
    original_image = Image.open(img_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(999)

    with torch.no_grad():
        reconstructed_image = pipe(
            prompt=f"a high quality photo of a {PLACEHOLDER}",
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
    
    if predicted_label == true_label:
        correct_predictions += 1
    else:
        true_status = "Good" if true_label == 0 else "Anomaly"
        pred_status = "Anomaly" if predicted_label == 1 else "Good"
        print(f"\nMismatch on {img_name}: True: {true_status} | Predicted: {pred_status} (Prob: {anomaly_probability:.4f})")

print("\n" + "="*40)
print("--- TEXTUAL INVERSION LIVE INFERENCE REPORT ---")
print(f"Total Images: {total_images}")
print(f"Live Accuracy: {(correct_predictions/total_images)*100:.2f}%")
print("========================================")