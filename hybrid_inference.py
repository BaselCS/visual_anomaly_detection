import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
import xgboost as xgb
import numpy as np
from tqdm import tqdm
import warnings

from metrics_factory import MetricsFactory

warnings.filterwarnings('ignore')

# ==========================================
# الإعدادات (Configurations)
# ==========================================
# استخدم العتبة الصارمة التي ظهرت لك في نتائج التدريب
OPTIMAL_THRESHOLD = 0.8058 
STRENGTH = 0.42
GUIDANCE = 5.5
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
    return os.path.join(base_dir, latest_folder)

latest_dir = get_latest_train_folder()
SD_MODEL_PATH = os.path.join(latest_dir, "best_model")
XGB_MODEL_PATH = os.path.join(latest_dir, f"xgboost_hybrid_S{STRENGTH}_G{GUIDANCE}.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- 🏭 QASSAS HYBRID INFERENCE SYSTEM ---")
print(f"Loading Stable Diffusion (DoRA) from: {SD_MODEL_PATH}")
print(f"Loading XGBoost Classifier from: {XGB_MODEL_PATH}")

# 1. تحميل النماذج
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

pipe.unet = PeftModel.from_pretrained(pipe.unet, SD_MODEL_PATH)
metrics_gen = MetricsFactory(device=device)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

# 2. تحديد مسار الصور وتصفيتها (فقط التي تبدأ بـ bottle)
test_dir = "data/test"
if os.path.exists(os.path.join(test_dir, CATEGORY)):
    test_dir = os.path.join(test_dir, CATEGORY)
    test_files = os.listdir(test_dir)
else:
    test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{CATEGORY}")]

if not test_files:
    raise FileNotFoundError(f"No test images found for {CATEGORY} in {test_dir}")

print(f"\nStarting live inspection on {len(test_files)} images...")

correct_predictions = 0
total_images = len(test_files)

# 3. فحص الصور واحدة تلو الأخرى
for img_name in tqdm(test_files, desc="Inspecting"):
    img_path = os.path.join(test_dir, img_name)
    
    # تحديد الحالة الحقيقية للصورة من اسمها (للتحقق من الدقة فقط)
    true_label = 0 if "good" in img_name.lower() else 1
    
    # تحميل الصورة
    original_image = Image.open(img_path).convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(999)

    # إعادة البناء التوليدي
    with torch.no_grad():
        reconstructed_image = pipe(
            prompt=f"a high quality photo of a perfect {CATEGORY}",
            image=original_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

    # استخراج الخصائص
    scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
    features_array = np.array([[scores['L1'], scores['L2'], scores['MS_SSIM'], scores['LPIPS'], scores['Max_Patch']]])

    # تصنيف XGBoost
    anomaly_probability = xgb_model.predict_proba(features_array)[0, 1]
    
    # اتخاذ القرار بناءً على العتبة
    predicted_label = 1 if anomaly_probability >= OPTIMAL_THRESHOLD else 0
    
    if predicted_label == true_label:
        correct_predictions += 1
        
    # طباعة نتائج العينات التي أخطأ فيها النظام لفهم طبيعة الخطأ
    if predicted_label != true_label:
        true_status = "Good" if true_label == 0 else "Anomaly"
        pred_status = "Anomaly" if predicted_label == 1 else "Good"
        print(f"\n⚠️ Mismatch on {img_name}:")
        print(f"   True: {true_status} | Predicted: {pred_status} (Confidence: {anomaly_probability:.4f})")

# 4. التقرير النهائي
print("\n" + "="*40)
print("--- 📊 LIVE INFERENCE REPORT ---")
print(f"Total Images Inspected: {total_images}")
print(f"Correct Predictions:    {correct_predictions}")
print(f"Live Accuracy:          {(correct_predictions/total_images)*100:.2f}%")
print("========================================")