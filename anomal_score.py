import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from tqdm import tqdm
import warnings

from metrics_factory import MetricsFactory
from results_logger import ResultsLogger

warnings.filterwarnings('ignore')

STRENGTH_OPTIONS = [0.35, 0.40, 0.45]
GUIDANCE_OPTIONS = [5.5, 6.5, 7.5]
PROMPT = "a high quality photo of a perfect bottle"

def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def get_latest_te_lora_dir(base_dir="trained_models"):
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_te_lora_"):
            try:
                num = int(folder_name.replace("train_te_lora_", ""))
                if num > max_num:
                    max_num = num
                    latest_folder = folder_name
            except ValueError:
                continue
    if not latest_folder: raise FileNotFoundError("No Text Encoder LoRA training found.")
    return os.path.join(base_dir, latest_folder)

# ==========================================
# 1. إعداد المسارات وتطهير البيانات القديمة
# ==========================================
te_lora_dir = get_latest_te_lora_dir()
csv_path = os.path.join(te_lora_dir, 'results_database.csv')

# 🔥 الحل 1: مسح قاعدة البيانات القديمة لتجنب تلوث الأرقام
if os.path.exists(csv_path):
    os.remove(csv_path)
    print("🧹 Old CSV database cleared successfully.")

logger = ResultsLogger(filepath=csv_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_gen = MetricsFactory(device=device)

# ==========================================
# 2. تحميل النماذج بشكل محمي
# ==========================================
print("Loading Base Pipeline (Safety Checker DISABLED)...")
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, 
        local_files_only=True,
        safety_checker=None, requires_safety_checker=False # 🔥 الحل 2
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

print(f"Applying Text Encoder LoRA natively from: {te_lora_dir}")
# 🔥 الحل 3: التحميل الأصلي (Native) لتجنب خلل PEFT
try:
    pipe.load_lora_weights(te_lora_dir, weight_name="adapter_model.safetensors")
except:
    pipe.load_lora_weights(te_lora_dir, weight_name="adapter_model.bin")

def dummy_checker(image, device, dtype):
    return image, [False] * len(image)

pipe.run_safety_checker = dummy_checker
# ==========================================
# 3. إعداد البيانات وبدء الاستخراج
# ==========================================
test_dir = "data/test/bottle" if os.path.exists("data/test/bottle") else "data/test"
test_files = [f for f in os.listdir(test_dir) if f.startswith("bottle_")]

print(f"Extracting features on {len(test_files)} images...")

for s in STRENGTH_OPTIONS:
    for g in GUIDANCE_OPTIONS:
        for img_name in tqdm(test_files, desc=f"Processing (S={s}, G={g})"):
            img_path = os.path.join(test_dir, img_name)
            label = 0 if "good" in img_name.lower() else 1
            
            try:
                original_image = Image.open(img_path).convert("RGB").resize((512, 512))
                generator = torch.Generator(device=device).manual_seed(999)

                with torch.no_grad():
                    reconstructed_image = pipe(
                        prompt=PROMPT,
                        image=original_image,
                        strength=s,
                        guidance_scale=g,
                        num_inference_steps=30,
                        generator=generator,
                    ).images[0]

                scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
                scores.update({
                    'Category': 'bottle',
                    'Technique_Used': 'Text_Encoder_LoRA',
                    'Strength': s,
                    'Guidance': g,
                    'Label': label 
                })
                logger.log_result(scores)
                
            except Exception as e:
                print(f"\nError processing {img_name}: {e}. Skipping.")
                continue
                
            # 🔥 الحل 4: تنظيف الذاكرة بعد كل صورة
            clean_vram()

print("Data extraction complete. You can now run train_hybrid_xgboost.py")