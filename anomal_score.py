import os
import torch
import gc
import random
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from tqdm import tqdm
import warnings

from metrics_factory import MetricsFactory

warnings.filterwarnings('ignore')

TARGET_STRENGTH = 0.40
TARGET_GUIDANCE = 6.5
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
                if num > max_num: max_num = num; latest_folder = folder_name
            except ValueError: continue
    if not latest_folder: raise FileNotFoundError("No OFT training directory found.")
    return os.path.join(base_dir, latest_folder)

oft_dir = get_latest_oft_dir()
# 🔥 إنشاء ملف جديد تماماً لتجنب أي تعارض مع ResultsLogger
csv_path = os.path.join(oft_dir, 'pure_results_database.csv') 

device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_gen = MetricsFactory(device=device)

print("Loading Base Pipeline (Safety Checker DISABLED)...")
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

print(f"Applying OFT Weights from: {oft_dir}")
pipe.unet = PeftModel.from_pretrained(pipe.unet, oft_dir)
pipe.unet.to(device, dtype=torch.float16)

train_dir = "data/train/bottle" if os.path.exists("data/train/bottle") else "data/train"
test_dir = "data/test/bottle" if os.path.exists("data/test/bottle") else "data/test"

train_files = [f for f in os.listdir(train_dir) if f.endswith(".png") or f.endswith(".jpg")]
test_files = [f for f in os.listdir(test_dir) if f.startswith("bottle_")]

train_files = random.sample(train_files, min(40, len(train_files)))

all_tasks = [("Train", f, train_dir) for f in train_files] + [("Test", f, test_dir) for f in test_files]

print(f"\nExtracting features strictly: {len(train_files)} Train images, {len(test_files)} Test images...")

all_results = [] # قائمة لحفظ النتائج مباشرة

for split, img_name, folder_path in tqdm(all_tasks, desc="Processing Images"):
    img_path = os.path.join(folder_path, img_name)
    label = 0 if "good" in img_name.lower() or split == "Train" else 1
    
    try:
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
        scores.update({
            'Image_Name': img_name,
            'Category': 'bottle',
            'Technique_Used': 'OFT_UNet',
            'Strength': TARGET_STRENGTH,
            'Guidance': TARGET_GUIDANCE,
            'Label': label,
            'Split': split # ستحفظ الآن رغماً عنها
        })
        all_results.append(scores)
        
    except Exception as e:
        continue
        
    clean_vram()

# حفظ الملف مباشرة باستخدام Pandas
df_final = pd.DataFrame(all_results)
df_final.to_csv(csv_path, index=False)
print(f"\n✅ Pure Data extraction complete. Saved securely to {csv_path}")