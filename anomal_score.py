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
PLACEHOLDER = "<perfect-bottle>"

def clean_vram():
    """تنظيف ذاكرة كرت الشاشة لتجنب التوقف المفاجئ (Out of Memory)"""
    gc.collect()
    torch.cuda.empty_cache()

def get_latest_ti_dir(base_dir="trained_models"):
    """البحث عن أحدث مجلد خاص بتدريب الانعكاس النصي"""
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

# ==========================================
# 1. إعداد المسارات
# ==========================================
ti_dir = get_latest_ti_dir()
embed_file_path = os.path.join(ti_dir, "learned_embeds.bin")
csv_path = os.path.join(ti_dir, 'results_database.csv')
logger = ResultsLogger(filepath=csv_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_gen = MetricsFactory(device=device)

# ==========================================
# 2. تحميل النماذج بآلية آمنة
# ==========================================
print("Loading Base Pipeline (Offline Mode)...")
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
except Exception as e:
    print(f"Offline load failed, trying online... error: {e}")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

print(f"Loading Textual Inversion Embeddings from {embed_file_path}...")
pipe.load_textual_inversion(ti_dir, weight_name="learned_embeds.bin")

# ==========================================
# 3. إعداد البيانات وبدء الاستخراج
# ==========================================
test_dir = "data/test/bottle" if os.path.exists("data/test/bottle") else "data/test"
test_files = [f for f in os.listdir(test_dir) if f.startswith("bottle_")]

print(f"Extracting features using Textual Inversion token: {PLACEHOLDER}")

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
                        prompt=f"a high quality photo of a {PLACEHOLDER}",
                        image=original_image,
                        strength=s,
                        guidance_scale=g,
                        num_inference_steps=30,
                        generator=generator,
                    ).images[0]

                scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
                scores.update({
                    'Category': 'bottle',
                    'Technique_Used': 'Textual_Inversion',
                    'Strength': s,
                    'Guidance': g,
                    'Label': label 
                })
                logger.log_result(scores)
                
            except Exception as e:
                print(f"\nError processing {img_name}: {e}. Skipping to next image.")
                continue
                
        clean_vram()

print("Data extraction complete. You can now run train_hybrid_xgboost.py")