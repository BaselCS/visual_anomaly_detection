import os
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from tqdm import tqdm
import warnings
import random
import numpy as np

from metrics_factory import MetricsFactory
from results_logger import ResultsLogger

warnings.filterwarnings('ignore')

STRENGTH_OPTIONS = [0.35,0.38, 0.40, 0.42]
GUIDANCE_OPTIONS = [5.5,6.0,6.5, 7.0, 7.5]

DEFAULT_EVAL_CONFIG = {
    "seed": 999,
    "categories": ["bottle"], # نبدأ بالزجاجة كما خططنا
    "reconstruction_steps": 30,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_gen = MetricsFactory(device=device)
import os

def get_latest_train_model(base_dir="trained_models"):
    """Find the latest trainX directory and return 'best_model', fallback to 'final_model'"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist. Please run main.py first.")
    
    max_num = 0
    latest_folder = None
    
    # 1. البحث عن أعلى مجلد trainX موجود
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
        
    # 2. التحقق من وجود best_model، وإذا لم يوجد نأخذ final_model
    best_model_path = os.path.join(base_dir, latest_folder, "best_model")
    final_model_path = os.path.join(base_dir, latest_folder, "final_model")
    
    if os.path.exists(best_model_path):
        return best_model_path
    elif os.path.exists(final_model_path):
        print(f"⚠ 'best_model' not found in {latest_folder}. Using 'final_model' instead.")
        return final_model_path
    else:
        raise FileNotFoundError(f"Neither 'best_model' nor 'final_model' found in {latest_folder}")


CURRENT_TECHNIQUE = "DoRA" 
BEST_TRAIN_MODEL = get_latest_train_model("trained_models")
print(f"✅ Auto-detected latest trained model: {BEST_TRAIN_MODEL}")

TRAIN_FOLDER_ROOT = os.path.dirname(BEST_TRAIN_MODEL)
csv_path = os.path.join(TRAIN_FOLDER_ROOT, 'results_database.csv')

logger = ResultsLogger(filepath=csv_path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def extract_features_for_category(pipe, category, strength, guidance):
    """
    هذه الدالة تستخرج الخصائص فقط وترسلها للـ Logger.
    لا يوجد اتخاذ قرار أو حساب AUC هنا.
    """
    test_dir = f"data/test/{category}" # تأكد من مسار مجلد الاختبار لديك
    if not os.path.exists(test_dir):
        # البحث في الملفات المباشرة إذا لم تكن مقسمة بمجلدات
        test_dir = "data/test"
        all_test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{category}_")]
    else:
        all_test_files = os.listdir(test_dir)
        
    if not all_test_files: 
        print(f"No test files found for {category}")
        return

    print(f"Extracting features for {category} | Strength: {strength} | Guidance: {guidance}")
    
    for img_name in tqdm(all_test_files, desc="Processing Images"):
        img_path = os.path.join(test_dir, img_name)
        
        # استخراج نوع العيب (سليم أم معيب)
        # افترضنا أن التسمية تحتوي على كلمة good للصور السليمة
        label = 0 if "good" in img_name.lower() else 1
        
        # قراءة الصورة وتوحيد حجمها فوراً لتطابق دقة التدريب وتتجنب تعارض الأحجام
        original_image = Image.open(img_path).convert("RGB").resize((512, 512))
        generator = torch.Generator(device=device).manual_seed(DEFAULT_EVAL_CONFIG["seed"])

        with torch.no_grad():
            reconstructed_image = pipe(
                prompt=f"a high quality photo of a perfect {category}",
                image=original_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=DEFAULT_EVAL_CONFIG["reconstruction_steps"],
                generator=generator,
            ).images[0]

        # حساب المقاييس باستخدام Factory الخاص بك
        scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
        
        # تجهيز السجل
        scores.update({
            'Category': category,
            'Technique_Used': CURRENT_TECHNIQUE,
            'Strength': strength,
            'Guidance': guidance,
            'Label': label 
        })
        
        # حفظ فوري في CSV
        logger.log_result(scores)
        
    clean_vram()

if __name__ == "__main__":
    set_seed(DEFAULT_EVAL_CONFIG["seed"])
    
    print(f"Loading Base Model & {CURRENT_TECHNIQUE} Weights...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    
    # تفعيل xformers لتسريع الاستنتاج
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.unet = PeftModel.from_pretrained(pipe.unet, BEST_TRAIN_MODEL)
    
    for cat in DEFAULT_EVAL_CONFIG["categories"]:
        print(f"\n--- 🛠 Feature Extraction: {cat.upper()} ---")
        for s in STRENGTH_OPTIONS:
            for g in GUIDANCE_OPTIONS:
                extract_features_for_category(pipe, cat, s, g)

    print("\n" + "="*40)
    print("FEATURE EXTRACTION COMPLETE! Data is ready for XGBoost.")