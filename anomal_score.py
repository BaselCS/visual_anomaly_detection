import os
import torch
import gc
import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from peft import PeftModel
from tqdm import tqdm
import warnings
import random

from metrics_factory import MetricsFactory
from results_logger import ResultsLogger

warnings.filterwarnings('ignore')

STRENGTH_OPTIONS = [0.35, 0.38, 0.40, 0.42]
GUIDANCE_OPTIONS = [5.5, 6.0, 6.5, 7.0]

DEFAULT_EVAL_CONFIG = {
    "seed": 999,
    "categories": ["bottle"], 
    "reconstruction_steps": 30,
    "controlnet_conditioning_scale": 0.8
}

device = "cuda" if torch.cuda.is_available() else "cpu"
metrics_gen = MetricsFactory(device=device)

def get_latest_train_model(base_dir="trained_models"):
    if not os.path.exists(base_dir): raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    max_num = 0
    latest_folder = None
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train") and os.path.isdir(os.path.join(base_dir, folder_name)):
            try:
                num = int(folder_name.replace("train", ""))
                if num > max_num: 
                    max_num = num
                    latest_folder = folder_name  # 👈 تم استرجاع السطر المفقود هنا
            except ValueError:
                continue
    if latest_folder is None: raise FileNotFoundError("No valid training found.")
    best_model_path = os.path.join(base_dir, latest_folder, "best_model")
    final_model_path = os.path.join(base_dir, latest_folder, "final_model")
    if os.path.exists(best_model_path): return best_model_path
    elif os.path.exists(final_model_path): return final_model_path
    else: raise FileNotFoundError("No models found in the latest directory.")

    
CURRENT_TECHNIQUE = "DoRA_ControlNet" 
BEST_TRAIN_MODEL = get_latest_train_model("trained_models")
print(f"Auto-detected latest trained model: {BEST_TRAIN_MODEL}")

TRAIN_FOLDER_ROOT = os.path.dirname(BEST_TRAIN_MODEL)
csv_path = os.path.join(TRAIN_FOLDER_ROOT, 'results_database.csv')
logger = ResultsLogger(filepath=csv_path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def get_canny_image(image):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_cv, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

def extract_features_for_category(pipe, category, strength, guidance):
    test_dir = f"data/test/{category}" 
    if not os.path.exists(test_dir):
        test_dir = "data/test"
        all_test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{category}_")]
    else:
        all_test_files = os.listdir(test_dir)
        
    if not all_test_files: return

    for img_name in tqdm(all_test_files, desc=f"Processing Images (S={strength}, G={guidance})"):
        img_path = os.path.join(test_dir, img_name)
        label = 0 if "good" in img_name.lower() else 1
        
        original_image = Image.open(img_path).convert("RGB").resize((512, 512))
        control_image = get_canny_image(original_image)
        generator = torch.Generator(device=device).manual_seed(DEFAULT_EVAL_CONFIG["seed"])

        with torch.no_grad():
            reconstructed_image = pipe(
                prompt=f"a high quality photo of a perfect {category}",
                image=original_image,
                control_image=control_image,
                strength=strength,
                guidance_scale=guidance,
                controlnet_conditioning_scale=DEFAULT_EVAL_CONFIG["controlnet_conditioning_scale"],
                num_inference_steps=DEFAULT_EVAL_CONFIG["reconstruction_steps"],
                generator=generator,
            ).images[0]

        scores = metrics_gen.calculate_metrics(original_image, reconstructed_image)
        scores.update({
            'Category': category,
            'Technique_Used': CURRENT_TECHNIQUE,
            'Strength': strength,
            'Guidance': guidance,
            'Label': label 
        })
        logger.log_result(scores)
        
    clean_vram()

if __name__ == "__main__":
    set_seed(DEFAULT_EVAL_CONFIG["seed"])
    
    print(f"Loading Base Model, ControlNet & {CURRENT_TECHNIQUE} Weights...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    ).to(device)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.unet = PeftModel.from_pretrained(pipe.unet, BEST_TRAIN_MODEL)
    
    for cat in DEFAULT_EVAL_CONFIG["categories"]:
        print(f"\n--- Feature Extraction: {cat.upper()} ---")
        for s in STRENGTH_OPTIONS:
            for g in GUIDANCE_OPTIONS:
                extract_features_for_category(pipe, cat, s, g)

    print("\nFEATURE EXTRACTION COMPLETE! Data is ready for XGBoost.")