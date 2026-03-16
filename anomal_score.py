import os
import torch
import csv
import gc
from torchvision import transforms
import numpy as np
import random
import hashlib
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
import lpips  
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
from pytorch_msssim import ms_ssim

warnings.filterwarnings('ignore')

BEST_TRAIN_MODEL = "trained_models/train9/best_model"
STRENGTH_OPTIONS = [0.29,0.32,0.35,0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]
GUIDANCE_OPTIONS = [5.0,5.5,6.0,6.5, 7.0, 7.5, 8.0,8.5, 9.0, 9.5, 10.0]
CSV_FILE = "all_categories_grid_search.csv"

DEFAULT_EVAL_CONFIG = {
    "seed": 999,
    "categories": ["capsule","toothbrush"], 
    "reconstruction_steps": 30,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)

def clean_vram():
    """تنظيف ذاكرة الكرت والذاكرة العشوائية"""
    gc.collect()
    torch.cuda.empty_cache()

def calculate_anomaly_score(original_image, reconstructed_image):
    resize_op = transforms.Resize((512, 512))
    to_tensor = transforms.ToTensor()
    org_t = to_tensor(resize_op(original_image)).to(device)
    rec_t = to_tensor(resize_op(reconstructed_image)).to(device)
    diff_t = torch.abs(org_t - rec_t)
    
    l1_score = float(diff_t.mean().item())
    local_max = torch.nn.functional.max_pool2d(diff_t.unsqueeze(0), kernel_size=16, stride=8)
    max_patch_score = float(local_max.max().item())

    org_batch, rec_batch = org_t.unsqueeze(0), rec_t.unsqueeze(0)
    msssim_dist = 1.0 - float(ms_ssim(org_batch, rec_batch, data_range=1.0).item())
    
    org_lpips, rec_lpips = (org_batch * 2) - 1, (rec_batch * 2) - 1
    with torch.no_grad():
        lpips_dist = float(loss_fn_vgg(org_lpips, rec_lpips).item())

    combined_score = (0.15 * l1_score + 0.25 * msssim_dist + 0.30 * lpips_dist + 0.30 * max_patch_score)
    return combined_score

def get_completed_runs(filename):
    """قراءة التجارب المكتملة لتجنب تكرارها"""
    completed = set()
    if os.path.exists(filename):
        with open(filename, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # تخزين مفتاح فريد: (الفئة، القوة، التوجيه)
                completed.add((row['Category'], float(row['Strength']), float(row['Guidance'])))
    return completed

def run_evaluation_for_category(pipe, category, strength, guidance):
    all_records = []
    test_dir = "data/test"
    all_test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{category}_")]
    
    if not all_test_files: return None

    for img_name in all_test_files:
        img_path = os.path.join(test_dir, img_name)
        defect_type = img_name.split('_')[1]
        label = 0 if defect_type == "good" else 1
        
        original_image = Image.open(img_path).convert("RGB")
        image_seed = DEFAULT_EVAL_CONFIG["seed"] + (stable_int(img_name) % 100000)
        generator = torch.Generator(device=device).manual_seed(image_seed)

        with torch.no_grad(): # توفير الذاكرة أثناء الاستنتاج
            reconstructed_image = pipe(
                prompt=f"a high quality photo of a perfect {category}",
                image=original_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=DEFAULT_EVAL_CONFIG["reconstruction_steps"],
                generator=generator,
            ).images[0]

        score = calculate_anomaly_score(original_image, reconstructed_image)
        all_records.append({"label": label, "score": score})
        
        # تنظيف بعد كل صورة لضمان استقرار الـ VRAM
        clean_vram()

    labels = [r["label"] for r in all_records]
    scores = [r["score"] for r in all_records]
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    
    preds = [1 if s > threshold else 0 for s in scores]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(labels)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"AUC": auc, "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy, "Threshold": threshold}

if __name__ == "__main__":
    set_seed(DEFAULT_EVAL_CONFIG["seed"])
    
    print("Loading Base Model & Multi-Concept LoRA (Train 9)...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, BEST_TRAIN_MODEL)
    
    # 1. فحص ما تم إنجازه سابقاً
    completed_runs = get_completed_runs(CSV_FILE)
    file_exists = os.path.exists(CSV_FILE)

    # 2. فتح الملف في وضع الإضافة (Append)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Category", "Strength", "Guidance", "AUC", "Precision", "Recall", "F1", "Accuracy", "Threshold"])

        for cat in DEFAULT_EVAL_CONFIG["categories"]:
            print(f"\n--- 🛠 Grid Search: {cat.upper()} ---")
            for s in STRENGTH_OPTIONS:
                for g in GUIDANCE_OPTIONS:
                    # 3. التحقق من الاستكمال
                    if (cat, s, g) in completed_runs:
                        print(f"Skipping {cat} (S={s}, G={g}) - Already done.")
                        continue
                    
                    print(f"Testing {cat}: S={s}, G={g}")
                    res = run_evaluation_for_category(pipe, cat, s, g)
                    
                    if res:
                        writer.writerow([cat, s, g, f"{res['AUC']:.4f}", f"{res['Precision']:.4f}", 
                                         f"{res['Recall']:.4f}", f"{res['F1']:.4f}", f"{res['Accuracy']:.4f}", f"{res['Threshold']:.4f}"])
                        f.flush() # حفظ فوري

                    # 4. تنظيف شامل بعد كل توليفة
                    clean_vram()

    print("\n" + "="*40)
    print("ALL CATEGORIES GRID SEARCH COMPLETE!")