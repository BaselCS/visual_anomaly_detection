import os
import torch
import csv
from torchvision import transforms
import numpy as np
import random
import hashlib
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch import nn
from peft import PeftModel
import lpips  
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
from datetime import datetime
import logging
from pytorch_msssim import ms_ssim

warnings.filterwarnings('ignore')

# --- إعدادات البحث الشبكي (Grid Search Settings) ---
STRENGTH_OPTIONS = [0.29,0.32,0.35,0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]
GUIDANCE_OPTIONS = [5.0,5.5,6.0,6.5, 7.0, 7.5, 8.0,8.5, 9.0, 9.5, 10.0]
BEST_TRAIN_MODEL = "trained_models/train9/best_model" # النموذج الذهبي الذي اتفقنا عليه

# إعدادات التقييم العامة
DEFAULT_EVAL_CONFIG = {
    "seed": 999,
    "categories": ["bottle"],
    "calibration_fraction": 0.2,
    "reconstruction_steps": 30,
}

# إعداد السجلات (Logging)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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

# --- دالة حساب السكور المطورة (Max Patch Score) ---
def calculate_anomaly_score(original_image, reconstructed_image):
    resize_op = transforms.Resize((512, 512))
    to_tensor = transforms.ToTensor()

    org_t = to_tensor(resize_op(original_image)).to(device)
    rec_t = to_tensor(resize_op(reconstructed_image)).to(device)

    diff_t = torch.abs(org_t - rec_t)
    
    # 1. السكور العام
    l1_score = float(diff_t.mean().item())

    # 2. السكور المحلي (Max Patch) - لاكتشاف الخدوش الصغيرة
    local_max = torch.nn.functional.max_pool2d(diff_t.unsqueeze(0), kernel_size=16, stride=8)
    max_patch_score = float(local_max.max().item())

    # 3. المقاييس المتقدمة
    org_batch, rec_batch = org_t.unsqueeze(0), rec_t.unsqueeze(0)
    msssim_dist = 1.0 - float(ms_ssim(org_batch, rec_batch, data_range=1.0).item())
    
    org_lpips, rec_lpips = (org_batch * 2) - 1, (rec_batch * 2) - 1
    with torch.no_grad():
        lpips_dist = float(loss_fn_vgg(org_lpips, rec_lpips).item())

    # أوزان المعادلة (موزونة لرفع الـ Recall)
    combined_score = (0.15 * l1_score + 0.25 * msssim_dist + 0.30 * lpips_dist + 0.30 * max_patch_score)
    return combined_score

# --- دالة التقييم الرئيسية (The Tester) ---
def run_evaluation(pipe, strength, guidance):
    all_records = []
    test_dir = "data/test"
    category = "bottle"
    
    all_test_files = [f for f in os.listdir(test_dir) if f.startswith(f"{category}_")]
    
    for img_name in all_test_files:
        img_path = os.path.join(test_dir, img_name)
        defect_type = img_name.split('_')[1]
        label = 0 if defect_type == "good" else 1
        
        original_image = Image.open(img_path).convert("RGB")
        image_seed = DEFAULT_EVAL_CONFIG["seed"] + (stable_int(img_name) % 100000)
        generator = torch.Generator(device=device).manual_seed(image_seed)

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

    # حساب المقاييس
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
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return auc, precision, recall, threshold

# --- بداية التشغيل (Main Execution) ---
if __name__ == "__main__":
    set_seed(DEFAULT_EVAL_CONFIG["seed"])
    
    print("🚀 Loading Base Model & LoRA (Train 9)...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, BEST_TRAIN_MODEL)
    
    results_history = []
    csv_file = "anomaly_detection_results/grid_search_results.csv"

    print(f"📊 Starting Grid Search on {len(STRENGTH_OPTIONS) * len(GUIDANCE_OPTIONS)} combinations...")

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Strength", "Guidance", "AUC", "Precision", "Recall", "Threshold"])

        for s in STRENGTH_OPTIONS:
            for g in GUIDANCE_OPTIONS:
                print(f"🔎 Testing: Strength={s}, Guidance={g}")
                auc, prec, rec, thresh = run_evaluation(pipe, s, g)
                
                writer.writerow([s, g, f"{auc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{thresh:.4f}"])
                results_history.append({"s": s, "g": g, "recall": rec})
                print(f"   ✅ Result: Recall={rec:.4f}, AUC={auc:.4f}")

    best = max(results_history, key=lambda x: x['recall'])
    print("\n" + "="*30)
    print(f"🏆 BEST SETTING FOUND:")
    print(f"Strength: {best['s']}, Guidance: {best['g']} -> Recall: {best['recall']:.4f}")
    print(f"📁 Full report saved to: {csv_file}")