import os
import math
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, ControlNetModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

def get_next_train_dir(base_dir="trained_models"):
    os.makedirs(base_dir, exist_ok=True)
    max_num = 0
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train") and os.path.isdir(os.path.join(base_dir, folder_name)):
            try:
                num = int(folder_name.replace("train", ""))
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    next_num = max_num + 1
    return os.path.join(base_dir, f"train{next_num}")

DEFAULT_CONFIG = {
    "categories": ["bottle"],
    "use_data_augmentation": False, # تم الإيقاف لضمان تطابق الصورة مع الحواف تماما
    "epochs": 150,
    "batch_size": 2,                  
    "gradient_accumulation_steps": 4, 
    "learning_rate": 5e-5,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,             
    "lr_scheduler": "cosine",         
    "lr_warmup_steps": 100,           
    "lr_eta_min": 1e-6,
    "save_every_n_epochs": 5,
    "save_log_every_n_epochs": 5,
    "output_dir": get_next_train_dir("trained_models"),
    "image_size": 512,
    "train_split": 0.9,               
    "seed": 999,
    "early_stop_patience": 20,     
    "early_stop_min_delta": 1e-4,
    "min_epochs_before_early_stop": 30, 
    "num_workers": 4              
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('training_controlnet.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def out(message: str, level: str = "info") -> None:
    if level == "error": logger.error(message)
    elif level == "warning": logger.warning(message)
    else: logger.info(message)
    print(message)

def load_config() -> dict: return dict(DEFAULT_CONFIG)
config = load_config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(config["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

out("Training Configuration (ControlNet Edition):")
for key, value in config.items(): out(f"  {key}: {value}")
os.makedirs(config["output_dir"], exist_ok=True)

out("\nLoading ControlNet Model...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
controlnet.requires_grad_(False)

out("Loading Stable Diffusion model (Offline Mode)...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        safety_checker=None, 
        requires_safety_checker=False,
        local_files_only=True  # 👈 هذا هو السطر الذي يمنع التعليق ويقرأ من جهازك فوراً
    ).to(device)
except Exception as e:
    out(f"⚠️ Offline load failed, trying online... error: {e}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        safety_checker=None, 
        requires_safety_checker=False
    ).to(device)
    
vae = pipe.vae                           
text_encoder = pipe.text_encoder         
tokenizer = pipe.tokenizer               
unet = pipe.unet                         

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    pass

unet.enable_gradient_checkpointing()
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

out("\nApplying LoRA configuration to U-Net...")
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.05, bias="none", use_dora=True
)
unet = get_peft_model(unet, lora_config)

def get_canny_image(image):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_cv, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

class AnomalyControlNetDataset(Dataset):
    def __init__(self, train_dir="data/train", categories=None, image_size=512):
        self.image_size = image_size
        self.samples = []
        all_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
        
        for img_file in all_files:
            parts = img_file.rsplit('_', 1)
            if len(parts) == 2 and parts[1].replace('.png', '').isdigit():
                category = parts[0]
                if categories is None or category in categories:
                    img_path = os.path.join(train_dir, img_file)
                    prompt = f"a high quality photo of a perfect {category}"
                    self.samples.append((img_path, prompt, category))
                    
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt, category = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            img_resized = image.resize((self.image_size, self.image_size))
            canny_image = get_canny_image(img_resized)
            
            pixel_values = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(img_resized)
            
            conditioning_pixel_values = transforms.ToTensor()(canny_image)
            
            return {
                "pixel_values": pixel_values,
                "conditioning_pixel_values": conditioning_pixel_values,
                "prompt": prompt,
                "category": category
            }
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))

out("\nPreparing dataset...")
full_dataset = AnomalyControlNetDataset(train_dir="data/train", categories=config["categories"], image_size=config["image_size"])

rng = random.Random(config["seed"])
train_indices, val_indices = [], []
for idx in range(len(full_dataset)): train_indices.append(idx)
split_at = int(len(train_indices) * config["train_split"])
rng.shuffle(train_indices)
val_indices = train_indices[split_at:]
train_indices = train_indices[:split_at]

if len(val_indices) == 0 and len(train_indices) > 1: val_indices.append(train_indices.pop())

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)

def validate(unet, controlnet, val_dataloader, vae, text_encoder, tokenizer, pipe, device):
    unet.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            try:
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(device, dtype=torch.float16)
                
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215                
                noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=validation_noise_generator)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device, generator=validation_noise_generator).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                inputs = tokenizer(batch["prompt"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=conditioning_pixel_values, return_dict=False,
                )
                
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=torch.float16) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16),
                ).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_losses.append(loss.item())
            except Exception as e:
                continue
    unet.train()
    return sum(val_losses) / len(val_losses) if val_losses else float('inf')

optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

from torch.optim.lr_scheduler import LambdaLR
updates_per_epoch = max(1, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]))
total_updates = updates_per_epoch * config["epochs"]
warmup_updates = min(config["lr_warmup_steps"], max(1, total_updates // 5))

def lr_lambda(current_update):
    if current_update < warmup_updates: return float(current_update + 1) / float(max(1, warmup_updates))
    progress = float(current_update - warmup_updates) / float(max(1, total_updates - warmup_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(config["lr_eta_min"] / config["learning_rate"], cosine)

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

training_noise_generator = torch.Generator(device=device).manual_seed(config["seed"] + 1000)
validation_noise_generator = torch.Generator(device=device).manual_seed(config["seed"] + 2000)

unet.train()
training_log = {
    "start_time": datetime.now().isoformat(), "config": config, "epochs": [],
    "best_epoch": None, "best_train_loss": float('inf'), "best_val_loss": float('inf')
}

global_step = 0
epochs_without_improvement = 0
scaler = torch.amp.GradScaler('cuda')

for epoch in range(config["epochs"]):
    epoch_losses = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for step, batch in enumerate(progress_bar):
        try:
            pixel_values = batch["pixel_values"].to(device)
            conditioning_pixel_values = batch["conditioning_pixel_values"].to(device)
            
            with torch.amp.autocast('cuda'):
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=training_noise_generator)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device, generator=training_noise_generator).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                inputs = tokenizer(batch["prompt"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=conditioning_pixel_values, return_dict=False,
                )
                
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise) / config["gradient_accumulation_steps"]
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_losses.append(loss.item() * config["gradient_accumulation_steps"])
            progress_bar.set_postfix({"loss": f"{epoch_losses[-1]:.4f}"})
            
        except Exception as e:
            continue

    total_batches = len(epoch_losses)
    if total_batches > 0 and (total_batches % config["gradient_accumulation_steps"] != 0):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    current_lr = optimizer.param_groups[0]['lr']
    
    avg_val_loss = validate(unet, controlnet, val_dataloader, vae, text_encoder, tokenizer, pipe, device)
    
    if avg_val_loss < (training_log["best_val_loss"] - config["early_stop_min_delta"]):
        training_log["best_val_loss"] = avg_val_loss
        training_log["best_train_loss"] = avg_train_loss
        training_log["best_epoch"] = epoch + 1
        epochs_without_improvement = 0
        best_model_dir = os.path.join(config["output_dir"], "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        unet.save_pretrained(best_model_dir)
    else:
        epochs_without_improvement += 1
    
    if (epoch + 1) >= config["min_epochs_before_early_stop"] and epochs_without_improvement >= config["early_stop_patience"]:
        break
    
    training_log["epochs"].append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "learning_rate": current_lr})

final_model_dir = os.path.join(config["output_dir"], "final_model")
os.makedirs(final_model_dir, exist_ok=True)
unet.save_pretrained(final_model_dir)
out("Training completed and saved.")