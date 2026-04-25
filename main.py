import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import OFTConfig, get_peft_model
from tqdm import tqdm

# تسريع عمليات الضرب للمصفوفات
torch.set_float32_matmul_precision('high')

def get_next_train_dir(base_dir="trained_models"):
    os.makedirs(base_dir, exist_ok=True)
    max_num = 0
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_oft_"):
            try:
                num = int(folder_name.replace("train_oft_", ""))
                if num > max_num: max_num = num
            except ValueError:
                continue
    return os.path.join(base_dir, f"train_oft_{max_num + 1}")

CONFIG = {
    "category": "bottle",
    "prompt": "a high quality photo of a perfect bottle",
    "epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4, 
    "oft_r": 8,  # عدد الكتل المتعامدة (Blocks)
    "output_dir": get_next_train_dir(),
    "seed": 999
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Diffusion pipeline (Safeguarded Mixed Precision)...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
except Exception:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

# ==========================================
# 1. تجميد النماذج بالكامل لتوفير الذاكرة
# ==========================================
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# ==========================================
# 2. إعداد الـ U-Net لتقنية OFT (الضبط المتعامد)
# ==========================================
unet.to(dtype=torch.float32)

oft_config = OFTConfig(
    r=CONFIG["oft_r"],
    oft_block_size=0, # 🔥 الحل البرمجي لخلل مكتبة PEFT
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    module_dropout=0.0,
    init_weights=True
)

# تركيب محولات OFT على U-Net
unet = get_peft_model(unet, oft_config)
print("--- OFT Architecture ---")
unet.print_trainable_parameters()

class OFTDataset(Dataset):
    def __init__(self, train_dir, category):
        self.image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{category}_") and f.endswith(".png")]
        self.prompt = CONFIG["prompt"]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.image_paths[i]).convert("RGB")
            return {"pixel_values": self.transform(img), "prompt": self.prompt}
        except Exception:
            next_idx = (i + 1) % len(self.image_paths)
            return self.__getitem__(next_idx)

dataset = OFTDataset("data/train", CONFIG["category"])
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

# تحسين أوزان الـ OFT فقط
optimizer = torch.optim.AdamW(unet.parameters(), lr=CONFIG["learning_rate"])

# الدرع الواقي (GradScaler) لمنع انفجار الأرقام
scaler = torch.amp.GradScaler('cuda')

print(f"\nStarting OFT (Orthogonal Finetuning) training...")
unet.train()

for epoch in range(CONFIG["epochs"]):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for batch in progress_bar:
        # VAE يعمل بدقة 16-بت
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]
        
        # تفعيل الدقة المختلطة المحمية (Autocast)
        with torch.amp.autocast('cuda'):
            # 🔥 توحيد لغة الأرقام لـ 32-بت لتجنب انهيار مكتبة xformers
            safe_noisy_latents = noisy_latents.to(dtype=torch.float32)
            safe_encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float32)
            
            noise_pred = unet(safe_noisy_latents, timesteps, safe_encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise.to(dtype=torch.float32))

        # التحديث الآمن
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

# حفظ أوزان الـ OFT فقط
unet.save_pretrained(CONFIG["output_dir"])
print(f"\n✅ OFT training completed perfectly. Weights saved to {CONFIG['output_dir']}")