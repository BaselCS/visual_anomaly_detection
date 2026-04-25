import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

def get_next_train_dir(base_dir="trained_models"):
    os.makedirs(base_dir, exist_ok=True)
    max_num = 0
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_te_lora_"):
            try:
                num = int(folder_name.replace("train_te_lora_", ""))
                if num > max_num: max_num = num
            except ValueError:
                continue
    return os.path.join(base_dir, f"train_te_lora_{max_num + 1}")

CONFIG = {
    "category": "bottle",
    "prompt": "a high quality photo of a perfect bottle",
    "epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4, 
    "lora_rank": 8,
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

# تحويل المشفر إلى 32-بت ليعمل مع الـ GradScaler
text_encoder.to(dtype=torch.float32)

lora_config = LoraConfig(
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_rank"],
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    init_lora_weights="gaussian"
)

text_encoder = get_peft_model(text_encoder, lora_config)
text_encoder.print_trainable_parameters()

vae.requires_grad_(False)
unet.requires_grad_(False)

class TextEncoderLoRADataset(Dataset):
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

dataset = TextEncoderLoRADataset("data/train", CONFIG["category"])
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=CONFIG["learning_rate"])

# إضافة الدرع الواقي (GradScaler) لمنع انفجار الـ Float16
scaler = torch.amp.GradScaler('cuda')

print(f"Starting Text Encoder LoRA training...")
text_encoder.train()

for epoch in range(CONFIG["epochs"]):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # تفعيل الدقة المختلطة الآمنة (Autocast)
        with torch.amp.autocast('cuda'):
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]
            # U-Net سيحسب التفاضلات بأمان تحت حماية Autocast
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # التحديث الآمن باستخدام Scaler
        scaler.scale(loss).backward()
        
        # فك التشفير للـ Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

text_encoder.save_pretrained(CONFIG["output_dir"])
print(f"Text Encoder LoRA training completed. Weights saved to {CONFIG['output_dir']}")