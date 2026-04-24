import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

def get_next_train_dir(base_dir="trained_models"):
    os.makedirs(base_dir, exist_ok=True)
    max_num = 0
    for folder_name in os.listdir(base_dir):
        if folder_name.startswith("train_ti_"):
            try:
                num = int(folder_name.replace("train_ti_", ""))
                if num > max_num: max_num = num
            except ValueError:
                continue
    return os.path.join(base_dir, f"train_ti_{max_num + 1}")

CONFIG = {
    "category": "bottle",
    "placeholder_token": "<perfect-bottle>",
    "initializer_token": "bottle",
    "epochs": 250,
    "batch_size": 2,
    "learning_rate": 5e-5, 
    "output_dir": get_next_train_dir(),
    "seed": 999
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Stable Diffusion pipeline (Hybrid Precision Mode)...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, local_files_only=True
    ).to(device)
except Exception:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

# 🔥 الخدعة 1: تحويل الـ Text Encoder بالكامل إلى Float32 لضمان استقرار التدريب
text_encoder.to(dtype=torch.float32)

num_added_tokens = tokenizer.add_tokens(CONFIG["placeholder_token"])
placeholder_token_id = tokenizer.convert_tokens_to_ids(CONFIG["placeholder_token"])
text_encoder.resize_token_embeddings(len(tokenizer))

initializer_token_id = tokenizer.convert_tokens_to_ids(CONFIG["initializer_token"])
initial_embed = text_encoder.get_input_embeddings().weight.data[initializer_token_id]
text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = initial_embed.clone()

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

text_encoder.get_input_embeddings().requires_grad_(True)

class TextualInversionDataset(Dataset):
    def __init__(self, train_dir, category):
        self.image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith(f"{category}_") and f.endswith(".png")]
        self.prompt = f"a photo of {CONFIG['placeholder_token']}"
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

dataset = TextualInversionDataset("data/train", CONFIG["category"])
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=CONFIG["learning_rate"])

print(f"Starting Textual Inversion training for '{CONFIG['placeholder_token']}'...")
text_encoder.train()

for epoch in range(CONFIG["epochs"]):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    for batch in progress_bar:
        # الصور تذهب للـ VAE بصيغة Float16
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # الكلمات تخرج من الـ Text Encoder بصيغة Float32
        encoder_hidden_states = text_encoder(inputs.input_ids)[0]
        
        # 🔥 الخدعة 2: نحول الكلمات إلى Float16 قبل إدخالها للـ UNet
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)
        
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
        # نحول كل شيء لـ Float32 عند حساب الخطأ لحماية الأرقام
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        
        loss.backward()
        
        grads = text_encoder.get_input_embeddings().weight.grad
        index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
        grads.data[index_no_updates, :] = grads.data[index_no_updates, :].fill_(0)
        
        torch.nn.utils.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

learned_embeds = text_encoder.get_input_embeddings().weight[placeholder_token_id]
learned_embeds_dict = {CONFIG["placeholder_token"]: learned_embeds.detach().cpu()}
torch.save(learned_embeds_dict, os.path.join(CONFIG["output_dir"], "learned_embeds.bin"))

print(f"Textual Inversion completed. Embeddings saved to {CONFIG['output_dir']}")