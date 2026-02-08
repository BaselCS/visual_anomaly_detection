import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from datetime import datetime

# 1. Dataset Configuration (إعدادات مجموعة البيانات)
class AnomalyDataset(Dataset):
    """Dataset for training on multiple product categories"""
    def __init__(self, root_dir, categories=None, transform=None):
        self.transform = transform
        self.samples = []
        
        if categories is None:
            categories = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))]
        
        for category in categories:
            train_path = os.path.join(root_dir, category, "train", "good")
            if not os.path.exists(train_path):
                print(f"Warning: {train_path} does not exist, skipping {category}")
                continue
                
            image_files = [f for f in os.listdir(train_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(train_path, img_file)
                prompt = f"a high quality photo of a perfect {category}"
                self.samples.append((img_path, prompt, category))
        
        print(f"Loaded {len(self.samples)} training images from {len(categories)} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt, category = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return {"pixel_values": image, "prompt": prompt, "category": category}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

# 2. Training Parameters (بارامترات التدريب)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# Training configuration
config = {
    "categories": ["bottle", "capsule", "pill", "toothbrush"],  # Train on all categories
    "epochs": 50,  # Increased for better results
    "batch_size": 2,  # Increased from 1 for RTX 3060
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 1e-4,
    "save_every_n_epochs": 10,
    "output_dir": "trained_models",
    "image_size": 512,
}

print("="*50)
print("Training Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print("="*50)

# Create output directory
os.makedirs(config["output_dir"], exist_ok=True)

# 3. Load Models (تحميل النماذج)
print("\nLoading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable to save memory
    requires_safety_checker=False
).to(device)

vae = pipe.vae
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
unet = pipe.unet

# Freeze VAE and text encoder to save memory
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

print("✓ Model loaded successfully")

# 4. Apply LoRA (تطبيق تقنية LoRA)
print("\nApplying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
    lora_dropout=0.1,
    bias="none"
)
unet = get_peft_model(unet, lora_config)
print("LoRA applied:")
unet.print_trainable_parameters()

# 5. Data Loader (تحميل البيانات)
print("\nPreparing dataset...")
preprocess = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = AnomalyDataset(
    root_dir="data", 
    categories=config["categories"], 
    transform=preprocess
)
train_dataloader = DataLoader(
    dataset, 
    batch_size=config["batch_size"], 
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# 6. Training Loop (حلقة التدريب)
optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])
unet.train()

# Training metrics
training_log = {
    "start_time": datetime.now().isoformat(),
    "config": config,
    "epochs": []
}

print(f"\nStarting training on {len(config['categories'])} categories...")
print(f"Total epochs: {config['epochs']}")
print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
print("\n" + "="*50)

global_step = 0
for epoch in range(config["epochs"]):
    epoch_losses = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for step, batch in enumerate(progress_bar):
        try:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            
            # Convert images to latent space (تحويل الصور إلى الفضاء الكامن)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Sample noise (توليد الضجيج)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embedding (تحويل النص إلى تمثيل رقمي)
            with torch.no_grad():
                inputs = tokenizer(
                    batch["prompt"], 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
            
            # Predict noise (التنبؤ بالضجيج)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_losses.append(loss.item() * config["gradient_accumulation_steps"])
            progress_bar.set_postfix({"loss": f"{epoch_losses[-1]:.4f}"})
            
        except Exception as e:
            print(f"\nError in training step: {e}")
            continue
    
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    
    training_log["epochs"].append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "samples_trained": len(epoch_losses)
    })
    
    # Save checkpoint
    if (epoch + 1) % config["save_every_n_epochs"] == 0:
        checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        unet.save_pretrained(checkpoint_dir)
        print(f"✓ Checkpoint saved to {checkpoint_dir}")

print("\n" + "="*50)
print("Training completed!")

# 7. Save Final Model (حفظ النموذج النهائي)
print("\nSaving final trained model...")
final_model_dir = os.path.join(config["output_dir"], "final_model")
os.makedirs(final_model_dir, exist_ok=True)
unet.save_pretrained(final_model_dir)

# Save training log
training_log["end_time"] = datetime.now().isoformat()
with open(os.path.join(config["output_dir"], "training_log.json"), "w") as f:
    json.dump(training_log, f, indent=2)

print(f"✓ Model saved to {final_model_dir}")
print(f"✓ Training log saved to {os.path.join(config['output_dir'], 'training_log.json')}")

# 8. Generate Test Samples (توليد عينات اختبار)
print("\nGenerating test samples...")
unet.eval()
pipe.unet = unet

test_output_dir = os.path.join(config["output_dir"], "test_generations")
os.makedirs(test_output_dir, exist_ok=True)

for category in config["categories"]:
    try:
        test_prompt = f"a high quality photo of a perfect {category}"
        output_image = pipe(test_prompt, num_inference_steps=30).images[0]
        output_path = os.path.join(test_output_dir, f"{category}_sample.png")
        output_image.save(output_path)
        print(f"✓ Generated test sample for {category}")
    except Exception as e:
        print(f"✗ Failed to generate sample for {category}: {e}")

print("\n" + "="*50)
print("All done! You can now run anomal_score.py to test the model.")
print("="*50)