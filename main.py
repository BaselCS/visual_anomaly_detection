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
    "all_categories": ["bottle", "capsule", "pill", "toothbrush"],  # Train all, one by one
    "epochs": 100,  # Epochs per category
    "batch_size": 2,  # Increased from 1 for RTX 3060
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 5e-5,  # REDUCED to prevent gradient explosion (was 1e-4)
    "max_grad_norm": 1.0,  # Gradient clipping to prevent explosion
    "lr_scheduler": "cosine",  # Learning rate decay
    "lr_warmup_steps": 100,  # Gradual LR warmup
    "save_every_n_epochs": 10,
    "output_dir": "trained_models",
    "image_size": 512,
    "early_stop_patience": 10,  # Stop if loss increases for N epochs
}

print("="*50)
print("Training Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print("="*50)
print(f"\n🚀 Will train {len(config['all_categories'])} categories sequentially:")
for i, cat in enumerate(config['all_categories'], 1):
    print(f"  {i}. {cat}")
print("="*50)

# Create output directory
os.makedirs(config["output_dir"], exist_ok=True)

# ==============================================================================
# MAIN TRAINING LOOP - Train each category separately
# ==============================================================================
for category_idx, current_category in enumerate(config['all_categories'], 1):
    print(f"\n\n{'='*70}")
    print(f"📦 TRAINING CATEGORY {category_idx}/{len(config['all_categories'])}: {current_category.upper()}")
    print(f"{'='*70}\n")

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
        r=16,  # INCREASED from 8 for more capacity
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,  # REDUCED dropout for better fitting
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    print("LoRA applied:")
    unet.print_trainable_parameters()

# 5. Data Loader (تحميل البيانات)
print("\nPreparing dataset...")
preprocess = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(p=0.3),  # Data augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Slight color variation
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

    dataset = AnomalyDataset(
        root_dir="data", 
        categories=[current_category],  # Train one category at a time
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

    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    total_steps = len(train_dataloader) * config["epochs"] // config["gradient_accumulation_steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    unet.train()

    # Training metrics
    training_log = {
        "category": current_category,
        "start_time": datetime.now().isoformat(),
        "config": config,
        "epochs": [],
        "best_epoch": None,
        "best_loss": float('inf')
    }

    print(f"\nStarting training for: {current_category.upper()}")
    print(f"Total epochs: {config['epochs']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Training samples: {len(dataset)}")
    print("\n" + "="*50)

    global_step = 0
    for epoch in range(config["epochs"]):
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"[{current_category}] Epoch {epoch+1}/{config['epochs']}")
        
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
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_losses.append(loss.item() * config["gradient_accumulation_steps"])
                progress_bar.set_postfix({"loss": f"{epoch_losses[-1]:.4f}"})
                
            except Exception as e:
                print(f"\nError in training step: {e}")
                continue
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{current_category}] Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        # Track best model
        if avg_loss < training_log["best_loss"]:
            training_log["best_loss"] = avg_loss
            training_log["best_epoch"] = epoch + 1
            print(f"  ✓ New best loss! Saving best model...")
            best_model_dir = os.path.join(config["output_dir"], f"{current_category}_best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            unet.save_pretrained(best_model_dir)
        
        # Early stopping check (if loss keeps increasing)
        if len(training_log["epochs"]) >= config["early_stop_patience"]:
            recent_losses = [e["avg_loss"] for e in training_log["epochs"][-config["early_stop_patience"]:]]
            if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                print(f"\n⚠ WARNING: Loss has been increasing for {config['early_stop_patience']} epochs!")
                print(f"  Consider stopping training. Best epoch was {training_log['best_epoch']} with loss {training_log['best_loss']:.4f}")
        
        training_log["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "learning_rate": current_lr,
            "samples_trained": len(epoch_losses)
        })
        
        # Save checkpoint
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            checkpoint_dir = os.path.join(config["output_dir"], f"{current_category}_checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            unet.save_pretrained(checkpoint_dir)
            print(f"✓ Checkpoint saved to {checkpoint_dir}")

    print("\n" + "="*50)
    print(f"Training completed for {current_category}!")
    print(f"\n📊 Training Summary for {current_category.upper()}:")
    print(f"  Best Epoch: {training_log['best_epoch']} (Loss: {training_log['best_loss']:.4f})")
    print(f"  Final Epoch: {config['epochs']} (Loss: {avg_loss:.4f})")
    if training_log['best_loss'] < avg_loss:
        print(f"  ⚠ Final model is worse than best - use '{current_category}_best_model' directory instead!")

    # 7. Save Final Model (حفظ النموذج النهائي)
    print("\nSaving final trained model...")
    final_model_dir = os.path.join(config["output_dir"], f"{current_category}_final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    unet.save_pretrained(final_model_dir)

    # Save training log
    training_log["end_time"] = datetime.now().isoformat()
    with open(os.path.join(config["output_dir"], f"{current_category}_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"✓ Model saved to {final_model_dir}")
    print(f"✓ Training log saved to {os.path.join(config['output_dir'], f'{current_category}_training_log.json')}")

    # 8. Generate Test Samples (توليد عينات اختبار)
    print("\nGenerating test samples...")
    unet.eval()
    pipe.unet = unet

    test_output_dir = os.path.join(config["output_dir"], "test_generations")
    os.makedirs(test_output_dir, exist_ok=True)

    try:
        test_prompt = f"a high quality photo of a perfect {current_category}"
        output_image = pipe(test_prompt, num_inference_steps=30).images[0]
        output_path = os.path.join(test_output_dir, f"{current_category}_sample.png")
        output_image.save(output_path)
        print(f"✓ Generated test sample for {current_category}")
    except Exception as e:
        print(f"✗ Failed to generate sample for {current_category}: {e}")
    
    # Clean up GPU memory before next category
    del unet, pipe, optimizer, scheduler
    torch.cuda.empty_cache()
    print(f"\n✓ {current_category.upper()} training complete! Moving to next category...\n")

# ==============================================================================
# ALL CATEGORIES TRAINED
# ==============================================================================
print("\n" + "="*70)
print("🎉 ALL CATEGORIES TRAINED SUCCESSFULLY!")
print("="*70)
print(f"\nTrained models for: {', '.join(config['all_categories'])}")
print(f"\nModel locations:")
for cat in config['all_categories']:
    print(f"  • {cat}: trained_models/{cat}_best_model/")
print("\n" + "="*70)
print("You can now run anomal_score.py to test all models.")
print("="*70)