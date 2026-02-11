import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

config = {
    "categories": ["bottle", "capsule", "pill", "toothbrush"],
    "epochs": 150, 
    "batch_size": 2,  # Optimized for RTX 3060
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 1e-7,  # (was 5e-5,1e-5)
    "max_grad_norm": 1.0,  # Gradient clipping to prevent explosion
    "lr_scheduler": "cosine",  # Learning rate decay
    "lr_warmup_steps": 100,  # Gradual LR warmup
    "save_every_n_epochs": 5,
    "save_log_every_n_epochs": 1,  # Save training log more frequently
    "output_dir": "trained_models",
    "image_size": 512,
    "early_stop_patience": 10,  # Stop if loss increases for N epochs
}

# Training Parameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

logger.info("Training Configuration:")
print("Training Configuration:")
for key, value in config.items():
    logger.info(f"  {key}: {value}")
    print(f"  {key}: {value}")
logger.info("="*50)
print("="*50)

os.makedirs(config["output_dir"], exist_ok=True)

logger.info("\nLoading Stable Diffusion model...")
print("\nLoading Stable Diffusion model...")
try:
    # Use local_files_only=True to prevent hanging on network issues
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable to save memory
        requires_safety_checker=False,
        local_files_only=True  # Prevent network hangs
    ).to(device)
except Exception as e:
    logger.error(f"Failed to load model in offline mode: {e}")
    print(f"Failed to load model in offline mode: {e}")
    print("Trying online mode (this may take time)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

vae = pipe.vae                           # Variational Autoencoder
text_encoder = pipe.text_encoder         # Text Encoder
tokenizer = pipe.tokenizer               # Tokenizer
unet = pipe.unet                         # U-Net

# Freeze VAE and text encoder to save memory
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

logger.info("✓ Model loaded successfully")
print("✓ Model loaded successfully")

# 4. Apply LoRA 
logger.info("\nApplying LoRA configuration...")
print("\nApplying LoRA configuration...")
lora_config = LoraConfig(
    r=16,               #rank
    lora_alpha=32,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"], # to_q for query, to_v for value, to_k for key, to_out.0 for output
    lora_dropout=0.05,  # REDUCED dropout for better fitting
    bias="none"
)


unet = get_peft_model(unet, lora_config)
logger.info("LoRA applied:")
print("LoRA applied:")
unet.print_trainable_parameters()


# 5. Data Loader 
logger.info("\nPreparing dataset...")
print("\nPreparing dataset...")
from torchvision import transforms

# data Augmation 
def get_transforms(category, image_size=512):
    """Get category-specific data augmentation transforms"""
    
    if category == "bottle":
        # Bottles are usually upright, minimal rotation, allow horizontal flip
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    elif category == "capsule":
        # Capsules can be rotated more, allow both flips
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    elif category == "pill":
        # Pills can be at any angle, allow all rotations and flips
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    elif category == "toothbrush":
        # Toothbrushes have orientation, minimal rotation, horizontal flip ok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    elif category == "zip":
        # Zippers have orientation, moderate rotation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    else:
        # Default transforms for unknown categories
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

class AnomalyDataset(Dataset):
    """Dataset for training on multiple product categories from flat directory"""
    def __init__(self, train_dir="data/train", categories=None, image_size=512):
        self.image_size = image_size
        self.samples = []
        self.category_transforms = {}  # Cache transforms per category
        
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory {train_dir} does not exist!")
        
        # Get all image files from flat train directory
        all_files = [f for f in os.listdir(train_dir) 
                    if f.lower().endswith('.png')]
        
        # Parse category from filename (format: category_001.png)
        category_counts = {}
        for img_file in all_files:
            # Extract category from filename (everything before last underscore and number)
            parts = img_file.rsplit('_', 1)
            if len(parts) == 2 and parts[1].replace('.png', '').isdigit():
                category = parts[0]
                
                # Filter by categories if specified
                if categories is None or category in categories:
                    img_path = os.path.join(train_dir, img_file)
                    prompt = f"a high quality photo of a perfect {category}"
                    self.samples.append((img_path, prompt, category))
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"Loaded {len(self.samples)} training images from {len(category_counts)} categories")
        print(f"Loaded {len(self.samples)} training images from {len(category_counts)} categories")
        for cat, count in sorted(category_counts.items()):
            logger.info(f"  {cat}: {count} images")
            print(f"  {cat}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt, category = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Get category-specific transform (cache it)
            if category not in self.category_transforms:
                self.category_transforms[category] = get_transforms(category, self.image_size)
            
            transform = self.category_transforms[category]
            image = transform(image)
            
            return {"pixel_values": image, "prompt": prompt, "category": category}
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            print(f"Error loading {img_path}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))


# Create full dataset
full_dataset = AnomalyDataset(
    train_dir="data/train", 
    categories=config["categories"], 
    image_size=config["image_size"]
)

# Split into train (90%) and validation (10%)
from torch.utils.data import random_split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

logger.info(f"\nDataset split: {train_size} training, {val_size} validation")
print(f"\nDataset split: {train_size} training, {val_size} validation")

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
# 6. Validation Function
def validate(unet, val_dataloader, vae, text_encoder, tokenizer, pipe, device):
    """Run validation and return average loss"""
    unet.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            try:
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                
                # Convert to latent space
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                inputs = tokenizer(
                    batch["prompt"], 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_losses.append(loss.item())
            except Exception as e:
                continue
    
    unet.train()
    return sum(val_losses) / len(val_losses) if val_losses else float('inf')

# 7. Training Loop 
optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])

# Learning rate scheduler with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
total_steps = len(train_dataloader) * config["epochs"] // config["gradient_accumulation_steps"]
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

unet.train()

# Training metrics
training_log = {
    "start_time": datetime.now().isoformat(),
    "config": config,
    "epochs": [],
    "best_epoch": None,
    "best_train_loss": float('inf'),
    "best_val_loss": float('inf')
}

logger.info(f"\nStarting training on {len(config['categories'])} categories...")
print(f"\nStarting training on {len(config['categories'])} categories...")
logger.info(f"Total epochs: {config['epochs']}")
print(f"Total epochs: {config['epochs']}")
logger.info(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
logger.info("\n" + "="*50)
print("\n" + "="*50)

global_step = 0
for epoch in range(config["epochs"]):
    epoch_losses = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    
    for step, batch in enumerate(progress_bar):
        try:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # text embedding
            with torch.no_grad():
                inputs = tokenizer(
                    batch["prompt"], 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]
            
            # Predict noise
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
            logger.error(f"\nError in training step: {e}")
            print(f"\nError in training step: {e}")
            continue
    avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    current_lr = optimizer.param_groups[0]['lr']
    
    # Run validation
    logger.info(f"Running validation...")
    print(f"Running validation...")
    avg_val_loss = validate(unet, val_dataloader, vae, text_encoder, tokenizer, pipe, device)
    
    logger.info(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
    print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
    
    # Track best model based on validation loss
    if avg_val_loss < training_log["best_val_loss"]:
        training_log["best_val_loss"] = avg_val_loss
        training_log["best_train_loss"] = avg_train_loss
        training_log["best_epoch"] = epoch + 1
        logger.info(f"  ✓ New best validation loss! Saving best model...")
        print(f"  ✓ New best validation loss! Saving best model...")
        best_model_dir = os.path.join(config["output_dir"], "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        unet.save_pretrained(best_model_dir)
        
        # Also save as a numbered checkpoint for the best epoch
        best_checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch+1}_BEST")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        unet.save_pretrained(best_checkpoint_dir)
        logger.info(f"  ✓ Best model checkpoint saved to {best_checkpoint_dir}")
        print(f"  ✓ Best model checkpoint saved to {best_checkpoint_dir}")
    
    # Early stopping check (if validation loss keeps increasing)
    if len(training_log["epochs"]) >= config["early_stop_patience"]:
        recent_val_losses = [e["val_loss"] for e in training_log["epochs"][-config["early_stop_patience"]:]]
        if all(recent_val_losses[i] < recent_val_losses[i+1] for i in range(len(recent_val_losses)-1)):
            logger.warning(f"\n⚠ WARNING: Validation loss has been increasing for {config['early_stop_patience']} epochs!")
            print(f"\n⚠ WARNING: Validation loss has been increasing for {config['early_stop_patience']} epochs!")
            logger.warning(f"  Consider stopping training. Best epoch was {training_log['best_epoch']} with val loss {training_log['best_val_loss']:.4f}")
            print(f"  Consider stopping training. Best epoch was {training_log['best_epoch']} with val loss {training_log['best_val_loss']:.4f}")
    
    training_log["epochs"].append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "learning_rate": current_lr,
        "samples_trained": len(epoch_losses)
    })
    
    # Save training log regularly to track progress
    if (epoch + 1) % config.get("save_log_every_n_epochs", 1) == 0:
        training_log["last_completed_epoch"] = epoch + 1
        with open(os.path.join(config["output_dir"], "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)
    
    # Save checkpoint
    if (epoch + 1) % config["save_every_n_epochs"] == 0:
        checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        unet.save_pretrained(checkpoint_dir)
        logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        print(f"✓ Checkpoint saved to {checkpoint_dir}")

logger.info("\n" + "="*50)
print("\n" + "="*50)
logger.info("Training completed!")
print("Training completed!")
logger.info(f"\n📊 Training Summary:")
print(f"\n📊 Training Summary:")
logger.info(f"  Best Epoch: {training_log['best_epoch']}")
print(f"  Best Epoch: {training_log['best_epoch']}")
logger.info(f"    Train Loss: {training_log['best_train_loss']:.4f}")
print(f"    Train Loss: {training_log['best_train_loss']:.4f}")
logger.info(f"    Val Loss: {training_log['best_val_loss']:.4f}")
print(f"    Val Loss: {training_log['best_val_loss']:.4f}")
logger.info(f"  Final Epoch: {config['epochs']}")
print(f"  Final Epoch: {config['epochs']}")
logger.info(f"    Train Loss: {avg_train_loss:.4f}")
print(f"    Train Loss: {avg_train_loss:.4f}")
logger.info(f"    Val Loss: {avg_val_loss:.4f}")
print(f"    Val Loss: {avg_val_loss:.4f}")
if training_log['best_val_loss'] < avg_val_loss:
    logger.warning(f"  ⚠ Final model has higher validation loss - use 'best_model' directory instead!")
    print(f"  ⚠ Final model has higher validation loss - use 'best_model' directory instead!")

# 7. Save Final Model 
logger.info("\nSaving final trained model...")
print("\nSaving final trained model...")
final_model_dir = os.path.join(config["output_dir"], "final_model")
os.makedirs(final_model_dir, exist_ok=True)
unet.save_pretrained(final_model_dir)

# Save training log
training_log["end_time"] = datetime.now().isoformat()
with open(os.path.join(config["output_dir"], "training_log.json"), "w") as f:
    json.dump(training_log, f, indent=2)

logger.info(f"✓ Model saved to {final_model_dir}")
print(f"✓ Model saved to {final_model_dir}")
logger.info(f"✓ Training log saved to {os.path.join(config['output_dir'], 'training_log.json')}")
print(f"✓ Training log saved to {os.path.join(config['output_dir'], 'training_log.json')}")

# 8. Generate Test Samples 
logger.info("\nGenerating test samples...")
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
        logger.info(f"✓ Generated test sample for {category}")
        print(f"✓ Generated test sample for {category}")
    except Exception as e:
        logger.error(f"✗ Failed to generate sample for {category}: {e}")
        print(f"✗ Failed to generate sample for {category}: {e}")

# 9. Plot Training and Validation Metrics
logger.info("\nGenerating training plots...")
print("\nGenerating training plots...")

try:
    epochs = [e['epoch'] for e in training_log['epochs']]
    train_losses = [e['train_loss'] for e in training_log['epochs']]
    val_losses = [e['val_loss'] for e in training_log['epochs']]
    learning_rates = [e['learning_rate'] for e in training_log['epochs']]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.legend(fontsize=11)
    
    # Mark best epoch
    best_epoch = training_log['best_epoch']
    best_val_loss = training_log['best_val_loss']
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
    ax1.scatter(best_epoch, best_val_loss, c='green', s=100, zorder=5, marker='*')
    ax1.legend(fontsize=11)
    
    # Plot 2: Learning Rate
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add training info
    info_text = f"Training Configuration:\n"
    info_text += f"Categories: {', '.join(config['categories'])}\n"
    info_text += f"Epochs: {config['epochs']}\n"
    info_text += f"Batch Size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})\n"
    info_text += f"Train/Val Split: 90/10\n"
    info_text += f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})"
    
    fig.text(0.98, 0.02, info_text, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    # Save the figure
    plot_path = os.path.join(config["output_dir"], 'training_validation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Training plot saved to {plot_path}")
    print(f"✓ Training plot saved to {plot_path}")
except Exception as e:
    logger.error(f"Failed to generate training plot: {e}")
    print(f"Failed to generate training plot: {e}")

logger.info("\n" + "="*50)
print("\n" + "="*50)
logger.info("All done! You can now run anomal_score.py to test the model.")
print("All done! You can now run anomal_score.py to test the model.")
logger.info("="*50)
print("="*50)