import os
import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt

def get_next_train_dir(base_dir="trained_models"):
    """Find the next available trainX directory"""
    os.makedirs(base_dir, exist_ok=True)
    train_num = 1
    while os.path.exists(os.path.join(base_dir, f"train{train_num}")):
        train_num += 1
    return os.path.join(base_dir, f"train{train_num}")

DEFAULT_CONFIG = {
    "categories": ["bottle", "capsule", "pill", "toothbrush"],
    "use_data_augmentation": False,  # Set to True to enable category-specific data augmentation
    "epochs": 120,
    "batch_size": 2,                  # Optimized for 3060 12GB
    "gradient_accumulation_steps": 4, # Effective batch size = 2 * 4 = 8
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "max_grad_norm": 1.0,             # Gradient clipping to prevent explosion
    "lr_scheduler": "cosine",         # Learning rate decay
    "lr_warmup_steps": 100,           # Number of warmup steps for learning rate scheduler
    "lr_eta_min": 1e-6,
    "save_every_n_epochs": 5,
    "save_log_every_n_epochs": 5,
    "output_dir": get_next_train_dir("trained_models"),
    "image_size": 512,
    "train_split": 0.9,               # 90% training, 10% validation split 
    "seed": 999,
    "early_stop_patience": 30,     # Stop if no improvement for x epochs
    "early_stop_min_delta": 1e-4,
    "min_epochs_before_early_stop": 50, # Don't allow early stopping before this many epochs
    "num_workers": 0,              # number of subprocesses for data loading, adjust based on CPU cores and memory, Windows users may want to set this to 0 for compatibility
}

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


def out(message: str, level: str = "info") -> None:
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)
    print(message)

def load_config() -> dict:
    return dict(DEFAULT_CONFIG)


config = load_config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") # For reproducibility in CUDA operations
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(config["seed"])

# Training Parameters 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

logger.info("Training Configuration:")
out("Training Configuration:")
for key, value in config.items():
    out(f"  {key}: {value}")
out("=" * 50)

os.makedirs(config["output_dir"], exist_ok=True)

out("\nLoading Stable Diffusion model...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None,  # NSFW safety, add memory overhead we don't need
        requires_safety_checker=False,
        local_files_only=True  # Prevent network hangs
    ).to(device)

except Exception as e:
    out(f"Failed to load model in offline mode: {e}", level="error")
    out("Trying online mode (this may take time)...")
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

out("✓ Model loaded successfully")

# 4. Apply LoRA 
out("\nApplying LoRA configuration...")
lora_config = LoraConfig(
    r=16,               #rank
    lora_alpha=32,
    target_modules=["to_q", "to_v", "to_k", "to_out.0"], # to_q for query, to_v for value, to_k for key, to_out.0 for output
    lora_dropout=0.05,  # REDUCED dropout for better fitting
    bias="none"
)


unet = get_peft_model(unet, lora_config)
out("LoRA applied:")
unet.print_trainable_parameters()


# 5. Data Loader 
out("\nPreparing dataset...")

# Data augmentation
def get_transforms(category, image_size=512):
    """Get category-specific data augmentation transforms"""
    category_params = {
        "bottle": {
            "rotation": 8,
            "hflip": 0.3,
            "vflip": 0.0,
            "translate": 0.03,
            "jitter": (0.1, 0.1, 0.05),
            "sharpness_p": 0.2,
        },
        "capsule": {
            "rotation": 30,
            "hflip": 0.5,
            "vflip": 0.3,
            "translate": 0.05,
            "jitter": (0.15, 0.15, 0.1),
            "sharpness_p": 0.25,
        },
        "pill": {
            "rotation": 180,
            "hflip": 0.5,
            "vflip": 0.5,
            "translate": 0.06,
            "jitter": (0.12, 0.12, 0.1),
            "sharpness_p": 0.2,
        },
        "toothbrush": {
            "rotation": 5,
            "hflip": 0.4,
            "vflip": 0.0,
            "translate": 0.04,
            "jitter": (0.1, 0.1, 0.08),
            "sharpness_p": 0.15,
        },
        "default": {
            "rotation": 15,
            "hflip": 0.3,
            "vflip": 0.0,
            "translate": 0.05,
            "jitter": (0.1, 0.1, 0.1),
            "sharpness_p": 0.2,
        },
    }

    params = category_params.get(category)
    if params is None:
        out(
            f"Warning: {'🚩' * 20} No specific transforms for category '{category}', using default.",
            level="warning",
        )
        params = category_params["default"]

    augmentation_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=params["rotation"]),
        transforms.RandomHorizontalFlip(p=params["hflip"]),
    ]

    if params["vflip"] > 0:
        augmentation_steps.append(transforms.RandomVerticalFlip(p=params["vflip"]))

    brightness, contrast, saturation = params["jitter"]
    augmentation_steps.extend(
        [
            transforms.RandomAffine(degrees=0, translate=(params["translate"], params["translate"])),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=params["sharpness_p"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return transforms.Compose(augmentation_steps)


def get_basic_transform(image_size=512):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class AnomalyDataset(Dataset):
    """
        Custom dataset loader that reads images from a flat directory, extracts category from filename, and applies category-specific data augmentation.
    """
    def __init__(self, train_dir="data/train", categories=None, image_size=512, use_data_augmentation=True):
        self.image_size = image_size
        self.use_data_augmentation = use_data_augmentation
        self.samples = []
        self.category_transforms = {}  # Cache transforms per category
        self.basic_transform = get_basic_transform(self.image_size)
        
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
        
        out(f"Loaded {len(self.samples)} training images from {len(category_counts)} categories")
        for cat, count in sorted(category_counts.items()):
            out(f"  {cat}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prompt, category = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")

            if self.use_data_augmentation:
                if category not in self.category_transforms:
                    self.category_transforms[category] = get_transforms(category, self.image_size)
                transform = self.category_transforms[category]
            else:
                transform = self.basic_transform

            image = transform(image)
            
            return {"pixel_values": image, "prompt": prompt, "category": category}
        except Exception as e:
            out(f"Error loading {img_path}: {e}", level="error")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))


# Create full dataset
full_dataset = AnomalyDataset(
    train_dir="data/train", 
    categories=config["categories"], 
    image_size=config["image_size"],
    use_data_augmentation=config["use_data_augmentation"],
)

# Create train/validation split with category balance
rng = random.Random(config["seed"])
category_to_indices = {}
for idx, (_, _, category) in enumerate(full_dataset.samples):
    category_to_indices.setdefault(category, []).append(idx)

train_indices = []
val_indices = []

for category, indices in sorted(category_to_indices.items()):
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * config["train_split"]))
    split_at = min(split_at, len(indices) - 1) if len(indices) > 1 else len(indices)
    train_indices.extend(indices[:split_at])
    val_indices.extend(indices[split_at:])

# Fallback safety in case of tiny datasets in Validation set - ensure at least 1 sample in validation if possible
if len(val_indices) == 0 and len(train_indices) > 1:
    val_indices.append(train_indices.pop())

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_size = len(train_indices)
val_size = len(val_indices)

logger.info(f"\nDataset split: {train_size} training, {val_size} validation")
out(f"\nDataset split: {train_size} training, {val_size} validation")

# Create weighted sampler for category-balanced training batches
train_category_counts = {}
for dataset_idx in train_indices:
    _, _, category = full_dataset.samples[dataset_idx]
    train_category_counts[category] = train_category_counts.get(category, 0) + 1

sample_weights = []
for dataset_idx in train_indices:
    _, _, category = full_dataset.samples[dataset_idx]
    sample_weights.append(1.0 / train_category_counts[category])

weighted_sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
    generator=torch.Generator().manual_seed(config["seed"]),
)

# pytorch DataLoaders with weighted sampler for training and simple sequential sampling for validation
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    sampler=weighted_sampler,
    num_workers=config["num_workers"],
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False,
    num_workers=config["num_workers"],
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
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215 # Scaling factor for Stable Diffusion latent space
                
                # Sample noise
                noise = torch.randn(
                    latents.shape,
                    device=device,
                    dtype=latents.dtype,
                    generator=validation_noise_generator,
                )
                # how much noise to add based on random timestep
                timesteps = torch.randint(
                    0,
                    1000,
                    (latents.shape[0],),
                    device=device,
                    generator=validation_noise_generator,
                ).long()
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

# only optimizing LoRA parameters
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

# Learning rate scheduler with linear warmup + cosine decay
from torch.optim.lr_scheduler import LambdaLR
updates_per_epoch = max(1, math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"]))
total_updates = updates_per_epoch * config["epochs"]
warmup_updates = min(config["lr_warmup_steps"], max(1, total_updates // 5))


def lr_lambda(current_update):
    if current_update < warmup_updates:
        return float(current_update + 1) / float(max(1, warmup_updates))

    progress = float(current_update - warmup_updates) / float(max(1, total_updates - warmup_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_factor = config["lr_eta_min"] / config["learning_rate"]
    return max(min_factor, cosine)


scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

training_noise_generator = torch.Generator(device=device)
training_noise_generator.manual_seed(config["seed"] + 1000)
validation_noise_generator = torch.Generator(device=device)
validation_noise_generator.manual_seed(config["seed"] + 2000)

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
epochs_without_improvement = 0
# Main training loop with validation and early stopping based on validation loss improvement
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
            noise = torch.randn(
                latents.shape,
                device=device,
                dtype=latents.dtype,
                generator=training_noise_generator,
            )
            timesteps = torch.randint(
                0,
                1000,
                (latents.shape[0],),
                device=device,
                generator=training_noise_generator,
            ).long()
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

    # Handle leftover gradients when steps are not divisible by accumulation
    total_batches = len(epoch_losses)
    if total_batches > 0 and (total_batches % config["gradient_accumulation_steps"] != 0):
        torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    current_lr = optimizer.param_groups[0]['lr']
    
    # Run validation
    logger.info(f"Running validation...")
    print(f"Running validation...")
    avg_val_loss = validate(unet, val_dataloader, vae, text_encoder, tokenizer, pipe, device)
    
    logger.info(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
    print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")
    
    # Track best model based on validation loss
    if avg_val_loss < (training_log["best_val_loss"] - config["early_stop_min_delta"]):
        training_log["best_val_loss"] = avg_val_loss
        training_log["best_train_loss"] = avg_train_loss
        training_log["best_epoch"] = epoch + 1
        epochs_without_improvement = 0
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
    else:
        epochs_without_improvement += 1
    
    # True early stopping on no meaningful improvement
    if (epoch + 1) >= config["min_epochs_before_early_stop"] and epochs_without_improvement >= config["early_stop_patience"]:
        logger.warning(
            f"\n⚠ Early stopping triggered at epoch {epoch + 1}: "
            f"no val-loss improvement > {config['early_stop_min_delta']} for {epochs_without_improvement} epochs."
        )
        print(
            f"\n⚠ Early stopping triggered at epoch {epoch + 1}: "
            f"no val-loss improvement > {config['early_stop_min_delta']} for {epochs_without_improvement} epochs."
        )
        logger.warning(
            f"  Best epoch was {training_log['best_epoch']} with val loss {training_log['best_val_loss']:.4f}"
        )
        print(f"  Best epoch was {training_log['best_epoch']} with val loss {training_log['best_val_loss']:.4f}")

        training_log["last_completed_epoch"] = epoch + 1
        with open(os.path.join(config["output_dir"], "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)
        break
    
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
logger.info(f"\n Training Summary:")
print(f"\n Training Summary:")
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
    logger.warning(f"  Final model has higher validation loss - use 'best_model' directory instead!")
    print(f"  Final model has higher validation loss - use 'best_model' directory instead!")

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

logger.info(f"Model saved to {final_model_dir}")
print(f"Model saved to {final_model_dir}")

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
        logger.info(f"Generated test sample for {category}")
        print(f"Generated test sample for {category}")
    except Exception as e:
        logger.error(f"Failed to generate sample for {category}: {e}")
        print(f" Failed to generate sample for {category}: {e}")

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
