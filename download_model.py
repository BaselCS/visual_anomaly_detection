#!/usr/bin/env python3
"""
Pre-download Stable Diffusion model to avoid downloading during training.
Run this once before training.
"""
import torch
from diffusers import StableDiffusionPipeline

print("Downloading Stable Diffusion model...")
print("This will take several minutes but only needs to be done once.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# Download and cache the model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable to save memory
    requires_safety_checker=False
)

print(f"✓ Model downloaded successfully and cached!")
print(f"Cache location: ~/.cache/huggingface/hub/")
print(f"You can now run main.py without waiting for downloads.")
