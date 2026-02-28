#!/usr/bin/env python3
"""
Pre-download Stable Diffusion model to avoid downloading during training.
Run this once before training.
"""
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
CACHE_LOCATION = "~/.cache/huggingface/hub/"


def main() -> None:
    print("Downloading Stable Diffusion model...")
    print("This will take several minutes but only needs to be done once.")

    _ = "cuda" if torch.cuda.is_available() else "cpu"

    StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("✓ Model downloaded successfully and cached!")
    print(f"Cache location: {CACHE_LOCATION}")
    print("You can now run main.py without waiting for downloads.")


if __name__ == "__main__":
    main()
