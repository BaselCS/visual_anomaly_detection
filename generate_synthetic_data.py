import argparse
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("synthetic_data_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_CATEGORIES = ["bottle", "capsule", "pill", "toothbrush"]
CATEGORY_PROMPTS = {
    "bottle": "a high quality studio photo of a clean intact bottle product, centered, same shape and material, plain background",
    "capsule": "a high quality close-up photo of a pharmaceutical drug capsule, intact medicine capsule, same color and shape, plain background",
    "pill": "a high quality close-up photo of a pharmaceutical pill tablet, intact medicine tablet, same color and geometry, plain background",
    "toothbrush": "a high quality studio photo of an intact toothbrush, same handle shape and bristle structure, plain background",
}
NEGATIVE_PROMPT = "defect, broken, crack, damaged, blur, text, logo, watermark, extra object, deformed, unrealistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic normal images for anomaly-detection training."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/train",
        help="Flat train directory with files like category_001.png",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/train",
        help="Where synthetic images are written (default: append to train dir)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Categories to generate for. If omitted, auto-detect from train data.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Synthetic-to-real ratio per category (recommended 0.2-0.4)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Hard cap of generated images per category",
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=8.0,
        help="Minimum sharpness score; lower is blurrier",
    )
    parser.add_argument(
        "--max-tries-multiplier",
        type=int,
        default=8,
        help="Max attempts is target_count * this value",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Generation image size",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Diffusion inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.22,
        help="Img2img strength (lower keeps closer features to training image)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def extract_category_and_index(filename: str) -> Tuple[str, int] | Tuple[None, None]:
    if not filename.lower().endswith(".png"):
        return None, None

    stem = filename[:-4]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None, None

    category, idx_text = parts
    if not idx_text.isdigit():
        return None, None

    return category, int(idx_text)


def scan_train_dir(train_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, set]]:
    category_counts: Dict[str, int] = {}
    category_max_idx: Dict[str, int] = {}
    category_hashes: Dict[str, set] = {}

    for filename in os.listdir(train_dir):
        category, idx = extract_category_and_index(filename)
        if category is None:
            continue

        category_counts[category] = category_counts.get(category, 0) + 1
        category_max_idx[category] = max(category_max_idx.get(category, 0), idx)

    for category in category_counts:
        category_hashes[category] = set()

    return category_counts, category_max_idx, category_hashes


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    mean = arr.mean()
    bits = arr > mean

    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def blur_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def brightness_ok(image: Image.Image, min_mean: float = 25.0, max_mean: float = 230.0) -> bool:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mean_val = float(gray.mean())
    return min_mean <= mean_val <= max_mean


def build_prompt(category: str) -> str:
    if category in CATEGORY_PROMPTS:
        return CATEGORY_PROMPTS[category]
    return (
        f"a high quality studio photo of a perfect {category}, "
        "same shape and features, isolated object, plain background"
    )


def collect_category_training_images(train_dir: str, categories: List[str]) -> Dict[str, List[str]]:
    category_images: Dict[str, List[str]] = {c: [] for c in categories}
    for filename in os.listdir(train_dir):
        category, _ = extract_category_and_index(filename)
        if category is None or category not in category_images:
            continue
        category_images[category].append(os.path.join(train_dir, filename))

    for category in category_images:
        category_images[category].sort()

    return category_images


def preload_existing_hashes(train_dir: str, categories: List[str], category_hashes: Dict[str, set]) -> None
    for filename in os.listdir(train_dir):
        category, _ = extract_category_and_index(filename)
        if category is None or category not in categories:
            continue

        img_path = os.path.join(train_dir, filename)
        try:
            with Image.open(img_path).convert("RGB") as img:
                category_hashes[category].add(average_hash(img))
        except Exception as exc:
            logger.warning(f"Failed to hash existing image {img_path}: {exc}")


def resolve_categories(requested_categories: List[str] | None, detected_categories: List[str]) -> List[str]:
    if requested_categories:
        return requested_categories
    if detected_categories:
        return sorted(detected_categories)
    return DEFAULT_CATEGORIES


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    category_counts, category_max_idx, category_hashes = scan_train_dir(args.train_dir)
    categories = resolve_categories(args.categories, list(category_counts.keys()))

    logger.info("Detected category counts from train dir:")
    for category in categories:
        logger.info(f"  {category}: {category_counts.get(category, 0)} real/total images")

    logger.info("Loading Stable Diffusion Img2Img pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True,
        ).to(device)
    except Exception as exc:
        logger.warning(f"Offline load failed ({exc}), trying online mode...")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    category_training_images = collect_category_training_images(args.train_dir, categories)

    preload_existing_hashes(args.train_dir, categories, category_hashes)

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "categories": {},
    }

    for category in categories:
        real_count = category_counts.get(category, 0)
        target_count = max(1, int(round(real_count * args.ratio))) if real_count > 0 else 50
        if args.max_per_category is not None:
            target_count = min(target_count, args.max_per_category)

        max_tries = max(target_count * args.max_tries_multiplier, target_count + 10)
        current_idx = category_max_idx.get(category, 0)

        logger.info("=" * 64)
        logger.info(
            f"Category: {category} | real={real_count} | target_synthetic={target_count} | max_tries={max_tries}"
        )

        source_images = category_training_images.get(category, [])
        if not source_images:
            logger.warning(f"No source training images found for {category}. Skipping.")
            run_summary["categories"][category] = {
                "real_count": real_count,
                "target_synthetic": target_count,
                "generated": 0,
                "attempted": 0,
                "rejected_blur": 0,
                "rejected_brightness": 0,
                "rejected_duplicate": 0,
                "files": [],
                "skipped": "no_source_training_images",
            }
            continue

        generated = 0
        attempted = 0
        rejected_blur = 0
        rejected_brightness = 0
        rejected_duplicate = 0

        category_out = []
        progress = tqdm(total=target_count, desc=f"Generating {category}")

        while generated < target_count and attempted < max_tries:
            attempted += 1
            prompt = build_prompt(category)

            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + attempted + (hash(category) % 10000))

            source_image_path = random.choice(source_images)
            try:
                source_image = Image.open(source_image_path).convert("RGB").resize(
                    (args.image_size, args.image_size), Image.Resampling.BILINEAR
                )
            except Exception as exc:
                logger.warning(f"Failed to open source image {source_image_path}: {exc}")
                continue

            try:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=source_image,
                    strength=args.strength,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            except Exception as exc:
                logger.warning(f"Generation failed for {category} at try {attempted}: {exc}")
                continue

            sharpness = blur_score(image)
            if sharpness < args.min_blur_score:
                rejected_blur += 1
                continue

            if not brightness_ok(image):
                rejected_brightness += 1
                continue

            candidate_hash = average_hash(image)
            existing_hashes = category_hashes.setdefault(category, set())
            is_duplicate = any(
                hamming_distance(candidate_hash, known_hash) <= 8 for known_hash in existing_hashes
            )
            if is_duplicate:
                rejected_duplicate += 1
                continue

            current_idx += 1
            filename = f"{category}_{current_idx:03d}.png"
            save_path = os.path.join(args.output_dir, filename)
            image.save(save_path)

            existing_hashes.add(candidate_hash)
            generated += 1
            progress.update(1)

            category_out.append(
                {
                    "file": filename,
                    "prompt": prompt,
                    "source_image": os.path.basename(source_image_path),
                    "sharpness": sharpness,
                }
            )

        progress.close()

        logger.info(
            f"Finished {category}: generated={generated}, attempts={attempted}, "
            f"rejected_blur={rejected_blur}, rejected_brightness={rejected_brightness}, "
            f"rejected_duplicate={rejected_duplicate}"
        )

        run_summary["categories"][category] = {
            "real_count": real_count,
            "target_synthetic": target_count,
            "generated": generated,
            "attempted": attempted,
            "rejected_blur": rejected_blur,
            "rejected_brightness": rejected_brightness,
            "rejected_duplicate": rejected_duplicate,
            "files": category_out,
        }

    summary_name = f"synthetic_generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = os.path.join(args.output_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    logger.info("=" * 64)
    logger.info(f"Synthetic data generation complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
