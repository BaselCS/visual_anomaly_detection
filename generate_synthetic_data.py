import argparse
import json
import logging
import os
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageEnhance, ImageFilter
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
NEGATIVE_PROMPT = (
    "defect, broken, crack, damaged, blur, text, logo, watermark, extra object, "
    "deformed, unrealistic, cartoon, painting, sketch, cgi, 3d render"
)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


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
        "--generation-method",
        type=str,
        default="augment",
        choices=["augment", "diffusion"],
        help="Synthetic generation method: augment (realism-first) or diffusion",
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
        default=5.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.12,
        help="Img2img strength (lower keeps closer features to training image)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=16,
        help="Perceptual hash size used for duplicate checking (higher = stricter detail)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=0,
        help="Max Hamming distance to treat as duplicate (0 = exact hash match only)",
    )
    parser.add_argument(
        "--randomize-environment",
        action="store_true",
        help="Randomize background/environment in augment mode to reduce environment bias",
    )
    parser.add_argument(
        "--environment-strength",
        type=float,
        default=0.7,
        help="How strongly to replace environment in augment mode (0..1)",
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


def preload_existing_hashes(
    train_dir: str,
    categories: List[str],
    category_hashes: Dict[str, set],
    hash_size: int,
) -> None:
    for filename in os.listdir(train_dir):
        category, _ = extract_category_and_index(filename)
        if category is None or category not in categories:
            continue

        img_path = os.path.join(train_dir, filename)
        try:
            with Image.open(img_path).convert("RGB") as img:
                category_hashes[category].add(average_hash(img, hash_size=hash_size))
        except Exception as exc:
            logger.warning(f"Failed to hash existing image {img_path}: {exc}")


def resolve_categories(requested_categories: List[str] | None, detected_categories: List[str]) -> List[str]:
    if requested_categories:
        return requested_categories
    if detected_categories:
        return sorted(detected_categories)
    return DEFAULT_CATEGORIES


def estimate_foreground_mask(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape

    border = max(6, int(min(h, w) * 0.06))
    border_pixels = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    bg_color = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(arr - bg_color, axis=2)

    threshold = max(14.0, float(np.percentile(dist, 72)))
    fg = (dist > threshold).astype(np.uint8) * 255

    mask = Image.fromarray(fg, mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    return mask


def make_random_environment(rng: random.Random, image_size: int) -> Image.Image:
    h, w = image_size, image_size
    base = np.zeros((h, w, 3), dtype=np.float32)

    c1 = np.array([rng.randint(150, 250), rng.randint(150, 250), rng.randint(150, 250)], dtype=np.float32)
    c2 = np.array([rng.randint(80, 220), rng.randint(80, 220), rng.randint(80, 220)], dtype=np.float32)

    horizontal = rng.random() < 0.5
    axis = np.linspace(0.0, 1.0, w if horizontal else h, dtype=np.float32)
    if horizontal:
        grad = axis[None, :, None]
    else:
        grad = axis[:, None, None]

    base = c1 * (1.0 - grad) + c2 * grad
    if horizontal:
        base = np.repeat(base, h, axis=0)
    else:
        base = np.repeat(base, w, axis=1)

    noise = np.random.normal(loc=0.0, scale=rng.uniform(4.0, 10.0), size=(h, w, 3)).astype(np.float32)
    base = np.clip(base + noise, 0, 255)
    return Image.fromarray(base.astype(np.uint8), mode="RGB")


def apply_environment_randomization(
    image: Image.Image,
    rng: random.Random,
    image_size: int,
    strength: float,
) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image

    fg_mask = estimate_foreground_mask(image)
    env = make_random_environment(rng, image_size)

    composite = Image.composite(image, env, fg_mask)
    return Image.blend(image, composite, strength)

def augment_from_source(
    source_image: Image.Image,
    rng: random.Random,
    image_size: int,
    randomize_environment: bool,
    environment_strength: float,
) -> Image.Image:
    img = source_image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    crop_scale = rng.uniform(0.92, 0.99)
    crop_w = int(image_size * crop_scale)
    crop_h = int(image_size * crop_scale)
    max_x = max(0, image_size - crop_w)
    max_y = max(0, image_size - crop_h)
    left = rng.randint(0, max_x) if max_x > 0 else 0
    top = rng.randint(0, max_y) if max_y > 0 else 0
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )

    brightness = rng.uniform(0.92, 1.08)
    contrast = rng.uniform(0.92, 1.08)
    color = rng.uniform(0.95, 1.05)
    sharpness = rng.uniform(0.95, 1.15)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    if randomize_environment:
        img = apply_environment_randomization(
            img,
            rng=rng,
            image_size=image_size,
            strength=environment_strength,
        )

    return img


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

    pipe = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.generation_method == "diffusion":
        logger.info("Loading Stable Diffusion Img2Img pipeline...")

        def load_pipe(target_device: str, local_only: bool):
            model_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if target_device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=local_only,
            ).to(target_device)
            model_pipe.enable_attention_slicing()
            model_pipe.enable_vae_slicing()
            return model_pipe

        try:
            pipe = load_pipe(device, local_only=True)
        except Exception as exc:
            is_oom = "out of memory" in str(exc).lower()
            if device == "cuda" and is_oom:
                logger.warning(f"GPU OOM while loading pipeline ({exc}). Falling back to CPU generation.")
                torch.cuda.empty_cache()
                device = "cpu"
                pipe = load_pipe(device, local_only=True)
            else:
                logger.warning(f"Offline load failed ({exc}), trying online mode...")
                pipe = load_pipe(device, local_only=False)
    else:
        logger.info("Using realism-first augmentation mode (no diffusion model loading).")

    category_training_images = collect_category_training_images(args.train_dir, categories)

    preload_existing_hashes(
        args.train_dir,
        categories,
        category_hashes,
        hash_size=args.hash_size,
    )

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

            source_image_path = random.choice(source_images)
            try:
                source_image = Image.open(source_image_path).convert("RGB").resize(
                    (args.image_size, args.image_size), Image.Resampling.BILINEAR
                )
            except Exception as exc:
                logger.warning(f"Failed to open source image {source_image_path}: {exc}")
                continue

            try:
                if args.generation_method == "augment":
                    aug_seed = args.seed + attempted + (stable_int(f"{category}:{os.path.basename(source_image_path)}") % 1_000_000)
                    aug_rng = random.Random(aug_seed)
                    image = augment_from_source(
                        source_image,
                        aug_rng,
                        args.image_size,
                        randomize_environment=args.randomize_environment,
                        environment_strength=args.environment_strength,
                    )
                else:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(args.seed + attempted + (stable_int(category) % 10000))
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

            candidate_hash = average_hash(image, hash_size=args.hash_size)
            existing_hashes = category_hashes.setdefault(category, set())
            is_duplicate = any(
                hamming_distance(candidate_hash, known_hash) <= args.duplicate_threshold
                for known_hash in existing_hashes
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
