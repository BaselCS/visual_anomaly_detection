from collections import defaultdict
from pathlib import Path
from typing import Dict

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _iter_split_images(root: Path, split: str) -> list[Path]:
    images: list[Path] = []
    split_dir = root / split

    if split_dir.exists():
        images.extend([path for path in split_dir.rglob("*") if _is_image_file(path)])

    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        nested_split = category_dir / split
        if nested_split.exists():
            images.extend([path for path in nested_split.rglob("*") if _is_image_file(path)])

    unique_images = list(dict.fromkeys(images))
    return unique_images


def _infer_category(image_path: Path, root: Path, split: str) -> str:
    parts = image_path.parts
    if split in parts:
        split_index = parts.index(split)
        if split_index > 0:
            category_candidate = parts[split_index - 1]
            if category_candidate not in {"data", split}:
                return category_candidate

    name = image_path.stem.lower()
    categories = [
        path.name.lower()
        for path in root.iterdir()
        if path.is_dir() and path.name not in {"train", "test", "zip", "ground_truth"}
    ]
    for category in sorted(categories, key=len, reverse=True):
        if name == category or name.startswith(f"{category}_"):
            return category

    if "_" in name:
        return name.split("_", 1)[0]

    return "unknown"


def get_train_test_image_sizes(data_root: str = "data") -> Dict[str, dict]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    result: Dict[str, dict] = {
        "train": {"total_images": 0, "sizes": defaultdict(int), "categories": {}},
        "test": {"total_images": 0, "sizes": defaultdict(int), "categories": {}},
    }

    for split in ("train", "test"):
        split_category_sizes: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
        image_paths = _iter_split_images(root, split)

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    size = (image.width, image.height)
            except Exception:
                continue

            category = _infer_category(image_path, root, split)
            split_category_sizes[category][size] += 1
            result[split]["sizes"][size] += 1
            result[split]["total_images"] += 1

        for category, category_sizes in split_category_sizes.items():
            result[split]["categories"][category] = {
                "image_count": sum(category_sizes.values()),
                "sizes": dict(sorted(category_sizes.items())),
            }

    result["train"]["sizes"] = dict(sorted(result["train"]["sizes"].items()))
    result["test"]["sizes"] = dict(sorted(result["test"]["sizes"].items()))
    return result


def _print_report(report: Dict[str, dict]) -> None:
    for split in ("train", "test"):
        split_data = report[split]
        print(f"\n{split.upper()} SUMMARY")
        print(f"Total images: {split_data['total_images']}")
        print("Sizes (width, height) -> count:")
        if split_data["sizes"]:
            for size, count in split_data["sizes"].items():
                print(f"  {size}: {count}")
        else:
            print("  No images found")

        print("Per category:")
        if split_data["categories"]:
            for category, info in split_data["categories"].items():
                print(f"  - {category}: {info['image_count']} images")
                for size, count in info["sizes"].items():
                    print(f"      {size}: {count}")
        else:
            print("  No category data")


if __name__ == "__main__":
    image_report = get_train_test_image_sizes("data")
    _print_report(image_report)
