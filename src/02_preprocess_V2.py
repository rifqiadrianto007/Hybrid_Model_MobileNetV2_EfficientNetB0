from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

IMG_SIZE = (224, 224)
RESAMPLE_METHOD = Image.Resampling.BILINEAR
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

def simple_crop_center(img):
    w, h = img.size
    crop = int(min(w, h) * 0.8)
    left = (w - crop) // 2
    top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))

def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = simple_crop_center(img)
    return img.resize(IMG_SIZE, RESAMPLE_METHOD)

def augment(img, rng):
    if rng.random() > 0.5:
        img = ImageOps.mirror(img)

    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

    brightness = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    contrast = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if rng.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)

def process_split(split, input_dir, output_dir, augment_train, aug_count, rng):
    if not input_dir.exists():
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        src = input_dir / cls
        dst = output_dir / cls

        if not src.exists():
            continue

        for file in src.rglob("*"):
            if file.suffix.lower() not in VALID_EXT:
                continue

            img = load_and_resize(file)
            save(img, dst / f"{file.stem}.jpg")

            if augment_train and split == "train":
                for i in range(aug_count):
                    aug_img = augment(img, rng)
                    save(aug_img, dst / f"{file.stem}_aug{i+1}.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-count", type=int, default=3)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1]
    split_root = base_dir / "data" / "splits"
    output_root = base_dir / "data" / "processed"

    rng = np.random.default_rng(SEED)

    for split in ["train", "validation", "test"]:
        process_split(
            split,
            split_root / split,
            output_root / split,
            args.augment,
            args.aug_count,
            rng
        )

if __name__ == "__main__":
    main()