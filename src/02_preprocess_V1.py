from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cv2

# CONFIG
IMG_SIZE = (224, 224)
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# =========================
# LOAD & RESIZE
# =========================
def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

# =========================
# ENHANCEMENT (SEKUNDER ONLY)
# =========================
def enhance_image(img: Image.Image) -> Image.Image:
    img_np = np.array(img)

    # Denoising ringan
    img_np = cv2.medianBlur(img_np, 3)

    # CLAHE (contrast enhancement)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img_np)

# =========================
# AUGMENTATION (TRAIN ONLY)
# =========================
def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    # Rotate
    angle = rng.uniform(15, 30) * rng.choice([-1, 1])
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

    # Flip
    if rng.random() > 0.5:
        img = ImageOps.mirror(img)

    # Zoom
    w, h = img.size
    zoom = rng.uniform(1.05, 1.2)
    nw, nh = int(w / zoom), int(h / zoom)
    left = rng.integers(0, w - nw + 1)
    top = rng.integers(0, h - nh + 1)
    img = img.crop((left, top, left + nw, top + nh))
    img = img.resize((w, h), Image.Resampling.BICUBIC)

    # Shift
    dx = int(rng.integers(-0.1*w, 0.1*w))
    dy = int(rng.integers(-0.1*h, 0.1*h))
    img = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )

    return img

# =========================
# SAVE
# =========================
def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)

# =========================
# PROCESS SPLIT
# =========================
def process_split(split, input_dir, output_dir, augment_train, aug_count, rng):
    if not input_dir.exists():
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        src_cls = input_dir / cls
        dst_cls = output_dir / cls

        if not src_cls.exists():
            continue

        for file in src_cls.rglob("*"):
            if file.suffix.lower() not in VALID_EXT:
                continue

            # Load + resize
            img = load_and_resize(file)

            # 🔥 Enhancement hanya untuk data sekunder
            if "sekunder" in file.name.lower():
                img = enhance_image(img)

            # Save original
            out_path = dst_cls / f"{file.stem}.jpg"
            save(img, out_path)

            # Augment hanya untuk train
            if augment_train and split == "train":
                for i in range(aug_count):
                    aug_img = augment(img, rng)
                    aug_path = dst_cls / f"{file.stem}_aug{i+1}.jpg"
                    save(aug_img, aug_path)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-count", type=int, default=1)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1]

    split_root = base_dir / "data" / "splits"
    output_root = base_dir / "data" / "processed"

    rng = np.random.default_rng(SEED)

    for split in ["train", "validation", "test"]:
        print(f"Processing {split}...")
        process_split(
            split=split,
            input_dir=split_root / split,
            output_dir=output_root / split,
            augment_train=args.augment,
            aug_count=max(1, args.aug_count),
            rng=rng
        )

    # COUNT
    def count_images(split_name):
        split_dir = output_root / split_name
        if not split_dir.exists():
            return 0
        return sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in VALID_EXT)

    train_count = count_images("train")
    val_count = count_images("validation")
    test_count = count_images("test")

    print("\n=== HASIL PREPROCESSING ===")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print(f"Total: {train_count + val_count + test_count}")
    print(f"Augmentasi: {args.augment}")
    print(f"Augment per image: {args.aug_count}")

if __name__ == "__main__":
    main()