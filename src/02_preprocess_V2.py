from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

IMG_SIZE = (224, 224) # Ukuran gambar yang diinginkan setelah resize
RESAMPLE_METHOD = Image.Resampling.BILINEAR # metode resampling untuk resize (BILINEAR lebih cepat, LANCZOS lebih berkualitas)
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42 # menjaga konsistensi hasil augmentasi

# fungsi center crop
def simple_crop_center(img) :
    w, h = img.size
    crop = int(min(w, h) * 0.8) # crop 80% dari sisi terpendek
    left = (w - crop) // 2
    top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))

def load_and_resize(path : Path) -> Image.Image :
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB") # convert ke RGB
    img = simple_crop_center(img)
    return img.resize(IMG_SIZE, RESAMPLE_METHOD)

def augment(img, rng) :
    if rng.random() > 0.5: # 50% chance untuk flip horizontal
        img = ImageOps.mirror(img)

    angle = rng.uniform(-20, 20) # rotasi acak antara -20 - 20 derajat
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

    brightness = rng.uniform(0.8, 1.2) # brightness antara 80% - 120%
    img = ImageEnhance.Brightness(img).enhance(brightness)

    contrast = rng.uniform(0.8, 1.2) # contrast 80% - 120%
    img = ImageEnhance.Contrast(img).enhance(contrast)

    if rng.random() > 0.7: # 30% chance untuk blur
        img = img.filter(ImageFilter.GaussianBlur(radius = 1))

    return img

def save(img, path) :
    path.parent.mkdir(parents = True, exist_ok = True)
    img.save(path, quality = 95) # simpan dengan kualotas 95%

def process_split(split, input_dir, output_dir, augment_train, aug_count, rng) :
    if not input_dir.exists() :
        return

    if output_dir.exists() :
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    for cls in CLASS_NAMES :
        src = input_dir / cls
        dst = output_dir / cls

        if not src.exists() :
            continue

        for file in src.rglob("*") :
            if file.suffix.lower() not in VALID_EXT :
                continue

            img = load_and_resize(file)
            save(img, dst / f"{file.stem}.jpg")

            if augment_train and split == "train" :
                for i in range(aug_count) :
                    aug_img = augment(img, rng)
                    save(aug_img, dst / f"{file.stem}_aug{i+1}.jpg")

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type = str, default = None)
    parser.add_argument("--augment", action = "store_true")
    parser.add_argument("--aug-count", type = int, default = 1)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1]

    split_root = base_dir / "data" / "splits"
    output_root = base_dir / "data" / "processed"

    rng = np.random.default_rng(SEED)

    for split in ["train", "validation", "test"] :
        print(f"Processing {split}...")
        process_split(
            split = split,
            input_dir = split_root / split,
            output_dir = output_root / split,
            augment_train = args.augment,
            aug_count = max(1, args.aug_count),
            rng = rng
        )

    # menghitung data hasil preprocess
    def count_images(split_name) :
        split_dir = output_root / split_name
        if not split_dir.exists() :
            return 0
        return sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in VALID_EXT)

    train_count = count_images("train")
    val_count = count_images("validation")
    test_count = count_images("test")

    print("\n=== HASIL PREPROCESSING ===")
    print(f"Train : {train_count}")
    print(f"Validation : {val_count}")
    print(f"Test : {test_count}")
    print(f"Total : {train_count + val_count + test_count}")
    print(f"Augmentasi : {args.augment}")
    print(f"Augment per image : {args.aug_count}")

if __name__ == "__main__" :
    main()