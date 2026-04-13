# Melakukan preprocessing dataset
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

# Konstanta global
IMG_SIZE = (224, 224)
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# Load, resize, dan convert dataset ke RGB
def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

# Augmentasi gambar untuk meningkatkan variasi data
def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    # Rotate 15-30 derajat secara acak
    angle = rng.uniform(15, 30) * rng.choice([-1, 1])
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC)

    # Flipping horizontal dengan probabilitas 50%
    if rng.random() > 0.5:
        img = ImageOps.mirror(img)

    # Zoom 5-20% dengan cropping dan resizing kembali
    w, h = img.size
    zoom = rng.uniform(1.05, 1.2)
    nw, nh = int(w / zoom), int(h / zoom)
    left = rng.integers(0, w - nw + 1)
    top = rng.integers(0, h - nh + 1)
    img = img.crop((left, top, left + nw, top + nh))
    img = img.resize((w, h), Image.Resampling.BICUBIC)

    # Shifting 10% secara acak
    dx = rng.integers(-0.1*w, 0.1*w)
    dy = rng.integers(-0.1*h, 0.1*h)
    img = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )

    return img

# Menyimpan gambar ke path yang ditentukan
def save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95) # Menjaga kualitas di 95%

# Membagi dataset hasil augmentasi ke folder train/validation/test
def process_split(split, input_dir, output_dir, augment_train, aug_count, rng):
    if not input_dir.exists():
        return

    # Hapus folder jika sudah ada
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Proses split sesuai nama kelas
    for cls in CLASS_NAMES:
        src_cls = input_dir / cls
        dst_cls = output_dir / cls
        # Jika folder kelas tidak ada, lewati
        if not src_cls.exists():
            continue
        # Iterasi semua file gambar di folder kelas
        for file in src_cls.rglob("*"):
            if file.suffix.lower() not in VALID_EXT:
                continue
            img = load_and_resize(file)

            # Simpan dengan nama baru .jpg
            out_path = dst_cls / f"{file.stem}.jpg"
            save(img, out_path)

            # Menjalankan seluruh augmentasi untuk folder train
            if augment_train and split == "train":
                for i in range(aug_count):
                    aug_img = augment(img, rng)
                    aug_path = dst_cls / f"{file.stem}_aug{i+1}.jpg"
                    save(aug_img, aug_path)

# Fungsi utama menjalankan proses preprocessing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-count", type=int, default=1)
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1]
    # Menentukan folder hasil split dan folder output untuk preprocessing
    split_root = base_dir / "data" / "splits"
    output_root = base_dir / "data" / "processed"
    # Menjaga konsistensi augmentasi dengan seed yang tetap
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
    # Hitung dataset hasil preprocessing
    def count_images(split_name: str) -> int:
        split_dir = output_root / split_name
        if not split_dir.exists():
            return 0
        return sum(
            1
            for p in split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXT
        )

    train_count = count_images("train")
    val_count = count_images("validation")
    test_count = count_images("test")
    total_count = train_count + val_count + test_count

    print("Preprocessing selesai")
    print(f"Total: {total_count}")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print(f"Saved under: {output_root}")
    print(f"Augmentasi: {args.augment}")
    print(f"Augment per image: {args.aug_count}")

if __name__ == "__main__":
    main()