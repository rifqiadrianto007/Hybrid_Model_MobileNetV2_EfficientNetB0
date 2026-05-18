from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage

try:
    from torchvision import transforms
except Exception:
    transforms = None

try:
    import torch
except Exception:
    torch = None

IMG_SIZE = (224, 224) # Ukuran gambar yang diinginkan setelah resize
# Use high-quality resampling (LANCZOS) as requested
RESAMPLE_METHOD = Image.Resampling.LANCZOS
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# fungsi center crop
def simple_crop_center(img, crop_frac: float = 0.9) -> Image.Image:
    """Center crop a fraction of the shortest side, default 90%."""
    w, h = img.size
    crop = int(min(w, h) * crop_frac)
    left = (w - crop) // 2
    top = (h - crop) // 2
    return img.crop((left, top, left + crop, top + crop))


def random_crop(img, crop_frac_min: float = 0.8, crop_frac_max: float = 1.0) -> Image.Image:
    """Random crop a region with side in [crop_frac_min, crop_frac_max] of shortest side."""
    w, h = img.size
    short = min(w, h)
    frac = random.uniform(crop_frac_min, crop_frac_max)
    crop = int(short * frac)
    if w == short:
        left = 0 if w == crop else random.randint(0, w - crop)
        top = random.randint(0, h - crop)
    else:
        left = random.randint(0, w - crop)
        top = 0 if h == crop else random.randint(0, h - crop)
    return img.crop((left, top, left + crop, top + crop))

def load_and_resize(path: Path, split: str = "train") -> Image.Image:
    """Load image, apply crop depending on split, and resize with LANCZOS."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # For training keep some randomness (random crop), for validation/test use center crop
    if split == "train":
        img = random_crop(img, crop_frac_min=0.8, crop_frac_max=1.0)
    else:
        img = simple_crop_center(img, crop_frac=0.9)
    return img.resize(IMG_SIZE, RESAMPLE_METHOD)

class AddGaussianNoise:
    """Add light gaussian noise to tensor image."""

    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        if torch is None:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


def get_online_transforms(img_size=IMG_SIZE):
    """Build torchvision online augmentation transforms for training and eval.

    Train: random crop + affine (shift/zoom/shear/rotation) + flip + gaussian noise.
    Val/Test: center crop only.
    """
    if transforms is None:
        raise ImportError("torchvision is not available")

    train_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomCrop(img_size),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2),
            shear=15,
            fill=255,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.02, p=0.3),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def save(img, path) :
    path.parent.mkdir(parents = True, exist_ok = True)
    img.save(path, quality = 95) # simpan dengan kualotas 95%

def process_split(split, input_dir, output_dir) :
    if not input_dir.exists() :
        return

    if output_dir.exists() :
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = get_online_transforms()

    for cls in CLASS_NAMES :
        src = input_dir / cls
        dst = output_dir / cls

        if not src.exists() :
            continue

        for file in src.rglob("*") :
            if file.suffix.lower() not in VALID_EXT :
                continue

            img = load_and_resize(file, split=split)
             # pilih transform
            transform = train_transform if split == "train" else eval_transform

            # transform -> tensor
            tensor_img = transform(img)

            # tensor -> PIL
            pil_img = transforms.ToPILImage()(tensor_img)

            save(pil_img, dst / f"{file.stem}.jpg")
            # Offline augmentation is intentionally removed.
            # Use online augmentation during training via torchvision transforms.

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type = str, default = None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).resolve().parents[1]

    split_root = base_dir / "data" / "splits"
    output_root = base_dir / "data" / "processed"

    random.seed(SEED)
    np.random.seed(SEED)

    for split in ["train", "validation", "test"] :
        print(f"Processing {split}...")
        process_split(
            split = split,
            input_dir = split_root / split,
            output_dir = output_root / split
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
    print("Augmentasi offline : disabled")
    print("Gunakan augmentasi online saat training via torchvision.transforms")

if __name__ == "__main__" :
    main()