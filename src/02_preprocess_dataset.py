"""Resize split images into a processed dataset folder.

Expected input:
- data/splits/train/<class_name>/*.jpg
- data/splits/validation/<class_name>/*.jpg
- data/splits/test/<class_name>/*.jpg

Output:
- data/processed/train/<class_name>/*.jpg
- data/processed/validation/<class_name>/*.jpg
- data/processed/test/<class_name>/*.jpg

The script also writes CSV manifests with the new file paths.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_filename(source_path: Path) -> str:
    return f"{source_path.stem.replace(' ', '_')}{source_path.suffix.lower()}"


def resize_and_save(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)


def process_split(split_name: str, input_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    if not input_dir.exists():
        return pd.DataFrame(columns=["filepath", "label", "source"])

    for class_name in CLASS_NAMES:
        class_input_dir = input_dir / class_name
        if not class_input_dir.exists():
            continue

        class_output_dir = output_dir / split_name / class_name
        if class_output_dir.exists():
            shutil.rmtree(class_output_dir)
        class_output_dir.mkdir(parents=True, exist_ok=True)

        for src_path in class_input_dir.rglob("*"):
            if not src_path.is_file() or src_path.suffix.lower() not in VALID_EXT:
                continue
            dst_path = class_output_dir / safe_filename(src_path)
            resize_and_save(src_path, dst_path)
            rows.append(
                {
                    "filepath": str(dst_path),
                    "label": class_name,
                    "source": split_name,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize split dataset into processed dataset folders.")
    parser.add_argument("--base-dir", type=str, default=None, help="Project root directory. Default: script parent")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    split_root = base_dir / "data" / "splits"
    processed_root = base_dir / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    outputs = []
    for split_name in ["train", "validation", "test"]:
        input_dir = split_root / split_name
        df = process_split(split_name, input_dir, processed_root)
        if not df.empty:
            df.to_csv(processed_root / f"{split_name}.csv", index=False)
        outputs.append((split_name, len(df)))

    print("Preprocessing completed")
    for split_name, count in outputs:
        print(f"{split_name}: {count}")
    print(f"Saved under: {processed_root}")


if __name__ == "__main__":
    main()
