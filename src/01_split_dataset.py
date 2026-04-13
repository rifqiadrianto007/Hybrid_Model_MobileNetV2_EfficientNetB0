"""Split the raw chili leaf dataset into train/validation/test folders.

Expected raw structure:
- data/raw/primer/<class_name>/*.jpg
- data/raw/sekunder/<class_name>/*.jpg

This script copies files into:
- data/splits/train/<class_name>/
- data/splits/validation/<class_name>/
- data/splits/test/<class_name>/

It also writes CSV manifests for each split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ["leaf curl", "leaf spot", "yellowish", "healthy leaf"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ALIASES = {
    "leaf curl": ["leaf curl", "leaf_curl", "leaf-curl", "curl"],
    "leaf spot": ["leaf spot", "leaf_spot", "leaf-spot", "spot"],
    "yellowish": ["yellowish", "yellow", "kuning"],
    "healthy leaf": ["healthy leaf", "healthy_leaf", "healthy-leaf", "healthy", "sehat"],
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def infer_label_from_path(path_obj: Path) -> str | None:
    parts_norm = [normalize_text(part) for part in path_obj.parts]
    for class_name in CLASS_NAMES:
        aliases = CLASS_ALIASES.get(class_name, [class_name])
        aliases_norm = [normalize_text(alias) for alias in aliases]
        if any(alias in parts_norm for alias in aliases_norm):
            return class_name
    return None


def collect_images(root_dir: Path, source_name: str) -> pd.DataFrame:
    rows = []
    if not root_dir.exists():
        return pd.DataFrame(columns=["filepath", "label", "source"])

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in VALID_EXT:
            continue

        label = infer_label_from_path(file_path)
        if label is None:
            continue

        rows.append(
            {
                "filepath": str(file_path),
                "label": label,
                "source": source_name,
            }
        )

    return pd.DataFrame(rows)


def safe_filename(source: str, class_name: str, original_path: Path) -> str:
    stem = original_path.stem.replace(" ", "_")
    stem = stem.replace("/", "_").replace("\\", "_")
    return f"{source}__{class_name.replace(' ', '_')}__{stem}{original_path.suffix.lower()}"


def copy_split(split_df: pd.DataFrame, split_dir: Path) -> None:
    for _, row in split_df.iterrows():
        src_path = Path(row["filepath"])
        class_dir = split_dir / row["label"]
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = class_dir / safe_filename(row["source"], row["label"], src_path)
        shutil.copy2(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation/test split folders.")
    parser.add_argument("--base-dir", type=str, default=None, help="Project root directory. Default: script parent")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[1]
    raw_primer = base_dir / "data" / "raw" / "primer"
    raw_sekunder = base_dir / "data" / "raw" / "sekunder"
    split_root = base_dir / "data" / "splits"
    split_root.mkdir(parents=True, exist_ok=True)

    primer_df = collect_images(raw_primer, "primer")
    sekunder_df = collect_images(raw_sekunder, "sekunder")
    data_df = pd.concat([primer_df, sekunder_df], ignore_index=True)

    if data_df.empty:
        raise ValueError("No images found. Check the raw dataset folders.")

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        data_df,
        test_size=(1.0 - args.train_ratio),
        random_state=SEED,
        stratify=data_df["label"],
    )

    val_ratio_adjusted = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_adjusted),
        random_state=SEED,
        stratify=temp_df["label"],
    )

    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split_dir = split_root / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        copy_split(split_df, split_dir)
        split_df.to_csv(split_root / f"{split_name}.csv", index=False)

    data_df.to_csv(split_root / "all_data.csv", index=False)

    print("Split completed")
    print(f"Total: {len(data_df)}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Saved under: {split_root}")


if __name__ == "__main__":
    main()
