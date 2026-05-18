"""
Labeled images ve label'larini train/val/test olarak boler.
Kullanim: python scripts/split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

SRC_IMAGES = Path("data/labeled/images")
SRC_LABELS = Path("data/labeled/labels")
DST_BASE   = Path("data/dataset")

SPLITS = {
    "train": TRAIN_RATIO,
    "val":   VAL_RATIO,
    "test":  TEST_RATIO,
}

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    image_paths = [
        p for p in SRC_IMAGES.iterdir()
        if p.suffix.lower() in SUPPORTED
    ]

    paired = []
    skipped = 0
    for img in image_paths:
        label = SRC_LABELS / (img.stem + ".txt")
        if label.exists():
            paired.append((img, label))
        else:
            print(f"  [SKIP] Label bulunamadi: {img.name}")
            skipped += 1

    if not paired:
        print("Hata: Hicbir eslesme bulunamadi. Gorsel ve label dosyalarini kontrol et.")
        return

    random.seed(42)
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train : n_train + n_val],
        "test":  paired[n_train + n_val :],
    }

    for split, items in splits.items():
        img_dst = DST_BASE / split / "images"
        lbl_dst = DST_BASE / split / "labels"
        if img_dst.exists():
            shutil.rmtree(img_dst)
        if lbl_dst.exists():
            shutil.rmtree(lbl_dst)
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img, lbl in items:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lbl, lbl_dst / lbl.name)

    print(f"\nToplam eslesme : {n}")
    print(f"  Train        : {len(splits['train'])}")
    print(f"  Val          : {len(splits['val'])}")
    print(f"  Test         : {len(splits['test'])}")
    if skipped:
        print(f"  Label eksik  : {skipped} (atlandi)")
    print("\nDataset hazir: data/dataset/")


if __name__ == "__main__":
    main()
