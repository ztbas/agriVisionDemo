"""
Offline data augmentation for YOLO dataset.
Reads from multiple train directories, writes augmented copies back.
Usage: python scripts/augment_offline.py
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from collections import Counter
import albumentations as A
from tqdm import tqdm

TRAIN_DIRS = [
    Path("data/dataset/train"),
    Path("data/roboflow_yeni/train")
]

# Her görsel için kaç kopyа üretilsin
AUGMENT_TIMES = 3

# Tekrarlanabilirlik
random.seed(42)
np.random.seed(42)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.PadIfNeeded(min_height=640, min_width=640,
                  border_mode=cv2.BORDER_REFLECT_101, p=1.0),
    A.RandomCrop(height=640, width=640, p=1.0),
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3,
))


def read_label(label_path):
    boxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            classes.append(int(float(parts[0])))
            boxes.append([float(x) for x in parts[1:5]])
    return boxes, classes


def write_label(label_path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


def augment_image(img_path, label_path, aug_index, images_dir, labels_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, classes = read_label(label_path)

    # Boş label (background) → sadece görsel dönüşümü, bbox yok
    if not boxes:
        aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ])
        result = aug_transform(image=img)
        aug_img = result["image"]
        aug_boxes, aug_classes = [], []
    else:
        try:
            result = transform(image=img, bboxes=boxes, class_labels=classes)
        except Exception:
            return False
        aug_img = result["image"]
        aug_boxes = list(result["bboxes"])
        aug_classes = list(result["class_labels"])

        # Tüm bbox'lar filtrelendiyse kaydetme
        if not aug_boxes and boxes:
            return False

    # Kaydet
    stem = img_path.stem
    suffix = img_path.suffix
    new_name = f"{stem}_aug{aug_index}"

    out_img = images_dir / f"{new_name}{suffix}"
    out_lbl = labels_dir / f"{new_name}.txt"

    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_img), aug_img_bgr)
    write_label(out_lbl, aug_boxes, aug_classes)
    return True


def class_counts(label_dirs):
    counts = Counter()
    for label_dir in label_dirs:
        for f in label_dir.glob("*.txt"):
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        counts[int(float(line.split()[0]))] += 1
    return counts


def main():
    names = {0:"sugar_beet", 1:"verbascum", 2:"marrubium",
             3:"isatis", 4:"artemisia", 5:"carduus"}

    label_dirs = [d / "labels" for d in TRAIN_DIRS]
    
    print("Augmentation öncesi sınıf dağılımı (Tüm Verisetleri):")
    before = class_counts(label_dirs)
    for k, v in sorted(before.items()):
        print(f"  {names.get(k, k)}: {v}")

    for train_dir in TRAIN_DIRS:
        print(f"\n--- İşleniyor: {train_dir} ---")
        images_dir = train_dir / "images"
        labels_dir = train_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"Uyarı: {train_dir} bulunamadı, atlanıyor...")
            continue

        exts = ["*.jpeg", "*.JPEG", "*.jpg", "*.JPG", "*.png", "*.PNG", "*.webp"]
        image_files = []
        for ext in exts:
            image_files += sorted(images_dir.glob(ext))

        # Sadece orijinal görseller (aug ile başlamayanlar)
        originals = [p for p in image_files if "_aug" not in p.stem]
        print(f"{len(originals)} orijinal görsel bulundu, her biri {AUGMENT_TIMES}x çoğaltılacak...")

        ok, skip = 0, 0
        for img_path in tqdm(originals, desc=f"Augmenting {train_dir.name}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            for i in range(AUGMENT_TIMES):
                success = augment_image(img_path, label_path, i + 1, images_dir, labels_dir)
                if success:
                    ok += 1
                else:
                    skip += 1

        print(f"Oluşturulan: {ok} | Atlanan: {skip}")

    print("\n==================================")
    print("Augmentation sonrası sınıf dağılımı (Tüm Verisetleri):")
    after = class_counts(label_dirs)
    for k, v in sorted(after.items()):
        print(f"  {names.get(k, k)}: {v}  (+{v - before.get(k,0)})")

    total_imgs = 0
    for train_dir in TRAIN_DIRS:
        total_imgs += len(list((train_dir / "images").glob("*.*")))
    print(f"\nToplam train görsel sayısı: {total_imgs}")


if __name__ == "__main__":
    main()
