"""
Labeled veriyi Roboflow'a yukler (YOLO format).
Kullanim: python3 scripts/upload_roboflow.py
"""

import os
from pathlib import Path
from roboflow import Roboflow

API_KEY      = "jsOQMzm5p2znD6KD0xYq"
PROJECT_NAME = "AgriVision1.1"

IMAGES_DIR = Path("data/labeled/images")
LABELS_DIR = Path("data/labeled/labels")

CLASSES = [
    "sugar_beet",
    "verbascum_pulverulentum",
    "marrubium_vulgare",
    "isatis_tinctoria",
    "artemisia_ludoviciana",
    "carduus_nutans",
]

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace()
    print(f"Workspace: {workspace.url}")

    project = workspace.project("agrivision1-7gphr")
    print(f"Proje: {project.name}")

    image_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in SUPPORTED]
    total = len(image_paths)
    uploaded = 0
    skipped = 0

    print(f"\nToplam gorsel: {total}")
    print("Yukleniyor...\n")

    for i, img_path in enumerate(image_paths, 1):
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"  [{i}/{total}] SKIP (label yok): {img_path.name}")
            skipped += 1
            continue

        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(label_path),
                annotation_labelmap=CLASSES,
                split="train",
                num_retry_uploads=3,
                batch_name="labeled_batch",
            )
            uploaded += 1
            if i % 50 == 0:
                print(f"  [{i}/{total}] {uploaded} yuklendi...")
        except Exception as e:
            print(f"  [{i}/{total}] HATA ({img_path.name}): {e}")

    print(f"\nTamamlandi: {uploaded} yuklendi, {skipped} atlandı (label eksik)")


if __name__ == "__main__":
    main()
