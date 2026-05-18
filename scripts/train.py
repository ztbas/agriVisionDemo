"""
YOLOv8n modelini egitir.
Kullanim: python scripts/train.py
"""

from pathlib import Path
from ultralytics import YOLO

CONFIG   = Path("config/dataset.yaml")
EPOCHS   = 100
IMG_SIZE = 640
BATCH    = 16        # GPU yoksa 8'e dusurebilirsin
WORKERS  = 4
PROJECT  = "runs"
NAME     = "agrivision_v4"


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(CONFIG),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        patience=20,        # 20 epoch iyilesme yoksa erken dur
        save=True,
        save_period=10,
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        degrees=10.0,       # hafif dönme arttirmasi
        fliplr=0.5,
        flipud=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        rect=False,
        amp=True,           # Mixed precision (varsa GPU hizlandirir)
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        plots=True,
        verbose=True,
    )

    best_model = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"\nEgitim tamamlandi.")
    print(f"En iyi model: {best_model}")
    print(f"Sonuclari gormek icin: runs/{NAME}/")


if __name__ == "__main__":
    main()
