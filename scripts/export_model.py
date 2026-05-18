"""
Egitilmis modeli Raspberry Pi icin ONNX formatina aktar.
Kullanim: python scripts/export_model.py --model runs/agrivision_v1/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Kaynak .pt model yolu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--simplify", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Hata: Model bulunamadi: {model_path}")
        return

    print(f"Model yukleniyor: {model_path}")
    model = YOLO(str(model_path))

    print("ONNX'e aktariliyor...")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        dynamic=False,
        opset=12,
    )

    out_path = Path(str(model_path).replace(".pt", ".onnx"))
    print(f"\nAktarim tamamlandi: {out_path}")
    print("\nRaspberry Pi'ye kopyalamak icin:")
    print(f"  scp {out_path} pi@<PI_IP>:/home/pi/agrivision/models/")
    print(f"\nPi'de kullanmak icin:")
    print(f"  python detect_batch.py --model models/best.onnx --input gorseller/")


if __name__ == "__main__":
    main()
