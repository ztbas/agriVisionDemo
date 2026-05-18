"""
Bir klasordeki tum goruntuleri toplu olarak isler ve sonuclari kaydeder.
Kullanim:
  python scripts/detect_batch.py --input klasor/yolu --model models/best.pt
  python scripts/detect_batch.py --input klasor/yolu --model models/best.onnx
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = {
    0: "sugar_beet",
    1: "verbascum_pulverulentum",
    2: "marrubium_vulgare",
    3: "isatis_tinctoria",
    4: "artemisia_ludoviciana",
    5: "carduus_nutans",
}

CLASS_COLORS = {
    "sugar_beet":               (0,   200,  50),   # yeşil
    "verbascum_pulverulentum":  (0,   100, 255),   # turuncu
    "marrubium_vulgare":        (255,  50,  50),   # mavi
    "isatis_tinctoria":         (180,   0, 255),   # mor
    "artemisia_ludoviciana":    (0,   220, 220),   # sarı-yeşil
    "carduus_nutans":           (50,   50, 255),   # kırmızı
}

SUPPORTED   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF_THRESH = 0.4
IOU_THRESH  = 0.45


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,  help="Gorsel klasoru")
    parser.add_argument("--output", default="output", help="Sonuc klasoru")
    parser.add_argument("--model",  default="models/best.pt", help="Model yolu (.pt veya .onnx)")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH)
    parser.add_argument("--iou",    type=float, default=IOU_THRESH)
    parser.add_argument("--save-json", action="store_true", help="JSON rapor kaydet")
    return parser.parse_args()


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["class_name"]
        conf  = det["confidence"]
        color = CLASS_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    args = parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED
    )

    if not image_paths:
        print(f"Hata: '{input_dir}' icinde desteklenen gorsel bulunamadi.")
        return

    print(f"Model  : {args.model}")
    print(f"Gorsel : {len(image_paths)} adet")
    print(f"Cikti  : {output_dir}/\n")

    all_results = []
    t_start = time.time()

    for idx, img_path in enumerate(image_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{idx}/{len(image_paths)}] Okunamadi: {img_path.name}")
            continue

        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id":   cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, "unknown"),
                    "confidence": round(float(box.conf[0]), 4),
                    "box":        [round(float(v), 1) for v in box.xyxy[0]],
                })

        annotated = draw_boxes(image.copy(), detections)
        out_path  = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        counts = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

        summary = ", ".join(f"{k}:{v}" for k, v in counts.items()) if counts else "tespit yok"
        print(f"  [{idx}/{len(image_paths)}] {img_path.name:40s} -> {summary}")

        all_results.append({
            "file":       img_path.name,
            "detections": detections,
        })

    elapsed = time.time() - t_start
    print(f"\n{len(image_paths)} gorsel islendi — {elapsed:.1f}s "
          f"({elapsed/len(image_paths)*1000:.0f}ms/gorsel)")
    print(f"Annotated gorseller: {output_dir}/")

    if args.save_json:
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"JSON rapor        : {json_path}")


if __name__ == "__main__":
    main()
