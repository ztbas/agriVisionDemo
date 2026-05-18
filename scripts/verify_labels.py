"""
Label dosyalarini dogrular: format hatasi, gecersiz koordinat, bosaltilmis label vs.
Kullanim: python scripts/verify_labels.py
"""

from pathlib import Path

LABEL_DIR = Path("data/labeled/labels")
IMAGE_DIR = Path("data/labeled/images")
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 6


def check_label_file(label_path: Path) -> list[str]:
    errors = []
    lines = label_path.read_text().strip().splitlines()

    if not lines:
        errors.append("Bos label dosyasi (background gorsel ise normal)")
        return errors

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"  Satir {i}: 5 deger olmali, {len(parts)} var -> '{line}'")
            continue
        try:
            cls_id = int(parts[0])
            vals   = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"  Satir {i}: sayi olmayan deger -> '{line}'")
            continue

        if cls_id < 0 or cls_id >= NUM_CLASSES:
            errors.append(f"  Satir {i}: gecersiz class_id={cls_id} (0-{NUM_CLASSES-1} olmali)")
        for v in vals:
            if not (0.0 <= v <= 1.0):
                errors.append(f"  Satir {i}: koordinat aralik disi ({v}) -> '{line}'")
                break

    return errors


def main():
    label_files = sorted(p for p in LABEL_DIR.glob("*.txt") if p.name != "classes.txt")

    if not label_files:
        print(f"Hata: '{LABEL_DIR}' icinde .txt bulunamadi.")
        return

    images_without_labels = []
    for img in IMAGE_DIR.iterdir():
        if img.suffix.lower() in SUPPORTED:
            if not (LABEL_DIR / (img.stem + ".txt")).exists():
                images_without_labels.append(img.name)

    total_errors  = 0
    error_files   = 0
    empty_files   = 0

    for lbl in label_files:
        errs = check_label_file(lbl)
        if errs:
            if errs[0].startswith("Bos"):
                empty_files += 1
            else:
                error_files += 1
                total_errors += len(errs)
                print(f"HATA [{lbl.name}]:")
                for e in errs:
                    print(f"  {e}")

    print(f"\n--- Ozet ---")
    print(f"Toplam label dosyasi : {len(label_files)}")
    print(f"Bos dosya (bg gorsel): {empty_files}")
    print(f"Hatali dosya         : {error_files}")
    print(f"Toplam hata          : {total_errors}")
    if images_without_labels:
        print(f"Label eksik gorsel   : {len(images_without_labels)}")
        for f in images_without_labels[:10]:
            print(f"  {f}")
        if len(images_without_labels) > 10:
            print(f"  ... ve {len(images_without_labels)-10} tane daha")
    else:
        print("Label eksik gorsel   : yok")

    if error_files == 0 and not images_without_labels:
        print("\nTum labellar gecerli!")


if __name__ == "__main__":
    main()
