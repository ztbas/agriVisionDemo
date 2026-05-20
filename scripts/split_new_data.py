import os
import random
import shutil
from pathlib import Path

def split_dataset():
    src_dir = Path(r"C:\Users\elisa\Downloads\AgriVision2.yolov8\train")
    src_images_dir = src_dir / "images"
    src_labels_dir = src_dir / "labels"

    dst_dir = Path(r"d:\Agrivision1\data\roboflow_yeni")

    # Create destination directories
    splits = ["train", "val", "test"]
    for split in splits:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get all image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    if src_images_dir.exists():
        for f in src_images_dir.iterdir():
            if f.suffix.lower() in valid_extensions:
                image_files.append(f)

    if not image_files:
        print("No image files found in the source directory.")
        return

    # Filter to ensure they have corresponding labels (optional but good practice)
    valid_data = []
    for img_path in image_files:
        label_path = src_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            valid_data.append((img_path, label_path))
        else:
            print(f"Warning: Label not found for {img_path.name}")

    # Shuffle the data
    random.seed(42) # For reproducibility
    random.shuffle(valid_data)

    total = len(valid_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_data = valid_data[:train_end]
    val_data = valid_data[train_end:val_end]
    test_data = valid_data[val_end:]

    print(f"Total: {total} files")
    print(f"Train: {len(train_data)} files")
    print(f"Val: {len(val_data)} files")
    print(f"Test: {len(test_data)} files")

    def copy_data(data_list, split_name):
        for img_path, label_path in data_list:
            shutil.copy2(img_path, dst_dir / split_name / "images" / img_path.name)
            shutil.copy2(label_path, dst_dir / split_name / "labels" / label_path.name)

    print("Copying train data...")
    copy_data(train_data, "train")
    print("Copying val data...")
    copy_data(val_data, "val")
    print("Copying test data...")
    copy_data(test_data, "test")

    print("Dataset splitting and copying completed successfully!")

if __name__ == "__main__":
    split_dataset()
