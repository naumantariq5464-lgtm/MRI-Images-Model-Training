import os
import shutil
import random
from pathlib import Path

# Set seed for reproducible split
random.seed(42)

BASE_DIR = Path(r"D:\Model Train")
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"

SRC_GLIOMA_TRAIN = DATASET_DIR / "glioma for traning"
SRC_GLIOMA_TEST = DATASET_DIR / "glioma for testing"

TARGET_RAW_GLIOMA = RAW_DIR / "Glioma_Tumor"
TARGET_TRAIN_GLIOMA = DATASET_DIR / "train" / "Glioma_Tumor"
TARGET_VAL_GLIOMA = DATASET_DIR / "val" / "Glioma_Tumor"
TARGET_TEST_GLIOMA = DATASET_DIR / "test" / "Glioma_Tumor"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def organize_glioma():
    print("=" * 60)
    print("Organizing Glioma Dataset...")
    print("=" * 60)

    # 1. Create target directories
    TARGET_RAW_GLIOMA.mkdir(parents=True, exist_ok=True)
    TARGET_TRAIN_GLIOMA.mkdir(parents=True, exist_ok=True)
    TARGET_VAL_GLIOMA.mkdir(parents=True, exist_ok=True)
    TARGET_TEST_GLIOMA.mkdir(parents=True, exist_ok=True)

    # 2. Get training and testing files
    train_files = [f for f in SRC_GLIOMA_TRAIN.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]
    test_files = [f for f in SRC_GLIOMA_TEST.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]

    print(f"Found {len(train_files)} files in '{SRC_GLIOMA_TRAIN.name}'")
    print(f"Found {len(test_files)} files in '{SRC_GLIOMA_TEST.name}'")

    if not train_files or not test_files:
        print("ERROR: Source files not found or empty!")
        return

    # 3. Shuffle and split training into Train (85%) and Val (15%)
    random.shuffle(train_files)
    split_idx = int(len(train_files) * 0.85)
    train_split = train_files[:split_idx]
    val_split = train_files[split_idx:]

    print(f"Glioma Train count: {len(train_split)}")
    print(f"Glioma Val count:   {len(val_split)}")
    print(f"Glioma Test count:  {len(test_files)}")

    # 4. Copy to train
    for f in train_split:
        shutil.copy2(f, TARGET_TRAIN_GLIOMA / f.name)
        shutil.copy2(f, TARGET_RAW_GLIOMA / f.name)

    # 5. Copy to val
    for f in val_split:
        shutil.copy2(f, TARGET_VAL_GLIOMA / f.name)
        shutil.copy2(f, TARGET_RAW_GLIOMA / f.name)

    # 6. Copy to test
    for f in test_files:
        shutil.copy2(f, TARGET_TEST_GLIOMA / f.name)
        shutil.copy2(f, TARGET_RAW_GLIOMA / f.name)

    print("\nVerifying final counts in target directories:")
    print(f"  dataset/raw/Glioma_Tumor:   {len(list(TARGET_RAW_GLIOMA.glob('*')))} files")
    print(f"  dataset/train/Glioma_Tumor: {len(list(TARGET_TRAIN_GLIOMA.glob('*')))} files")
    print(f"  dataset/val/Glioma_Tumor:   {len(list(TARGET_VAL_GLIOMA.glob('*')))} files")
    print(f"  dataset/test/Glioma_Tumor:  {len(list(TARGET_TEST_GLIOMA.glob('*')))} files")
    print("=" * 60)
    print("Glioma dataset successfully organized!")

if __name__ == "__main__":
    organize_glioma()
