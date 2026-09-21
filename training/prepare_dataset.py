import os
import shutil
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

BASE_DIR = Path(r"D:\Model Train")
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"

# Existing folders:
SRC_PITUITARY = DATASET_DIR / "pitiurity"
SRC_NO_TUMOR = DATASET_DIR / "without tumor"

# Target raw directories (standardized names)
TARGET_RAW_PITUITARY = RAW_DIR / "Pituitary_Tumor"
TARGET_RAW_NO_TUMOR = RAW_DIR / "No_Tumor"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def organize_raw_dataset():
    print("=" * 60)
    print("STEP 1.1: Organizing raw dataset into standard names...")
    print("=" * 60)
    
    TARGET_RAW_PITUITARY.mkdir(parents=True, exist_ok=True)
    TARGET_RAW_NO_TUMOR.mkdir(parents=True, exist_ok=True)
    
    # Move/Copy pitiurity -> raw/Pituitary_Tumor if not already done
    if SRC_PITUITARY.exists() and not any(TARGET_RAW_PITUITARY.iterdir()):
        print("Moving 'pitiurity' images to 'dataset/raw/Pituitary_Tumor'...")
        for item in SRC_PITUITARY.iterdir():
            if item.is_file() and item.suffix.lower() in VALID_EXTENSIONS:
                shutil.move(str(item), str(TARGET_RAW_PITUITARY / item.name))
        try:
            SRC_PITUITARY.rmdir()
        except OSError:
            pass

    # Move/Copy without tumor -> raw/No_Tumor if not already done
    if SRC_NO_TUMOR.exists() and not any(TARGET_RAW_NO_TUMOR.iterdir()):
        print("Moving 'without tumor' images to 'dataset/raw/No_Tumor'...")
        for item in SRC_NO_TUMOR.iterdir():
            if item.is_file() and item.suffix.lower() in VALID_EXTENSIONS:
                shutil.move(str(item), str(TARGET_RAW_NO_TUMOR / item.name))
        try:
            SRC_NO_TUMOR.rmdir()
        except OSError:
            pass

    pit_files = [f for f in TARGET_RAW_PITUITARY.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
    no_tumor_files = [f for f in TARGET_RAW_NO_TUMOR.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

    print(f"Total Pituitary Tumor images: {len(pit_files)}")
    print(f"Total No Tumor images:        {len(no_tumor_files)}")
    return pit_files, no_tumor_files

def split_and_copy(files, class_name, train_ratio=0.70, val_ratio=0.15):
    random.shuffle(files)
    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }
    
    for split_name, split_files in splits.items():
        dest_dir = DATASET_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files if re-running
        for existing in dest_dir.iterdir():
            if existing.is_file():
                existing.unlink()
                
        for file_path in split_files:
            shutil.copy2(str(file_path), str(dest_dir / file_path.name))
            
        print(f"  [{split_name.upper():5}] {class_name:15}: {len(split_files)} images copied to {dest_dir}")

def main():
    pit_files, no_tumor_files = organize_raw_dataset()
    
    if not pit_files or not no_tumor_files:
        print("Error: Could not find images in raw folders!")
        return
        
    print("\n" + "=" * 60)
    print("STEP 1.2: Splitting dataset into Train (70%), Val (15%), Test (15%)...")
    print("=" * 60)
    
    split_and_copy(pit_files, "Pituitary_Tumor")
    split_and_copy(no_tumor_files, "No_Tumor")
    
    print("\n" + "=" * 60)
    print(">>> STEP 1 COMPLETE! Dataset successfully split & ready. <<<")
    print("=" * 60)

if __name__ == "__main__":
    main()
