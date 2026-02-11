"""
Script for moving all train and test images to flat directories.
- Train images go to `data/train/` with format: {category}_{number}.png
- Test images go to `data/test/` with format: {category}_{test_type}_{number}.png
"""

import os
import shutil
from pathlib import Path

def reorganize_training_data():
    """Move all training images to data/train with category prefixes."""
    
    new_train_dir = Path("data/train")
    new_train_dir.mkdir(exist_ok=True)
    
    # Find all category directories (exclude 'train', 'test', 'zip')
    data_dir = Path("data")
    categories = [d for d in data_dir.iterdir() 
                  if d.is_dir() and d.name not in ["train", "test", "zip"]]
    
    total_moved = 0
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        old_train_dir = category_dir / "train"
        
        if not old_train_dir.exists():
            print(f"Skipping {category_name}: no train directory found")
            continue
        
        # Find all PNG files recursively in the old train directory    
        image_files = list(old_train_dir.rglob("*.png"))
        print(f"\nProcessing {category_name} train: {len(image_files)} images")
        
        
        for idx, image_path in enumerate(sorted(image_files), start=1):
            new_filename = f"{category_name}_{idx:03d}.png"
            new_path = new_train_dir / new_filename
            
            shutil.copy2(image_path, new_path)
            total_moved += 1
        
        print(f"  Moved {len(image_files)} images from {category_name}")
    
    print(f"\n{'='*50}")
    print(f"Total train images moved: {total_moved}")
    print(f"New location: {new_train_dir.absolute()}")
    print(f"{'='*50}")
    
    return categories

def reorganize_test_data(categories):
    """Move all test images to data/test with category and test type prefixes."""
    
    new_test_dir = Path("data/test")
    new_test_dir.mkdir(exist_ok=True)
    
    total_moved = 0
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        old_test_dir = category_dir / "test"
        
        if not old_test_dir.exists():
            print(f"\nSkipping {category_name}: no test directory found")
            continue
        
        # Find all test type subdirectories
        test_types = [d for d in old_test_dir.iterdir() if d.is_dir()]
        
        for test_type_dir in sorted(test_types):
            test_type_name = test_type_dir.name
            
            # Find all PNG files in this test type directory
            image_files = list(test_type_dir.glob("*.png"))
            
            print(f"\nProcessing {category_name}/{test_type_name}: {len(image_files)} images")
            
            # Move and rename each image
            for idx, image_path in enumerate(sorted(image_files), start=1):
                # Create new filename with category and test type prefix
                new_filename = f"{category_name}_{test_type_name}_{idx:04d}.png"
                new_path = new_test_dir / new_filename
                
                # Move the file
                shutil.copy2(image_path, new_path)
                total_moved += 1
            
            print(f"  Moved {len(image_files)} images from {category_name}/{test_type_name}")
    
    print(f"\n{'='*50}")
    print(f"Total test images moved: {total_moved}")
    print(f"New location: {new_test_dir.absolute()}")
    print(f"{'='*50}")

def cleanup_old_directories(categories):
    """Delete old train and test subdirectories."""
    print("\nImages have been copied to data/train and data/test.")
    response = input("Do you want to delete the old train and test subdirectories? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        for category_dir in sorted(categories):
            # Delete old train directory
            old_train_dir = category_dir / "train"
            if old_train_dir.exists():
                shutil.rmtree(old_train_dir)
                print(f"Deleted {old_train_dir}")
            
            # Delete old test directory
            old_test_dir = category_dir / "test"
            if old_test_dir.exists():
                shutil.rmtree(old_test_dir)
                print(f"Deleted {old_test_dir}")
        
        print("\nOld train and test directories have been removed.")
    else:
        print("\nOld directories kept. You can delete them manually if needed.")

if __name__ == "__main__":
    # Reorganize training data
    categories = reorganize_training_data()
    
    # Reorganize test data
    reorganize_test_data(categories)
    
    # Cleanup old directories
    cleanup_old_directories(categories)
