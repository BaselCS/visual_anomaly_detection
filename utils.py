"""
Utility functions for anomaly detection project
"""
import os
from PIL import Image
from collections import defaultdict

def validate_data_structure(data_root="data"):
    """
    Validate the data folder structure and report any issues.
    
    Expected structure:
    data/
        {category}/
            train/good/
            test/{defect_type}/
            ground_truth/{defect_type}/
    """
    print("Validating data structure...\n")
    
    if not os.path.exists(data_root):
        print(f"❌ ERROR: Data directory '{data_root}' not found!")
        return False
    
    categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    if not categories:
        print(f"❌ ERROR: No categories found in '{data_root}'")
        return False
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}\n")
    
    all_valid = True
    stats = defaultdict(lambda: {"train": 0, "test_good": 0, "test_defect": 0})
    
    for category in categories:
        cat_path = os.path.join(data_root, category)
        print(f"📦 {category.upper()}:")
        
        # Check training data
        train_path = os.path.join(cat_path, "train", "good")
        if not os.path.exists(train_path):
            print(f"  ❌ Missing: train/good/")
            all_valid = False
        else:
            train_images = [f for f in os.listdir(train_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            stats[category]["train"] = len(train_images)
            print(f"  ✓ Train (good): {len(train_images)} images")
            
            # Validate image files
            corrupted = []
            for img_file in train_images[:5]:  # Check first 5
                try:
                    Image.open(os.path.join(train_path, img_file)).verify()
                except Exception as e:
                    corrupted.append(img_file)
            
            if corrupted:
                print(f"  ⚠ Warning: {len(corrupted)} corrupted images detected")
        
        # Check test data
        test_path = os.path.join(cat_path, "test")
        if not os.path.exists(test_path):
            print(f"  ❌ Missing: test/")
            all_valid = False
        else:
            test_types = [d for d in os.listdir(test_path) 
                         if os.path.isdir(os.path.join(test_path, d))]
            
            for test_type in test_types:
                test_type_path = os.path.join(test_path, test_type)
                test_images = [f for f in os.listdir(test_type_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if test_type == "good":
                    stats[category]["test_good"] = len(test_images)
                else:
                    stats[category]["test_defect"] += len(test_images)
                
                print(f"  ✓ Test ({test_type}): {len(test_images)} images")
        
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY:")
    print(f"{'Category':<15} {'Train':<10} {'Test Good':<12} {'Test Defect':<12}")
    print("-"*60)
    for cat, data in stats.items():
        print(f"{cat:<15} {data['train']:<10} {data['test_good']:<12} {data['test_defect']:<12}")
    print("="*60)
    
    if all_valid:
        print("✅ Data structure is valid!")
    else:
        print("⚠ Some issues found. Please fix them before training.")
    
    return all_valid


def get_dataset_statistics(data_root="data"):
    """Get detailed statistics about the dataset"""
    stats = {
        "categories": {},
        "total_train": 0,
        "total_test": 0
    }
    
    categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for category in categories:
        cat_path = os.path.join(data_root, category)
        cat_stats = {"train": {}, "test": {}}
        
        # Train data
        train_path = os.path.join(cat_path, "train", "good")
        if os.path.exists(train_path):
            train_count = len([f for f in os.listdir(train_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            cat_stats["train"]["good"] = train_count
            stats["total_train"] += train_count
        
        # Test data
        test_path = os.path.join(cat_path, "test")
        if os.path.exists(test_path):
            for test_type in os.listdir(test_path):
                test_type_path = os.path.join(test_path, test_type)
                if os.path.isdir(test_type_path):
                    test_count = len([f for f in os.listdir(test_type_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    cat_stats["test"][test_type] = test_count
                    stats["total_test"] += test_count
        
        stats["categories"][category] = cat_stats
    
    return stats


if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_data_structure()
