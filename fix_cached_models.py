"""
Script to fix cached models by adding num_labels to them.

This script loads each cached model, sets num_labels, and re-saves it.
"""
import os
import sys
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def fix_cached_model(cache_path: str):
    """Load a cached model, add num_labels, and re-save it."""
    print(f"Processing: {cache_path}")
    
    try:
        # Load the model
        with open(cache_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if num_labels already exists
        if 'num_labels' in model_data:
            print(f"  ✅ num_labels already present: {model_data['num_labels']}")
            return True
        
        # Calculate num_labels from classes_
        if 'classes_' in model_data and model_data['classes_'] is not None:
            num_labels = len(model_data['classes_'])
            model_data['num_labels'] = num_labels
            print(f"  📝 Adding num_labels: {num_labels}")
            
            # Re-save the model
            with open(cache_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"  ✅ Model updated and saved")
            return True
        else:
            print(f"  ❌ No classes_ found in model data")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("="*80)
    print("FIX CACHED MODELS - ADD num_labels")
    print("="*80)
    
    cache_dir = "cache"
    
    # Find all cached RoBERTa models
    cached_models = []
    for filename in sorted(os.listdir(cache_dir)):
        if filename.startswith("fusion_roberta_model_"):
            filepath = os.path.join(cache_dir, filename)
            cached_models.append(filepath)
    
    print(f"\nFound {len(cached_models)} cached models")
    print()
    
    success_count = 0
    for cache_path in cached_models:
        if fix_cached_model(cache_path):
            success_count += 1
        print()
    
    print("="*80)
    print(f"SUMMARY: {success_count}/{len(cached_models)} models fixed successfully")
    print("="*80)


if __name__ == "__main__":
    main()
