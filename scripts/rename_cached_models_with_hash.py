#!/usr/bin/env python3
"""
Rename cached model files to include dataset hash instead of percentage.

This script:
1. Loads the full training dataset
2. For each percentage (10%, 20%, ..., 100%), creates the stratified subset
3. Calculates hash of the subset
4. Renames cached model files from 'fusion_roberta_model_{pct}pct' to 'fusion_roberta_model_{hash}'
"""

import os
import sys
import hashlib
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.evaluation.eval_goemotions import create_stratified_subset


def calculate_dataset_hash(df: pd.DataFrame) -> str:
    """Calculate hash of a dataset based on its content.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        8-character hex hash
    """
    # Create a stable string representation
    # Use text + all label columns, sorted by index
    df_sorted = df.sort_index()
    
    # Concatenate text and labels into a single string
    content_str = ""
    text_col = 'text'
    label_cols = [col for col in df.columns if col != text_col and col != 'Unnamed: 0']
    
    for idx, row in df_sorted.iterrows():
        content_str += str(row[text_col])
        for label_col in sorted(label_cols):
            content_str += str(row[label_col])
    
    # Calculate SHA256 hash
    hash_obj = hashlib.sha256(content_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def main():
    """Main function to rename cached model files."""
    
    # Paths
    cache_dir = project_root / "cache"
    data_dir = project_root / "data" / "goemotions"
    
    # Load full training dataset
    print("Loading full training dataset...")
    train_file = data_dir / "goemotions_all_train_balanced.csv"
    
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return 1
    
    df_train = pd.read_csv(train_file)
    print(f"  Loaded {len(df_train)} training samples")
    
    # Get label columns (all except 'text' and index)
    label_columns = [col for col in df_train.columns if col not in ['text', 'Unnamed: 0']]
    print(f"  Label columns: {len(label_columns)} labels")
    
    # Percentages to process
    percentages = [i * 0.1 for i in range(1, 11)]  # 10% to 100%
    
    # Process each percentage
    rename_mapping = {}
    
    print("\nCalculating hashes for each percentage...")
    for pct in percentages:
        pct_int = int(pct * 100)
        
        # Create stratified subset
        print(f"\n{pct_int}%:")
        df_subset = create_stratified_subset(df_train, pct, label_columns, random_state=42)
        print(f"  Subset size: {len(df_subset)} samples")
        
        # Calculate hash
        dataset_hash = calculate_dataset_hash(df_subset)
        print(f"  Hash: {dataset_hash}")
        
        # Old and new filenames
        old_name = f"fusion_roberta_model_{pct_int}pct"
        new_name = f"fusion_roberta_model_{dataset_hash}"
        
        old_path = cache_dir / old_name
        new_path = cache_dir / new_name
        
        if old_path.exists():
            rename_mapping[old_name] = new_name
            print(f"  Will rename: {old_name} -> {new_name}")
        else:
            print(f"  Warning: File not found: {old_name}")
    
    # Ask for confirmation
    if not rename_mapping:
        print("\nNo files to rename!")
        return 0
    
    print(f"\n\nReady to rename {len(rename_mapping)} files.")
    response = input("Proceed with renaming? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return 0
    
    # Perform renaming
    print("\nRenaming files...")
    for old_name, new_name in rename_mapping.items():
        old_path = cache_dir / old_name
        new_path = cache_dir / new_name
        
        try:
            old_path.rename(new_path)
            print(f"  ✓ {old_name} -> {new_name}")
        except Exception as e:
            print(f"  ✗ Failed to rename {old_name}: {e}")
    
    print("\n✅ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
