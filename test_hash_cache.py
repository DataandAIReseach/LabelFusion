#!/usr/bin/env python3
"""Test script to verify hash-based cache matching works correctly."""

import json
import hashlib
import pandas as pd
from pathlib import Path

def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text for matching."""
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

def test_hash_matching():
    """Test the hash-based cache matching logic."""
    
    # Load the validation dataset
    val_file = "data/reuters/val.csv"
    if not Path(val_file).exists():
        print(f"❌ Validation file not found: {val_file}")
        return
    
    val_df = pd.read_csv(val_file)
    text_column = 'text'
    
    print(f"📊 Loaded validation dataset: {len(val_df)} samples")
    
    # Step 1: Compute hash for all texts in dataset
    dataset_hash_map = {}
    for pos, (_, row) in enumerate(val_df.iterrows()):
        text = row.get(text_column, "")
        text_hash = compute_text_hash(text)
        dataset_hash_map[text_hash] = (pos, text[:50])  # Store first 50 chars for display
    
    print(f"✅ Dataset: {len(val_df)} texts with {len(dataset_hash_map)} unique hashes")
    
    # Step 2: Load cache file
    cache_file = "cache/val_b6e3cb0f.json"
    if not Path(cache_file).exists():
        print(f"❌ Cache file not found: {cache_file}")
        return
    
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    cached_predictions = cache_data.get('predictions', [])
    print(f"📦 Cache file has {len(cached_predictions)} predictions")
    
    # Step 3: Match cached predictions by hash
    cached_hashes = set()
    matched_predictions = {}
    
    for cache_entry in cached_predictions:
        cached_text = cache_entry.get('text', '')
        cached_pred = cache_entry.get('prediction', None)
        
        if cached_pred and isinstance(cached_pred, list):
            cached_hash = compute_text_hash(cached_text)
            cached_hashes.add(cached_hash)
            
            # If this hash exists in our dataset, record the match
            if cached_hash in dataset_hash_map:
                pos, text_preview = dataset_hash_map[cached_hash]
                matched_predictions[pos] = cached_pred
                print(f"  ✓ Match: position {pos}, hash {cached_hash[:8]}...")
    
    print(f"\n✅ Matched {len(matched_predictions)} cached predictions to dataset")
    
    # Step 4: Identify uncached positions
    uncached_positions = []
    for text_hash, (pos, text_preview) in dataset_hash_map.items():
        if text_hash not in cached_hashes:
            uncached_positions.append(pos)
    
    print(f"🔄 Need to process {len(uncached_positions)} uncached texts ({len(uncached_positions)/len(val_df)*100:.1f}%)")
    print(f"   First 5 uncached positions: {uncached_positions[:5]}")
    
    # Step 5: Verify batch calculation
    batch_size = 32
    total_batches = (len(uncached_positions) + batch_size - 1) // batch_size
    print(f"📊 Will process in {total_batches} batches of size {batch_size}")
    
    # Verify this matches expected behavior
    expected_remaining = len(val_df) - len(matched_predictions)
    if len(uncached_positions) == expected_remaining:
        print(f"✅ Hash matching logic correct: {len(uncached_positions)} uncached == {expected_remaining} remaining")
    else:
        print(f"❌ Mismatch: {len(uncached_positions)} uncached != {expected_remaining} remaining")

if __name__ == "__main__":
    test_hash_matching()
