#!/usr/bin/env python3
"""Test script to debug cache saving."""

import json
from pathlib import Path

# Check current cache state
cache_file = Path("cache/val_b6e3cb0f.json")

if cache_file.exists():
    with open(cache_file, 'r') as f:
        data = json.load(f)
    
    print(f"Cache file: {cache_file}")
    print(f"Predictions count: {data['metadata']['predictions_count']}")
    print(f"Last updated: {data['metadata']['last_updated']}")
    print(f"Actual predictions: {len(data['predictions'])}")
    
    # Show first few text hashes
    import hashlib
    print("\nFirst 5 text hashes in cache:")
    for i, entry in enumerate(data['predictions'][:5]):
        text = entry.get('text', '')
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        print(f"  {i}: {text_hash} - {text[:50]}...")
else:
    print(f"Cache file not found: {cache_file}")
