#!/usr/bin/env python3
"""Quick cache detection test."""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import after sys.path modification
from textclassify.llm.base import BaseLLMClassifier

# Test DataFrame with same data as validation
df_val = pd.read_csv("data/reuters/validationrefined.csv")
print(f"Loaded {len(df_val)} rows")

# Create minimal classifier instance
class TestClassifier(BaseLLMClassifier):
    def __init__(self):
        self.text_column = 'text'
        self.cache_dir = 'cache'
        self.verbose = True

clf = TestClassifier()

print("\n" + "="*80)
print("TESTING has_test_cache_for_dataset")
print("="*80)
result = clf.has_test_cache_for_dataset(df_val)
print(f"\n{'='*80}")
print(f"RESULT: {result}")
print("="*80)
