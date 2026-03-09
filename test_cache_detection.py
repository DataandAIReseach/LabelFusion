#!/usr/bin/env python3
"""Test cache detection logic."""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from textclassify.llm.openai_classifier import OpenAIClassifier

# Load validation data
data_dir = "data/reuters"
df_val = pd.read_csv(f"{data_dir}/validationrefined.csv")

print(f"Loaded {len(df_val)} validation samples")
print(f"First few rows:")
print(df_val.head(2))

# Create a simple config
class SimpleConfig:
    def __init__(self):
        self.parameters = {
            'model': 'gpt-5-nano',
            'temperature': 0.0,
            'max_tokens': 50
        }

# Create LLM model
label_columns = [col for col in df_val.columns if col not in ['text', 'doc_id', 'split']]
print(f"\nLabel columns: {label_columns}")

llm_model = OpenAIClassifier(
    config=SimpleConfig(),
    text_column='text',
    label_columns=label_columns,
    multi_label=True,
    few_shot_mode=0,
    verbose=True,
    cache_dir="cache"
)

# Test cache detection
print("\n" + "="*80)
print("Testing has_test_cache_for_dataset...")
print("="*80)

has_cache = llm_model.has_test_cache_for_dataset(df_val)
print(f"\nResult: {has_cache}")
