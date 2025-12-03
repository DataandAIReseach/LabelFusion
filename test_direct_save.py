#!/usr/bin/env python3
"""Direct test of _save_cached_llm_predictions method."""

import json
import pandas as pd
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier

# Load validation data
val_df = pd.read_csv('data/reuters/val.csv')
print(f"Loaded {len(val_df)} validation samples")

# Setup models
ml_model = RoBERTaClassifier(
    label_columns=['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'],
    text_column='text',
    model_name='roberta-base',
    batch_size=16,
    max_length=256
)

llm_model = OpenAIClassifier(
    label_columns=['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'],
    text_column='text',
    classification_type='multi-label',
    model_name='gpt-4o-mini',
    batch_size=32,
    cache_dir='cache'
)

# Create ensemble
ensemble = FusionEnsemble(
    ml_model=ml_model,
    llm_model=llm_model,
    classification_type='multi-label'
)

# Load existing cache
with open('cache/val_b6e3cb0f.json', 'r') as f:
    cache_data = json.load(f)

print(f"Existing cache has {len(cache_data['predictions'])} predictions")

# Create dummy predictions (all zeros for testing)
test_predictions = [[0] * 10 for _ in range(len(val_df))]

# Copy existing predictions to test_predictions
for entry in cache_data['predictions']:
    text = entry['text']
    # Find index in val_df
    for idx, row in val_df.iterrows():
        if row['text'] == text:
            test_predictions[idx] = entry['prediction']
            break

print(f"Prepared {len(test_predictions)} predictions for save test")
print(f"Non-zero predictions: {sum(1 for p in test_predictions if sum(p) > 0)}")

# Test save
print("\n=== Testing _save_cached_llm_predictions ===")
try:
    ensemble._save_cached_llm_predictions(test_predictions, 'cache/val', val_df)
    print("✅ Save completed")
except Exception as e:
    print(f"❌ Save failed: {e}")
    import traceback
    traceback.print_exc()

# Check result
print("\n=== Checking result ===")
with open('cache/val_b6e3cb0f.json', 'r') as f:
    new_cache = json.load(f)

print(f"Cache now has {len(new_cache['predictions'])} predictions")
print(f"Updated: {new_cache['metadata']['last_updated']}")
