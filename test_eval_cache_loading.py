"""Test if eval_goemotions correctly loads ML cache."""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import ModelConfig, ModelType

print("="*80)
print("TESTING ML CACHE LOADING IN EVAL_GOEMOTIONS CONTEXT")
print("="*80)

# Simulate what eval_goemotions does
label_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Simulate 10% of training data (this would generate a specific hash)
cache_dir = "cache/old_models"
dataset_hash = "108fe7f5"  # Example hash for 10% data
ml_cache_path = os.path.join(cache_dir, f"fusion_roberta_model_{dataset_hash}")

print(f"\n1. Cache path: {ml_cache_path}")
print(f"   Exists: {os.path.exists(ml_cache_path)}")

if not os.path.exists(ml_cache_path):
    print("   ❌ Cache not found!")
    sys.exit(1)

# Create model as in eval_goemotions
print(f"\n2. Creating ML model...")
ml_config = ModelConfig(
    model_name='roberta-base',
    model_type=ModelType.TRADITIONAL_ML,
    parameters={
        'model_name': 'roberta-base',
        'max_length': 256,
        'learning_rate': 2e-5,
        'num_epochs': 2,
        'batch_size': 32,
    }
)

ml_model = RoBERTaClassifier(
    config=ml_config,
    text_column='text',
    label_columns=label_columns,
    multi_label=True,
    auto_save_path=ml_cache_path,
    auto_save_results=True,
    output_dir="outputs/test",
    experiment_name="cache_test"
)
print(f"   ✅ Model created")

# Check and load cached model (as in eval_goemotions line 296-305)
ml_model_loaded_from_cache = False
if os.path.exists(ml_cache_path):
    print(f"\n3. 📦 Loading cached RoBERTa model from: {ml_cache_path}")
    try:
        ml_model.load_model(ml_cache_path)
        print("   ✅ Successfully loaded cached model")
        ml_model_loaded_from_cache = True
        print(f"   📊 is_trained: {ml_model.is_trained}")
        print(f"   📊 num_labels: {ml_model.num_labels}")
    except Exception as e:
        print(f"   ⚠️ Failed to load cached model: {e}")
        import traceback
        traceback.print_exc()

if ml_model_loaded_from_cache:
    print("\n4. ✅ ML CACHE LOADING WORKS!")
    print("   The eval_goemotions script should be able to load cached models.")
else:
    print("\n4. ❌ ML CACHE LOADING FAILED!")
    print("   There's an issue with loading cached models in eval_goemotions.")

print("\n" + "="*80)
