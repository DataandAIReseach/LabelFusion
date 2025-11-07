"""Test script to check ML model cache loading."""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelConfig, ModelType

# Test configuration
cache_dir = "cache/old_models"
test_model_hash = "108fe7f5"  # One of the cached models
ml_cache_path = os.path.join(cache_dir, f"fusion_roberta_model_{test_model_hash}")

print("="*80)
print("ML CACHE LOADING TEST")
print("="*80)

# Check if cache file exists
print(f"\n1. Checking cache file existence...")
print(f"   Cache path: {ml_cache_path}")
if os.path.exists(ml_cache_path):
    print(f"   ✅ Cache file exists")
    file_size_mb = os.path.getsize(ml_cache_path) / (1024 * 1024)
    print(f"   📦 File size: {file_size_mb:.2f} MB")
else:
    print(f"   ❌ Cache file does NOT exist")
    sys.exit(1)

# Create a RoBERTa classifier
print(f"\n2. Creating RoBERTa classifier...")
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

# GoEmotions label columns
label_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

ml_model = RoBERTaClassifier(
    config=ml_config,
    text_column='text',
    label_columns=label_columns,
    multi_label=True,
    auto_save_path=None,  # Don't auto-save for this test
    auto_save_results=False,
    output_dir="outputs/test",
    experiment_name="cache_test"
)
print(f"   ✅ Classifier created")

# Try to load the cached model
print(f"\n3. Attempting to load cached model...")
try:
    ml_model.load_model(ml_cache_path)
    print(f"   ✅ Model loaded successfully!")
    print(f"   📊 Model trained: {ml_model.is_trained}")
    print(f"   📊 Number of labels: {ml_model.num_labels}")
    print(f"   📊 Classes: {len(ml_model.classes_)} emotions")
    
    # Check if model components are loaded
    print(f"\n4. Checking loaded components...")
    print(f"   Model object: {'✅ Loaded' if ml_model.model is not None else '❌ Missing'}")
    print(f"   Tokenizer: {'✅ Loaded' if ml_model.tokenizer is not None else '❌ Missing'}")
    print(f"   Classification type: {ml_model.classification_type}")
    
    # Try a simple prediction test
    print(f"\n5. Testing prediction with loaded model...")
    test_df = pd.DataFrame({
        'text': ['I am so happy today!', 'This is terrible and makes me angry.'],
        # Add dummy labels
        **{col: [0, 0] for col in label_columns}
    })
    
    try:
        result = ml_model.predict(test_df)
        print(f"   ✅ Prediction successful!")
        print(f"   📊 Predicted {len(result.predictions)} samples")
        print(f"   📊 First prediction: {result.predictions[0]}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*80)
