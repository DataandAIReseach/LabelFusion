"""Simple test to check if ML model cache loading works."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import ModelConfig, ModelType

# Test with one of the cached models
cache_path = "cache/fusion_roberta_model_2662ee63"  # 10% model

print("="*80)
print("TESTING ML MODEL CACHE LOADING")
print("="*80)

print(f"\nCache file: {cache_path}")
print(f"Exists: {os.path.exists(cache_path)}")

if not os.path.exists(cache_path):
    print("❌ Cache file not found!")
    sys.exit(1)

# Create classifier
ml_config = ModelConfig(
    model_name='roberta-base',
    model_type=ModelType.TRADITIONAL_ML,
    parameters={'model_name': 'roberta-base'}
)

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
    multi_label=True
)

print(f"\n📦 Attempting to load model from: {cache_path}")
try:
    ml_model.load_model(cache_path)
    print("✅ Model loaded successfully!")
    print(f"   - is_trained: {ml_model.is_trained}")
    print(f"   - num_labels: {ml_model.num_labels}")
    print(f"   - has model: {ml_model.model is not None}")
    print(f"   - has tokenizer: {ml_model.tokenizer is not None}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test prediction with loaded model
print("\n📊 Testing prediction with loaded model...")
import pandas as pd

test_df = pd.DataFrame({
    'text': ['I am so happy today!', 'This is terrible and makes me angry.'],
    **{col: [0, 0] for col in label_columns}
})

try:
    result = ml_model.predict(test_df)
    print("✅ Prediction successful!")
    print(f"   - Predicted {len(result.predictions)} samples")
    print(f"   - First prediction (emotions): {result.predictions[0]}")
    if result.metadata:
        metrics = result.metadata.get('metrics', {})
        print(f"   - Accuracy: {metrics.get('accuracy', 'N/A')}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ TEST PASSED!")
