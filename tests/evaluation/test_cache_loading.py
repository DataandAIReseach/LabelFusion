"""
Test script to verify that LLM cache loading works correctly.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelConfig, ModelType


def main():
    """Test cache loading."""
    
    print("="*80)
    print("TESTING LLM CACHE LOADING")
    print("="*80)
    
    # Configuration
    data_dir = "data/goemotions"
    cache_dir = "cache"
    
    # Load datasets (just 10 samples for testing)
    print(f"\nLoading test dataset...")
    df_train = pd.read_csv(os.path.join(data_dir, "goemotions_all_train_balanced.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "goemotions_all_test_balanced.csv")).head(10)
    
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # Label columns
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    text_column = 'text'
    
    # Create LLM model with auto_use_cache enabled
    print(f"\nCreating OpenAI classifier with auto_use_cache=True...")
    llm_config = ModelConfig(
        model_name="gpt-5-nano",
        model_type=ModelType.LLM,
        parameters={
            "model": "gpt-5-nano",
            "temperature": 0.1,
            "max_completion_tokens": 150,
            "top_p": 1.0
        }
    )
    
    llm_model = OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        auto_use_cache=True,
        cache_dir=cache_dir,
        multi_label=True,
        auto_save_results=False,
        output_dir="outputs/cache_test"
    )
    
    # Try to predict - should load from cache
    print("\n" + "="*80)
    print("TESTING CACHE LOADING")
    print("="*80)
    print("\nAttempting prediction with auto_use_cache enabled...")
    print("If cache is found, this should complete in < 1 second!")
    print("If cache is NOT found, it will take ~3 minutes...\n")
    
    import time
    start = time.time()
    
    result = llm_model.predict(train_df=df_train, test_df=df_test)
    
    duration = time.time() - start
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"\n⏱️  Duration: {duration:.2f} seconds")
    
    if duration < 5:
        print("✅ SUCCESS! Cache was loaded (< 5 seconds)")
    else:
        print("❌ FAILED! No cache found, ran full inference")
    
    if result.metadata and 'metrics' in result.metadata:
        metrics = result.metadata['metrics']
        print(f"\nMetrics:")
        print(f"  F1 Score: {metrics.get('f1', 0.0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    
    print()


if __name__ == "__main__":
    main()
