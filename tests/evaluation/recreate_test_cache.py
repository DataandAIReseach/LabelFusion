"""
Script to recreate LLM cache for test set only.
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
    """Create LLM cache for test set."""
    
    print("="*80)
    print("RECREATING LLM CACHE FOR TEST SET")
    print("="*80)
    
    # Configuration
    data_dir = "data/goemotions"
    cache_dir = "cache"
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    
    # Load datasets
    print(f"\nLoading datasets from {data_dir}...")
    df_train = pd.read_csv(os.path.join(data_dir, "goemotions_all_train_balanced.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "goemotions_all_test_balanced.csv"))
    
    print(f"  Training samples: {len(df_train)}")
    print(f"  Test samples: {len(df_test)}")
    
    # Label columns (28 emotions)
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    text_column = 'text'
    
    print(f"\nConfiguration:")
    print(f"  LLM provider: {llm_provider}")
    print(f"  Text column: {text_column}")
    print(f"  Label columns: {len(label_columns)} emotions")
    print(f"  Cache directory: {cache_dir}")
    
    # Create LLM model
    print(f"\nCreating {llm_provider.upper()} classifier...")
    
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
        auto_use_cache=False,  # Disable to force recreation
        cache_dir=cache_dir,
        multi_label=True,
        auto_save_results=True,
        output_dir=cache_dir
    )
    
    # Run LLM prediction on test set
    print("\n" + "="*80)
    print("RUNNING LLM PREDICTION ON TEST SET")
    print("="*80)
    print(f"\nUsing ALL training data ({len(df_train)} samples) for few-shot learning...")
    print(f"Predicting on {len(df_test)} test samples...")
    print("\nThis will take approximately 2-3 hours...\n")
    
    import time
    start = time.time()
    
    # Predict on test set
    test_result = llm_model.predict(train_df=df_train, test_df=df_test)
    
    duration = time.time() - start
    
    # Show results
    print("\n" + "="*80)
    print("TEST SET CACHE CREATION COMPLETED")
    print("="*80)
    
    print(f"\n⏱️  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    print(f"⚡ Average per sample: {duration/len(df_test):.2f} seconds")
    
    if test_result.metadata and 'metrics' in test_result.metadata:
        metrics = test_result.metadata['metrics']
        print(f"\nLLM Performance on Test Set:")
        print(f"  F1 Score: {metrics.get('f1', 0.0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0.0):.4f}")
        print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    
    print(f"\n✅ Test cache created in: {cache_dir}")
    print()


if __name__ == "__main__":
    main()
