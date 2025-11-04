"""
Test script to verify that curly braces in texts are handled correctly.
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
    """Test with texts containing curly braces."""
    
    print("="*80)
    print("TEST: CURLY BRACES IN TEXTS")
    print("="*80)
    
    # Create test data with problematic curly braces
    test_texts = [
        "This is a normal text without any issues",
        "Error: {variable} not found",
        "Something went wrong }",
        "Config: {key: value}",
        "Mixed { and } braces"
    ]
    
    # Create DataFrame with all required columns (28 emotion labels)
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Create test DataFrame
    data = {'text': test_texts}
    for label in label_columns:
        data[label] = [0, 1, 0, 0, 1]  # Dummy labels
    
    df_test = pd.DataFrame(data)
    
    # Load training data
    data_dir = "data/goemotions"
    df_train = pd.read_csv(os.path.join(data_dir, "goemotions_all_train_balanced.csv")).head(1000)
    
    print(f"\nTest texts with curly braces:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")
    
    print(f"\nTraining samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    
    text_column = 'text'
    cache_dir = "cache_test_braces"
    
    # Create LLM model
    print(f"\nCreating OpenAI classifier...")
    
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
        auto_use_cache=False,
        cache_dir=cache_dir,
        multi_label=True,
        auto_save_results=True,
        output_dir=cache_dir
    )
    
    # Run prediction
    print("\n" + "="*80)
    print("RUNNING PREDICTION")
    print("="*80)
    print("\nThis will test if curly braces cause crashes...\n")
    
    import time
    start = time.time()
    
    try:
        test_result = llm_model.predict(train_df=df_train, test_df=df_test)
        duration = time.time() - start
        
        print("\n" + "="*80)
        print("✅ SUCCESS - NO CRASH!")
        print("="*80)
        
        print(f"\n⏱️  Duration: {duration:.2f} seconds")
        print(f"⚡ Average per sample: {duration/len(df_test):.2f} seconds")
        
        if test_result.metadata and 'metrics' in test_result.metadata:
            metrics = test_result.metadata['metrics']
            print(f"\nMetrics:")
            print(f"  F1 Score: {metrics.get('f1', 0.0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0.0):.4f}")
        
        print("\n✅ Curly braces were handled correctly!")
        print("   The prompt rendering fix is working!\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        print("\nThe curly braces caused a crash. Fix needed!\n")
        raise


if __name__ == "__main__":
    main()
