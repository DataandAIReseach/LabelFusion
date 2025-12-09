"""Standalone LLM Classification Example

Demonstrates using OpenAI LLM classifier for both single-label and multi-label
classification tasks with real datasets (AG News and Reuters).
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, ModelType


def example_single_label_classification():
    """Single-label classification with AG News dataset."""
    print("="*80)
    print("SINGLE-LABEL CLASSIFICATION (AG NEWS)")
    print("="*80)
    
    # Load AG News dataset
    ag_news_dir = project_root / "data" / "ag_news"
    try:
        ag_train = pd.read_csv(ag_news_dir / "ag_train_balanced.csv").sample(n=5, random_state=42)
        ag_test = pd.read_csv(ag_news_dir / "ag_test_balanced.csv").sample(n=3, random_state=42)
        print(f"\nLoaded AG News: train={len(ag_train)}, test={len(ag_test)}")
    except Exception as e:
        print(f"Could not load AG News dataset: {e}")
        print("Using mock data instead...")
        ag_train = pd.DataFrame({
            'description': [f'Sample news {i}' for i in range(5)],
            'label_World': [1,0,0,0,1],
            'label_Sports': [0,1,0,1,0],
            'label_Business': [0,0,1,0,0],
            'label_Sci/Tech': [0,0,0,0,0]
        })
        ag_test = ag_train.sample(n=3, random_state=42)
    
    # Detect columns
    text_column = 'description' if 'description' in ag_train.columns else 'text'
    label_columns = [col for col in ag_train.columns if col.startswith('label_')]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns}")
    
    # Configure LLM model
    llm_config = ModelConfig(
        model_name="gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_completion_tokens": 50,
        }
    )
    
    # Initialize classifier
    llm_model = OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=False,
        auto_save_results=False,
        output_dir="examples/temp",
        experiment_name="example_llm_ag",
        cache_dir="examples/temp/cache"
    )
    
    # Predict (LLM uses few-shot from train_df)
    print("\nPredicting with LLM...")
    result = llm_model.predict(train_df=ag_train, test_df=ag_test)
    
    print(f"\nPredictions: {result.predictions}")
    if hasattr(result, 'metadata') and 'metrics' in result.metadata:
        print(f"Metrics: {result.metadata['metrics']}")
    print("\nSingle-label classification: PASSED")


def example_multi_label_classification():
    """Multi-label classification with Reuters dataset."""
    print("\n" + "="*80)
    print("MULTI-LABEL CLASSIFICATION (REUTERS)")
    print("="*80)
    
    # Load Reuters dataset
    reuters_dir = project_root / "data" / "reuters"
    try:
        reuters_train = pd.read_csv(reuters_dir / "train.csv").sample(n=5, random_state=42)
        reuters_test = pd.read_csv(reuters_dir / "test.csv").sample(n=3, random_state=42)
        print(f"\nLoaded Reuters: train={len(reuters_train)}, test={len(reuters_test)}")
    except Exception as e:
        print(f"Could not load Reuters dataset: {e}")
        print("Using mock data instead...")
        reuters_train = pd.DataFrame({
            'text': [f'Economic report {i}' for i in range(5)],
            'earn': [1,0,1,0,1],
            'acq': [0,1,0,1,0],
            'crude': [1,0,0,0,1]
        })
        reuters_test = reuters_train.sample(n=3, random_state=42)
    
    # Detect columns
    text_column = 'text'
    label_columns = [col for col in reuters_train.columns 
                     if col not in ['text', 'id'] and reuters_train[col].dtype in ['int64', 'float64']]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns[:5]}... ({len(label_columns)} total)")
    
    # Configure LLM model
    llm_config = ModelConfig(
        model_name="gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_completion_tokens": 100,
        }
    )
    
    # Initialize classifier
    llm_model = OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        auto_save_results=False,
        output_dir="examples/temp",
        experiment_name="example_llm_reuters",
        cache_dir="examples/temp/cache"
    )
    
    # Predict (LLM uses few-shot from train_df)
    print("\nPredicting with LLM...")
    result = llm_model.predict(train_df=reuters_train, test_df=reuters_test)
    
    print(f"\nPredictions: {result.predictions}")
    if hasattr(result, 'metadata') and 'metrics' in result.metadata:
        print(f"Metrics: {result.metadata['metrics']}")
    print("\nMulti-label classification: PASSED")


if __name__ == '__main__':
    print("LLM STANDALONE CLASSIFICATION EXAMPLES")
    print("="*80)
    print("Note: Requires OpenAI API key in environment (OPENAI_API_KEY)")
    print("="*80)
    
    try:
        example_single_label_classification()
    except Exception as e:
        print(f"\nSingle-label example failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_multi_label_classification()
    except Exception as e:
        print(f"\nMulti-label example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
