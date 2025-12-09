"""Fusion Ensemble Classification Example

Demonstrates combining ML (RoBERTa) and LLM (OpenAI) classifiers with a trainable
fusion layer (MLP) for both single-label and multi-label classification tasks.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionClassifier
from textclassify.core.types import ModelConfig, ModelType


def example_single_label_fusion():
    """Single-label fusion with AG News dataset."""
    print("="*80)
    print("SINGLE-LABEL FUSION (AG NEWS)")
    print("="*80)
    
    # Load AG News dataset
    ag_news_dir = project_root / "data" / "ag_news"
    try:
        ag_train = pd.read_csv(ag_news_dir / "ag_train_balanced.csv").sample(n=10, random_state=42)
        ag_val = pd.read_csv(ag_news_dir / "ag_val_balanced.csv").sample(n=5, random_state=42)
        ag_test = pd.read_csv(ag_news_dir / "ag_test_balanced.csv").sample(n=5, random_state=42)
        print(f"\nLoaded AG News: train={len(ag_train)}, val={len(ag_val)}, test={len(ag_test)}")
    except Exception as e:
        print(f"Could not load AG News dataset: {e}")
        print("Using mock data instead...")
        ag_train = pd.DataFrame({
            'description': [f'Sample news {i}' for i in range(10)],
            'label_World': [1,0,0,0,1,0,1,0,0,1],
            'label_Sports': [0,1,0,1,0,1,0,0,1,0],
            'label_Business': [0,0,1,0,0,0,0,1,0,0],
            'label_Sci/Tech': [0,0,0,0,0,0,0,0,0,0]
        })
        ag_val = ag_train.sample(n=5, random_state=42)
        ag_test = ag_train.sample(n=5, random_state=43)
    
    # Detect columns
    text_column = 'description' if 'description' in ag_train.columns else 'text'
    label_columns = [col for col in ag_train.columns if col.startswith('label_')]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns}")
    
    # Configure models
    ml_config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.ML,
        parameters={"batch_size": 4, "num_epochs": 1, "learning_rate": 2e-5}
    )
    
    llm_config = ModelConfig(
        model_name="gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={"model": "gpt-4o-mini", "temperature": 0.1, "max_completion_tokens": 50}
    )
    
    # Initialize fusion ensemble
    fusion = FusionClassifier(
        ml_configs=[ml_config],
        llm_configs=[llm_config],
        text_column=text_column,
        label_columns=label_columns,
        multi_label=False,
        output_dir="examples/temp",
        experiment_name="example_fusion_ag",
        cache_dir="examples/temp/cache"
    )
    
    # Train
    print("\nTraining fusion ensemble...")
    fusion.train(
        train_df=ag_train,
        val_df=ag_val,
        epochs=1,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Predict
    print("\nPredicting with fusion ensemble...")
    result = fusion.predict(test_df=ag_test)
    
    print(f"\nPredictions: {result.predictions}")
    if hasattr(result, 'metadata') and 'metrics' in result.metadata:
        print(f"Metrics: {result.metadata['metrics']}")
    print("\nSingle-label fusion: PASSED")


def example_multi_label_fusion():
    """Multi-label fusion with Reuters dataset."""
    print("\n" + "="*80)
    print("MULTI-LABEL FUSION (REUTERS)")
    print("="*80)
    
    # Load Reuters dataset
    reuters_dir = project_root / "data" / "reuters"
    try:
        reuters_train = pd.read_csv(reuters_dir / "train.csv").sample(n=10, random_state=42)
        reuters_val = pd.read_csv(reuters_dir / "val.csv").sample(n=5, random_state=42)
        reuters_test = pd.read_csv(reuters_dir / "test.csv").sample(n=5, random_state=42)
        print(f"\nLoaded Reuters: train={len(reuters_train)}, val={len(reuters_val)}, test={len(reuters_test)}")
    except Exception as e:
        print(f"Could not load Reuters dataset: {e}")
        print("Using mock data instead...")
        reuters_train = pd.DataFrame({
            'text': [f'Economic report {i}' for i in range(10)],
            'earn': [1,0,1,0,1,0,1,0,1,0],
            'acq': [0,1,0,1,0,1,0,1,0,1],
            'crude': [1,0,0,0,1,1,0,0,1,0]
        })
        reuters_val = reuters_train.sample(n=5, random_state=42)
        reuters_test = reuters_train.sample(n=5, random_state=43)
    
    # Detect columns
    text_column = 'text'
    label_columns = [col for col in reuters_train.columns 
                     if col not in ['text', 'id'] and reuters_train[col].dtype in ['int64', 'float64']]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns[:5]}... ({len(label_columns)} total)")
    
    # Configure models
    ml_config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.ML,
        parameters={"batch_size": 4, "num_epochs": 1, "learning_rate": 2e-5}
    )
    
    llm_config = ModelConfig(
        model_name="gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={"model": "gpt-4o-mini", "temperature": 0.1, "max_completion_tokens": 100}
    )
    
    # Initialize fusion ensemble
    fusion = FusionClassifier(
        ml_configs=[ml_config],
        llm_configs=[llm_config],
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        output_dir="examples/temp",
        experiment_name="example_fusion_reuters",
        cache_dir="examples/temp/cache"
    )
    
    # Train
    print("\nTraining fusion ensemble...")
    fusion.train(
        train_df=reuters_train,
        val_df=reuters_val,
        epochs=1,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Predict
    print("\nPredicting with fusion ensemble...")
    result = fusion.predict(test_df=reuters_test)
    
    print(f"\nPredictions: {result.predictions}")
    if hasattr(result, 'metadata') and 'metrics' in result.metadata:
        print(f"Metrics: {result.metadata['metrics']}")
    print("\nMulti-label fusion: PASSED")


if __name__ == '__main__':
    print("FUSION ENSEMBLE CLASSIFICATION EXAMPLES")
    print("="*80)
    print("Note: Requires OpenAI API key in environment (OPENAI_API_KEY)")
    print("="*80)
    
    try:
        example_single_label_fusion()
    except Exception as e:
        print(f"\nSingle-label fusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_multi_label_fusion()
    except Exception as e:
        print(f"\nMulti-label fusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
