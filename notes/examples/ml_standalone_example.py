"""Standalone ML Classification Example

Demonstrates using RoBERTa classifier for both single-label and multi-label
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

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import ModelConfig, ModelType


def example_single_label_classification():
    """Single-label classification with AG News dataset."""
    print("="*80)
    print("SINGLE-LABEL CLASSIFICATION (AG NEWS)")
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
            'label_World': [1,0,0,0,1,0,0,1,0,0],
            'label_Sports': [0,1,0,1,0,0,1,0,0,0],
            'label_Business': [0,0,1,0,0,1,0,0,1,0],
            'label_Sci/Tech': [0,0,0,0,0,0,0,0,0,1]
        })
        ag_val = ag_train.sample(n=5, random_state=42)
        ag_test = ag_train.sample(n=5, random_state=43)
    
    # Detect columns
    text_column = 'description' if 'description' in ag_train.columns else 'text'
    label_columns = [col for col in ag_train.columns if col.startswith('label_')]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns}")
    
    # Configure ML model
    ml_config = ModelConfig(
        model_name='roberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'roberta-base',
            'max_length': 128,
            'learning_rate': 2e-5,
            'num_epochs': 1,
            'batch_size': 4,
        }
    )
    
    # Initialize classifier
    ml_model = RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=False,
        auto_save_results=False,
        output_dir="examples/temp",
        experiment_name="example_ml_ag"
    )
    
    # Train
    print("\nTraining ML model...")
    ml_model.fit(ag_train, ag_val)
    
    # Predict
    print("Predicting...")
    result = ml_model.predict(ag_test)
    
    print(f"\nPredictions: {result.predictions[:3]}")
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
        reuters_train = pd.read_csv(reuters_dir / "train.csv").sample(n=10, random_state=42)
        reuters_val = pd.read_csv(reuters_dir / "val.csv").sample(n=5, random_state=42)
        reuters_test = pd.read_csv(reuters_dir / "test.csv").sample(n=5, random_state=42)
        print(f"\nLoaded Reuters: train={len(reuters_train)}, val={len(reuters_val)}, test={len(reuters_test)}")
    except Exception as e:
        print(f"Could not load Reuters dataset: {e}")
        print("Using mock data instead...")
        reuters_train = pd.DataFrame({
            'text': [f'Economic report {i}' for i in range(10)],
            'earn': [1,0,1,0,1,0,0,1,0,0],
            'acq': [0,1,0,1,0,1,0,0,1,0],
            'crude': [1,0,0,0,1,0,1,0,0,1]
        })
        reuters_val = reuters_train.sample(n=5, random_state=42)
        reuters_test = reuters_train.sample(n=5, random_state=43)
    
    # Detect columns
    text_column = 'text'
    label_columns = [col for col in reuters_train.columns 
                     if col not in ['text', 'id'] and reuters_train[col].dtype in ['int64', 'float64']]
    
    print(f"Text column: {text_column}")
    print(f"Label columns: {label_columns[:5]}... ({len(label_columns)} total)")
    
    # Configure ML model
    ml_config = ModelConfig(
        model_name='roberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'roberta-base',
            'max_length': 128,
            'learning_rate': 2e-5,
            'num_epochs': 1,
            'batch_size': 4,
        }
    )
    
    # Initialize classifier
    ml_model = RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        auto_save_results=False,
        output_dir="examples/temp",
        experiment_name="example_ml_reuters"
    )
    
    # Train
    print("\nTraining ML model...")
    ml_model.fit(reuters_train, reuters_val)
    
    # Predict
    print("Predicting...")
    result = ml_model.predict(reuters_test)
    
    print(f"\nPredictions: {result.predictions[:3]}")
    if hasattr(result, 'metadata') and 'metrics' in result.metadata:
        print(f"Metrics: {result.metadata['metrics']}")
    print("\nMulti-label classification: PASSED")


if __name__ == '__main__':
    print("ML STANDALONE CLASSIFICATION EXAMPLES")
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
