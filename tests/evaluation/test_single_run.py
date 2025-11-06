#!/usr/bin/env python3
"""
Quick test script to run a single experiment with 0.01% data
to test cache loading functionality.
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from textclassify.core.types import ModelConfig, ModelType
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionEnsemble

def main():
    print("="*80)
    print("SINGLE TEST RUN - 0.01% Data (21 samples)")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    df_train = pd.read_csv("data/goemotions/goemotions_all_train_balanced.csv")
    df_val = pd.read_csv("data/goemotions/goemotions_all_val_balanced.csv")
    df_test = pd.read_csv("data/goemotions/goemotions_all_test_balanced.csv")
    
    print(f"  Full training: {len(df_train)} samples")
    print(f"  Validation: {len(df_val)} samples")
    print(f"  Test: {len(df_test)} samples")
    
    # Create tiny subset (0.01% = ~21 samples)
    df_train_tiny = df_train.sample(n=21, random_state=42)
    print(f"\nCreated tiny subset: {len(df_train_tiny)} samples (0.01%)")
    
    # Label columns
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Create LLM model with cache enabled
    print("\nCreating LLM model with cache enabled...")
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
        text_column='text',
        label_columns=label_columns,
        auto_use_cache=True,
        cache_dir="cache",
        multi_label=True,
        auto_save_results=False,
        output_dir="outputs/test_single_run",
        experiment_name="test_cache"
    )
    
    print("\n" + "="*80)
    print("TESTING CACHE LOAD WITH FULL TRAINING DATA")
    print("="*80)
    
    print("\nAttempting to predict validation set with FULL training data...")
    print(f"Train data for LLM: {len(df_train)} samples")
    print(f"Val data: {len(df_val)} samples")
    
    try:
        result = llm_model.predict(train_df=df_train, test_df=df_val)
        print("\n✅ SUCCESS! Cache loaded successfully!")
        
        if result.metadata:
            metrics = result.metadata.get('metrics', {})
            print(f"\nValidation Metrics:")
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"  F1: {metrics.get('f1', 0.0):.4f}")
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
    
    print("\n" + "="*80)
    print("TESTING CACHE LOAD WITH TINY SUBSET")
    print("="*80)
    
    print("\nAttempting to predict validation set with TINY training subset...")
    print(f"Train data for LLM: {len(df_train_tiny)} samples")
    print(f"Val data: {len(df_val)} samples")
    
    try:
        result = llm_model.predict(train_df=df_train_tiny, test_df=df_val)
        print("\n✅ SUCCESS! Prediction completed!")
        
        if result.metadata:
            metrics = result.metadata.get('metrics', {})
            print(f"\nValidation Metrics:")
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"  F1: {metrics.get('f1', 0.0):.4f}")
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")

if __name__ == "__main__":
    main()
