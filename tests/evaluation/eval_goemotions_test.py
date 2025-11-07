#!/usr/bin/env python3
"""Quick test script for Fusion Ensemble on GoEmotions.

This script creates a small Fusion ensemble using RoBERTa (ML) and OpenAI (LLM)
configured for the GoEmotions multi-label task. It's intended as a drop-in
test file (not a full experiment runner) so you can quickly run a sanity check.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so local package imports resolve when run from tests/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import ModelConfig, ModelType, EnsembleConfig


def main():
    # Load your dataset (GoEmotions balanced files)
    train_df = pd.read_csv("/scratch/users/u19147/LabelFusion/data/goemotions/goemotions_all_train_balanced.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.read_csv("/scratch/users/u19147/LabelFusion/data/goemotions/goemotions_all_val_balanced.csv").sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv("/scratch/users/u19147/LabelFusion/data/goemotions/goemotions_all_test_balanced.csv").sample(frac=1, random_state=42).reset_index(drop=True)

    # Shared dataset settings
    text_column = 'text'
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # Configure ML model (RoBERTa)
    ml_config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "model_name": "roberta-base",
            "learning_rate": 2e-5,
            "num_epochs": 2,  # Fast training for testing
            "batch_size": 8,
            "max_length": 256
        }
    )

    # Create RoBERTa classifier
    ml_classifier = RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        auto_save_path="cache/experimente/fusion_roberta_model",
        auto_save_results=True
    )

    # Configure LLM model (OpenAI)
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

    # Create OpenAI LLM classifier
    llm_classifier = OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        enable_cache=True,
        cache_dir="cache/experimente/fusion_openai_cache",
        multi_label=True,
        auto_save_results=True
    )

    # Configure Fusion Ensemble
    fusion_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_classifier, llm_classifier],
        parameters={
            "fusion_hidden_dims": [64, 32],
            "ml_lr": 1e-5,
            "fusion_lr": 5e-4,
            "num_epochs": 10,
            "batch_size": 8,
            "classification_type": "multi_label",
            "val_llm_cache_path": "cache/fusion_openai_cache/val",
            "test_llm_cache_path": "cache/fusion_openai_cache/test"
        }
    )

    # Create Fusion Ensemble
    fusion_ensemble = FusionEnsemble(
        fusion_config,
        output_dir="outputs",
        experiment_name="fusion_test",
        auto_save_results=True,
        save_intermediate_llm_predictions=False
    )

    # Add models to ensemble
    fusion_ensemble.add_ml_model(ml_classifier)
    fusion_ensemble.add_llm_model(llm_classifier)

    # Train the fusion ensemble
    print("Starting fusion training (this may take a while)...")
    try:
        training_result = fusion_ensemble.fit(train_df, val_df)
        print("Fusion training finished")

        # Test prediction
        print("Running fusion prediction on test set...")
        result = fusion_ensemble.predict(test_df=test_df)
        print("Prediction complete. Result metadata:")
        print(result.metadata if hasattr(result, 'metadata') else "No metadata available")
    except Exception as e:
        print(f"Error while running fusion test: {e}")


if __name__ == '__main__':
    main()
