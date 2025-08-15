#!/usr/bin/env python3
"""
Command-line interface for training LLM+ML Fusion models.

This script provides a complete pipeline for training fusion ensembles that combine
traditional ML models (like RoBERTa) with LLM models (like DeepSeek/OpenAI/Gemini).

Example usage:
    python train_fusion.py --config fusion_config.yaml
    python train_fusion.py --data data.csv --ml-model roberta-base --llm-model deepseek-chat --output fusion_model
"""

import argparse
import yaml
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.llm.gemini_classifier import GeminiClassifier
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.config.settings import Config
from textclassify.core.types import ModelType, TrainingData, ModelConfig, ClassificationType, EnsembleConfig


class FusionTrainer:
    """Main trainer class for fusion ensembles."""
    
    def __init__(self, config: Dict):
        """Initialize fusion trainer with configuration."""
        self.config = config
        self.ml_model = None
        self.llm_model = None
        self.fusion_ensemble = None
        
        # Setup paths
        self.output_dir = Path(config.get('output_dir', 'fusion_output'))
        self.output_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> TrainingData:
        """Load and prepare training data."""
        data_config = self.config['data']
        
        # Load CSV data
        df = pd.read_csv(data_config['file_path'])
        
        # Extract texts and labels
        text_column = data_config['text_column']
        label_columns = data_config['label_columns']
        
        texts = df[text_column].tolist()
        
        # Prepare labels based on classification type
        classification_type = ClassificationType.MULTI_LABEL if data_config.get('multi_label', False) else ClassificationType.MULTI_CLASS
        
        if classification_type == ClassificationType.MULTI_CLASS:
            # For multi-class, convert string labels to one-hot encoding
            if isinstance(df[label_columns[0]].iloc[0], str):
                # String labels - convert to one-hot
                unique_labels = sorted(df[label_columns[0]].unique())
                labels = []
                for label in df[label_columns[0]]:
                    one_hot = [0] * len(unique_labels)
                    one_hot[unique_labels.index(label)] = 1
                    labels.append(one_hot)
                self.label_columns = unique_labels
            else:
                # Already binary encoded
                labels = df[label_columns].values.tolist()
                self.label_columns = label_columns
        else:
            # Multi-label - use binary encoding directly
            labels = df[label_columns].values.tolist()
            self.label_columns = label_columns
        
        return TrainingData(
            texts=texts,
            labels=labels,
            classification_type=classification_type
        )
    
    def create_ml_model(self) -> RoBERTaClassifier:
        """Create and configure ML model."""
        ml_config_dict = self.config['models']['ml']
        
        ml_config = ModelConfig(
            model_name=ml_config_dict.get('model_name', 'roberta-base'),
            model_type=ModelType.TRADITIONAL_ML,
            parameters={
                'model_name': ml_config_dict.get('model_name', 'roberta-base'),
                'max_length': ml_config_dict.get('max_length', 512),
                'learning_rate': ml_config_dict.get('learning_rate', 2e-5),
                'num_epochs': ml_config_dict.get('num_epochs', 3),
                'batch_size': ml_config_dict.get('batch_size', 16),
                'label_columns': self.label_columns
            }
        )
        
        return RoBERTaClassifier(config=ml_config)
    
    def create_llm_model(self) -> 'BaseLLMClassifier':
        """Create and configure LLM model."""
        llm_config_dict = self.config['models']['llm']
        provider = llm_config_dict.get('provider', 'deepseek')
        
        llm_config = Config()
        llm_config.model_type = ModelType.LLM
        llm_config.parameters = {
            'model': llm_config_dict.get('model', 'deepseek-chat'),
            'temperature': llm_config_dict.get('temperature', 0.1),
            'max_completion_tokens': llm_config_dict.get('max_completion_tokens', 150)
        }
        
        # Add provider-specific parameters
        if provider == 'openai':
            return OpenAIClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.config['data'].get('multi_label', False)
            )
        elif provider == 'gemini':
            llm_config.parameters.update({
                'top_p': llm_config_dict.get('top_p', 0.95),
                'top_k': llm_config_dict.get('top_k', 40)
            })
            return GeminiClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.config['data'].get('multi_label', False)
            )
        else:  # default to deepseek
            llm_config.parameters.update({
                'top_p': llm_config_dict.get('top_p', 1.0),
                'frequency_penalty': llm_config_dict.get('frequency_penalty', 0.0),
                'presence_penalty': llm_config_dict.get('presence_penalty', 0.0)
            })
            return DeepSeekClassifier(
                config=llm_config,
                label_columns=self.label_columns,
                multi_label=self.config['data'].get('multi_label', False)
            )
    
    def create_fusion_ensemble(self) -> FusionEnsemble:
        """Create and configure fusion ensemble."""
        fusion_config_dict = self.config['fusion']
        
        fusion_config = EnsembleConfig(
            ensemble_method='fusion',
            models=[self.ml_model, self.llm_model],
            parameters={
                'fusion_hidden_dims': fusion_config_dict.get('hidden_dims', [64, 32]),
                'ml_lr': fusion_config_dict.get('ml_learning_rate', 1e-5),
                'fusion_lr': fusion_config_dict.get('fusion_learning_rate', 1e-3),
                'num_epochs': fusion_config_dict.get('num_epochs', 10),
                'batch_size': fusion_config_dict.get('batch_size', 16)
            }
        )
        
        fusion_ensemble = FusionEnsemble(fusion_config)
        fusion_ensemble.add_ml_model(self.ml_model)
        fusion_ensemble.add_llm_model(self.llm_model)
        
        return fusion_ensemble
    
    def train(self):
        """Execute complete training pipeline."""
        print("Starting Fusion Training Pipeline")
        print("="*50)
        
        # Step 1: Load data
        print("1. Loading training data...")
        training_data = self.load_data()
        print(f"   ✓ Loaded {len(training_data.texts)} samples")
        print(f"   ✓ Classification type: {training_data.classification_type.value}")
        print(f"   ✓ Labels: {self.label_columns}")
        
        # Step 2: Create models
        print("\n2. Creating models...")
        self.ml_model = self.create_ml_model()
        print(f"   ✓ ML Model: {self.ml_model.model_name}")
        
        self.llm_model = self.create_llm_model()
        print(f"   ✓ LLM Model: {self.llm_model.model}")
        
        # Step 3: Create fusion ensemble
        print("\n3. Creating fusion ensemble...")
        self.fusion_ensemble = self.create_fusion_ensemble()
        print("   ✓ Fusion ensemble configured")
        
        # Step 4: Train
        print("\n4. Training fusion ensemble...")
        try:
            self.fusion_ensemble.fit(training_data)
            print("   ✓ Training completed successfully!")
        except Exception as e:
            print(f"   ✗ Training failed: {str(e)}")
            raise
        
        # Step 5: Save model
        print("\n5. Saving trained model...")
        self.save_model()
        print(f"   ✓ Model saved to {self.output_dir}")
        
        print("\n" + "="*50)
        print("Training completed successfully!")
    
    def save_model(self):
        """Save trained fusion ensemble and configuration."""
        # Save fusion ensemble
        model_path = self.output_dir / 'fusion_ensemble.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.fusion_ensemble, f)
        
        # Save configuration
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save label mapping
        labels_path = self.output_dir / 'label_columns.txt'
        with open(labels_path, 'w') as f:
            for label in self.label_columns:
                f.write(f"{label}\n")
        
        # Save model info
        info_path = self.output_dir / 'model_info.yaml'
        model_info = {
            'ml_model': self.ml_model.model_info,
            'llm_model': self.llm_model.model_info,
            'fusion_ensemble': {
                'classification_type': self.fusion_ensemble.classification_type.value,
                'num_labels': len(self.label_columns),
                'classes': self.label_columns
            }
        }
        with open(info_path, 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
    
    def evaluate(self, test_data_path: str):
        """Evaluate trained model on test data."""
        if not self.fusion_ensemble or not self.fusion_ensemble.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating fusion ensemble...")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        test_texts = test_df[self.config['data']['text_column']].tolist()
        
        # Prepare test labels
        label_columns = self.config['data']['label_columns']
        if self.fusion_ensemble.classification_type == ClassificationType.MULTI_CLASS:
            if isinstance(test_df[label_columns[0]].iloc[0], str):
                test_labels = []
                for label in test_df[label_columns[0]]:
                    one_hot = [0] * len(self.label_columns)
                    if label in self.label_columns:
                        one_hot[self.label_columns.index(label)] = 1
                    test_labels.append(one_hot)
            else:
                test_labels = test_df[label_columns].values.tolist()
        else:
            test_labels = test_df[label_columns].values.tolist()
        
        # Make predictions
        result = self.fusion_ensemble.predict(test_texts, test_labels)
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 40)
        
        if result.metadata and 'metrics' in result.metadata:
            metrics = result.metadata['metrics']
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    continue  # Skip classification report details
                print(f"{metric_name}: {value:.4f}")
        
        # Save evaluation results
        eval_path = self.output_dir / 'evaluation_results.yaml'
        eval_data = {
            'predictions': result.predictions,
            'metrics': result.metadata.get('metrics', {}) if result.metadata else {}
        }
        with open(eval_path, 'w') as f:
            yaml.dump(eval_data, f, default_flow_style=False)
        
        print(f"\nEvaluation results saved to {eval_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict:
    """Create default configuration."""
    return {
        'data': {
            'file_path': 'data/train.csv',
            'text_column': 'text',
            'label_columns': ['label'],
            'multi_label': False
        },
        'models': {
            'ml': {
                'model_name': 'roberta-base',
                'max_length': 512,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'batch_size': 16
            },
            'llm': {
                'provider': 'deepseek',
                'model': 'deepseek-chat',
                'temperature': 0.1,
                'max_completion_tokens': 150
            }
        },
        'fusion': {
            'hidden_dims': [64, 32],
            'ml_learning_rate': 1e-5,
            'fusion_learning_rate': 1e-3,
            'num_epochs': 10,
            'batch_size': 16
        },
        'output_dir': 'fusion_output'
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train LLM+ML Fusion models')
    
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--test-data', type=str, help='Path to test data CSV file (for evaluation)')
    parser.add_argument('--ml-model', type=str, default='roberta-base', help='ML model name')
    parser.add_argument('--llm-model', type=str, default='deepseek-chat', help='LLM model name')
    parser.add_argument('--llm-provider', type=str, default='deepseek', 
                       choices=['deepseek', 'openai', 'gemini'], help='LLM provider')
    parser.add_argument('--output', type=str, default='fusion_output', help='Output directory')
    parser.add_argument('--multi-label', action='store_true', help='Use multi-label classification')
    parser.add_argument('--create-config', type=str, help='Create default config file at specified path')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Default configuration created at {args.create_config}")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Create config from command line arguments
        config = create_default_config()
        if args.data:
            config['data']['file_path'] = args.data
        if args.ml_model:
            config['models']['ml']['model_name'] = args.ml_model
        if args.llm_model:
            config['models']['llm']['model'] = args.llm_model
        if args.llm_provider:
            config['models']['llm']['provider'] = args.llm_provider
        if args.output:
            config['output_dir'] = args.output
        if args.multi_label:
            config['data']['multi_label'] = True
    
    # Create and run trainer
    trainer = FusionTrainer(config)
    
    try:
        trainer.train()
        
        # Run evaluation if test data provided
        if args.test_data:
            trainer.evaluate(args.test_data)
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
