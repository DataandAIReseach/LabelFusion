"""
Multi-class Text Classification Example

This example demonstrates how to use the textclassify package for multi-class
text classification using different models and ensemble methods.
"""

import os
from textclassify import (
    OpenAIClassifier, ClaudeClassifier, RoBERTaClassifier,
    VotingEnsemble, WeightedEnsemble,
    Config, APIKeyManager,
    ClassificationType, ModelType
)
from textclassify.utils import DataLoader, split_data, evaluate_predictions
from textclassify.core.types import TrainingData, ModelConfig, EnsembleConfig


def create_sample_data():
    """Create sample data for demonstration."""
    texts = [
        "I love this movie! It's absolutely fantastic.",
        "This film is terrible. I want my money back.",
        "The movie was okay, nothing special.",
        "Amazing cinematography and great acting!",
        "Boring and predictable plot.",
        "One of the best movies I've ever seen!",
        "Not worth watching, very disappointing.",
        "Decent movie with some good moments.",
        "Incredible story and brilliant performances.",
        "Waste of time, poorly made film."
    ]
    
    labels = [
        "positive", "negative", "neutral",
        "positive", "negative", "positive",
        "negative", "neutral", "positive", "negative"
    ]
    
    return TrainingData(
        texts=texts,
        labels=labels,
        classification_type=ClassificationType.MULTI_CLASS
    )


def example_openai_classifier():
    """Example using OpenAI classifier."""
    print("=== OpenAI Classifier Example ===")
    
    # Create configuration
    config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        api_key=os.getenv("OPENAI_API_KEY"),  # Set your API key
        parameters={
            "temperature": 0.1,
            "max_tokens": 50
        }
    )
    
    # Create classifier
    classifier = OpenAIClassifier(config)
    
    # Create sample data
    training_data = create_sample_data()
    
    # Train (for LLMs, this sets up few-shot examples)
    classifier.fit(training_data)
    
    # Test predictions
    test_texts = [
        "This movie is absolutely wonderful!",
        "I didn't like it at all.",
        "It was an average film."
    ]
    
    # Get predictions
    result = classifier.predict(test_texts)
    print(f"Predictions: {result.predictions}")
    
    # Get predictions with probabilities
    result_proba = classifier.predict_proba(test_texts)
    print(f"Predictions with probabilities:")
    for i, (text, pred, prob) in enumerate(zip(test_texts, result_proba.predictions, result_proba.probabilities)):
        print(f"  Text: {text}")
        print(f"  Prediction: {pred}")
        print(f"  Probabilities: {prob}")
        print()


def example_roberta_classifier():
    """Example using RoBERTa classifier."""
    print("=== RoBERTa Classifier Example ===")
    
    # Create configuration
    config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "max_length": 128,
            "batch_size": 8,
            "num_epochs": 2,
            "learning_rate": 2e-5
        }
    )
    
    # Create classifier
    classifier = RoBERTaClassifier(config)
    
    # Create sample data (you would use more data in practice)
    training_data = create_sample_data()
    
    # Split data
    train_data, val_data = split_data(training_data, train_ratio=0.8)
    
    print(f"Training samples: {len(train_data.texts)}")
    print(f"Validation samples: {len(val_data.texts)}")
    
    # Train the model
    print("Training RoBERTa model...")
    classifier.fit(train_data)
    
    # Test predictions
    result = classifier.predict(val_data.texts)
    print(f"Predictions: {result.predictions}")
    
    # Evaluate performance
    metrics = evaluate_predictions(result, val_data.labels)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")


def example_ensemble_classifier():
    """Example using ensemble methods."""
    print("=== Ensemble Classifier Example ===")
    
    # Create individual model configurations
    openai_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        api_key=os.getenv("OPENAI_API_KEY"),
        parameters={"temperature": 0.1, "max_tokens": 50}
    )
    
    claude_config = ModelConfig(
        model_name="claude-3-haiku-20240307",
        model_type=ModelType.LLM,
        api_key=os.getenv("CLAUDE_API_KEY"),
        parameters={"temperature": 0.1, "max_tokens": 50}
    )
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        models=[openai_config, claude_config],
        ensemble_method="voting"
    )
    
    # Create ensemble
    ensemble = VotingEnsemble(ensemble_config)
    
    # Add individual classifiers
    if openai_config.api_key:
        ensemble.add_model(OpenAIClassifier(openai_config), "openai")
    if claude_config.api_key:
        ensemble.add_model(ClaudeClassifier(claude_config), "claude")
    
    if len(ensemble.models) == 0:
        print("No API keys available for ensemble example")
        return
    
    # Train ensemble
    training_data = create_sample_data()
    ensemble.fit(training_data)
    
    # Test predictions
    test_texts = [
        "This movie is absolutely wonderful!",
        "I didn't like it at all."
    ]
    
    result = ensemble.predict(test_texts)
    print(f"Ensemble predictions: {result.predictions}")
    
    # Get model info
    print(f"Ensemble info: {ensemble.model_info}")


def example_weighted_ensemble():
    """Example using weighted ensemble."""
    print("=== Weighted Ensemble Example ===")
    
    # Create configurations with different weights
    model_configs = [
        ModelConfig(
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        ModelConfig(
            model_name="claude-3-haiku-20240307",
            model_type=ModelType.LLM,
            api_key=os.getenv("CLAUDE_API_KEY")
        )
    ]
    
    # Create weighted ensemble with custom weights
    ensemble_config = EnsembleConfig(
        models=model_configs,
        ensemble_method="weighted",
        weights=[0.7, 0.3]  # Give more weight to first model
    )
    
    ensemble = WeightedEnsemble(ensemble_config)
    
    # Add models with weights
    if model_configs[0].api_key:
        ensemble.add_model(OpenAIClassifier(model_configs[0]), "openai", weight=0.7)
    if model_configs[1].api_key:
        ensemble.add_model(ClaudeClassifier(model_configs[1]), "claude", weight=0.3)
    
    if len(ensemble.models) == 0:
        print("No API keys available for weighted ensemble example")
        return
    
    # Train and test
    training_data = create_sample_data()
    ensemble.fit(training_data)
    
    test_texts = ["This is an amazing movie!"]
    result = ensemble.predict_proba(test_texts)
    
    print(f"Weighted ensemble prediction: {result.predictions[0]}")
    print(f"Probabilities: {result.probabilities[0]}")
    print(f"Weights: {ensemble.weights}")


def example_configuration_management():
    """Example using configuration management."""
    print("=== Configuration Management Example ===")
    
    # Create and configure API key manager
    api_manager = APIKeyManager()
    
    # Check for existing API keys
    providers = ['openai', 'claude', 'gemini', 'deepseek']
    for provider in providers:
        has_key = api_manager.has_key(provider)
        print(f"{provider}: {'✓' if has_key else '✗'}")
    
    # Create configuration
    config = Config()
    
    # Set some configuration values
    config.set('llm.default_provider', 'openai')
    config.set('general.default_batch_size', 16)
    
    # Create model config from main config
    model_config = config.create_model_config(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        provider="openai"
    )
    
    print(f"Model config: {model_config.model_name}")
    print(f"Batch size: {model_config.batch_size}")
    
    # Save configuration
    config_path = "example_config.yaml"
    config.save(config_path)
    print(f"Configuration saved to {config_path}")


def main():
    """Run all examples."""
    print("TextClassify Multi-class Classification Examples")
    print("=" * 50)
    
    try:
        # Configuration example
        example_configuration_management()
        print()
        
        # Individual classifier examples
        if os.getenv("OPENAI_API_KEY"):
            example_openai_classifier()
            print()
        else:
            print("Skipping OpenAI example (no API key)")
            print()
        
        # Note: RoBERTa example requires transformers library
        try:
            example_roberta_classifier()
            print()
        except ImportError:
            print("Skipping RoBERTa example (transformers not installed)")
            print()
        
        # Ensemble examples
        if os.getenv("OPENAI_API_KEY") or os.getenv("CLAUDE_API_KEY"):
            example_ensemble_classifier()
            print()
            example_weighted_ensemble()
        else:
            print("Skipping ensemble examples (no API keys)")
    
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure to set your API keys as environment variables:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export CLAUDE_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()

