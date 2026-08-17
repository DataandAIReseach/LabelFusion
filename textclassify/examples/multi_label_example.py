"""
Multi-label Text Classification Example

This example demonstrates how to use the textclassify package for multi-label
text classification where each text can have multiple labels.
"""

import os
from textclassify import (
    OpenAIClassifier, GeminiClassifier, RoBERTaClassifier,
    VotingEnsemble, ClassRoutingEnsemble,
    Config, APIKeyManager,
    ClassificationType, ModelType
)
from textclassify.utils import DataLoader, evaluate_predictions, get_data_statistics
from textclassify.core.types import TrainingData, ModelConfig, EnsembleConfig


def create_sample_multilabel_data():
    """Create sample multi-label data for demonstration."""
    texts = [
        "This action movie has amazing special effects and great fight scenes.",
        "A romantic comedy with hilarious moments and heartwarming romance.",
        "Thrilling horror film with jump scares and supernatural elements.",
        "Documentary about science and technology innovations.",
        "Family-friendly animated movie with comedy and adventure.",
        "Dramatic thriller with mystery and suspense elements.",
        "Musical comedy with great songs and funny characters.",
        "Science fiction adventure with action and futuristic themes.",
        "Romantic drama with emotional depth and beautiful cinematography.",
        "Horror comedy that blends scares with humor effectively."
    ]
    
    labels = [
        ["action", "special_effects"],
        ["comedy", "romance"],
        ["horror", "thriller"],
        ["documentary", "science"],
        ["family", "comedy", "adventure"],
        ["drama", "thriller", "mystery"],
        ["comedy", "musical"],
        ["sci_fi", "action", "adventure"],
        ["romance", "drama"],
        ["horror", "comedy"]
    ]
    
    return TrainingData(
        texts=texts,
        labels=labels,
        classification_type=ClassificationType.MULTI_LABEL
    )


def example_multilabel_llm():
    """Example using LLM for multi-label classification."""
    print("=== Multi-label LLM Classification ===")
    
    # Create configuration
    config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        api_key=os.getenv("OPENAI_API_KEY"),
        parameters={
            "temperature": 0.1,
            "max_tokens": 100
        }
    )
    
    if not config.api_key:
        print("OpenAI API key not found, skipping example")
        return
    
    # Create classifier
    classifier = OpenAIClassifier(config)
    
    # Create and analyze training data
    training_data = create_sample_multilabel_data()
    stats = get_data_statistics(training_data)
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Average labels per sample: {stats['labels_per_sample']['mean']:.2f}")
    print(f"  Class distribution: {stats['class_distribution']}")
    print()
    
    # Train classifier
    classifier.fit(training_data)
    
    # Test predictions
    test_texts = [
        "An exciting action movie with romance and comedy elements.",
        "Scary horror film with supernatural themes.",
        "Educational documentary about environmental science."
    ]
    
    print("Test predictions:")
    result = classifier.predict(test_texts)
    
    for i, (text, prediction) in enumerate(zip(test_texts, result.predictions)):
        print(f"Text: {text}")
        print(f"Predicted labels: {prediction}")
        print()
    
    # Get predictions with probabilities
    result_proba = classifier.predict_proba(test_texts)
    
    print("Predictions with probabilities:")
    for i, (text, prediction, probabilities) in enumerate(zip(
        test_texts, result_proba.predictions, result_proba.probabilities
    )):
        print(f"Text: {text}")
        print(f"Predicted labels: {prediction}")
        print(f"Probabilities:")
        for label, prob in probabilities.items():
            print(f"  {label}: {prob:.3f}")
        print()


def example_multilabel_roberta():
    """Example using RoBERTa for multi-label classification."""
    print("=== Multi-label RoBERTa Classification ===")
    
    try:
        # Create configuration
        config = ModelConfig(
            model_name="roberta-base",
            model_type=ModelType.TRADITIONAL_ML,
            parameters={
                "max_length": 128,
                "batch_size": 4,
                "num_epochs": 2,
                "learning_rate": 2e-5,
                "threshold": 0.5  # Threshold for multi-label prediction
            }
        )
        
        # Create classifier
        classifier = RoBERTaClassifier(config)
        
        # Create training data
        training_data = create_sample_multilabel_data()
        
        print(f"Training RoBERTa on {len(training_data.texts)} samples...")
        
        # Train the model
        classifier.fit(training_data)
        
        # Test predictions
        test_texts = [
            "Action-packed adventure with comedy elements.",
            "Romantic drama with beautiful cinematography."
        ]
        
        result = classifier.predict(test_texts)
        
        print("RoBERTa predictions:")
        for text, prediction in zip(test_texts, result.predictions):
            print(f"Text: {text}")
            print(f"Predicted labels: {prediction}")
            print()
        
        # Get probabilities
        result_proba = classifier.predict_proba(test_texts)
        
        print("RoBERTa predictions with probabilities:")
        for i, (text, prediction, probabilities) in enumerate(zip(
            test_texts, result_proba.predictions, result_proba.probabilities
        )):
            print(f"Text: {text}")
            print(f"Predicted labels: {prediction}")
            print(f"Top probabilities:")
            # Sort probabilities and show top 5
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs[:5]:
                print(f"  {label}: {prob:.3f}")
            print()
    
    except ImportError:
        print("RoBERTa example requires transformers library")
        print("Install with: pip install transformers torch")


def example_multilabel_ensemble():
    """Example using ensemble for multi-label classification."""
    print("=== Multi-label Ensemble Classification ===")
    
    # Create model configurations
    openai_config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        api_key=os.getenv("OPENAI_API_KEY"),
        parameters={"temperature": 0.1, "max_tokens": 100}
    )
    
    gemini_config = ModelConfig(
        model_name="gemini-1.5-flash",
        model_type=ModelType.LLM,
        api_key=os.getenv("GEMINI_API_KEY"),
        parameters={"temperature": 0.1, "max_tokens": 100}
    )
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        models=[openai_config, gemini_config],
        ensemble_method="voting"
    )
    
    # Create voting ensemble
    ensemble = VotingEnsemble(ensemble_config)
    
    # Add models if API keys are available
    models_added = 0
    if openai_config.api_key:
        ensemble.add_model(OpenAIClassifier(openai_config), "openai")
        models_added += 1
    if gemini_config.api_key:
        ensemble.add_model(GeminiClassifier(gemini_config), "gemini")
        models_added += 1
    
    if models_added == 0:
        print("No API keys available for ensemble example")
        return
    
    print(f"Created ensemble with {models_added} models")
    
    # Train ensemble
    training_data = create_sample_multilabel_data()
    ensemble.fit(training_data)
    
    # Test predictions
    test_texts = [
        "Funny romantic comedy with great music and songs.",
        "Scary thriller with mystery and supernatural elements."
    ]
    
    result = ensemble.predict(test_texts)
    
    print("Ensemble predictions:")
    for text, prediction in zip(test_texts, result.predictions):
        print(f"Text: {text}")
        print(f"Predicted labels: {prediction}")
        print()
    
    # Show ensemble info
    print(f"Ensemble info:")
    info = ensemble.model_info
    print(f"  Method: {info['ensemble_method']}")
    print(f"  Models: {info['model_names']}")


def example_class_routing_ensemble():
    """Example using class routing ensemble for multi-label classification."""
    print("=== Class Routing Ensemble ===")
    
    # Create model configurations
    model_configs = [
        ModelConfig(
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            api_key=os.getenv("OPENAI_API_KEY"),
            parameters={"temperature": 0.1}
        ),
        ModelConfig(
            model_name="gemini-1.5-flash",
            model_type=ModelType.LLM,
            api_key=os.getenv("GEMINI_API_KEY"),
            parameters={"temperature": 0.1}
        )
    ]
    
    # Create routing rules (different models for different classes)
    routing_rules = {
        "action": "openai",
        "comedy": "openai",
        "horror": "gemini",
        "thriller": "gemini",
        "romance": "openai",
        "sci_fi": "gemini"
    }
    
    ensemble_config = EnsembleConfig(
        models=model_configs,
        ensemble_method="routing",
        routing_rules=routing_rules
    )
    
    # Create routing ensemble
    ensemble = ClassRoutingEnsemble(ensemble_config)
    
    # Add models
    models_added = 0
    if model_configs[0].api_key:
        ensemble.add_model(OpenAIClassifier(model_configs[0]), "openai")
        models_added += 1
    if model_configs[1].api_key:
        ensemble.add_model(GeminiClassifier(model_configs[1]), "gemini")
        models_added += 1
    
    if models_added < 2:
        print("Class routing requires multiple API keys")
        return
    
    # Train ensemble
    training_data = create_sample_multilabel_data()
    ensemble.fit(training_data)
    
    # Show routing summary
    print("Routing rules:")
    routing_summary = ensemble.get_routing_summary()
    for class_name, model_name in routing_summary.items():
        print(f"  {class_name} -> {model_name}")
    print()
    
    # Test predictions
    test_texts = [
        "Action-packed sci-fi adventure with thrilling scenes.",
        "Romantic comedy with hilarious moments."
    ]
    
    result = ensemble.predict(test_texts)
    
    print("Class routing predictions:")
    for text, prediction in zip(test_texts, result.predictions):
        print(f"Text: {text}")
        print(f"Predicted labels: {prediction}")
        print()


def example_data_loading():
    """Example of loading multi-label data from files."""
    print("=== Data Loading Example ===")
    
    # Create sample data
    training_data = create_sample_multilabel_data()
    
    # Save to CSV
    csv_path = "sample_multilabel_data.csv"
    DataLoader.save_to_csv(training_data, csv_path)
    print(f"Saved data to {csv_path}")
    
    # Load from CSV
    loaded_data = DataLoader.from_csv(
        csv_path,
        text_column='text',
        label_column='label',
        classification_type=ClassificationType.MULTI_LABEL
    )
    
    print(f"Loaded {len(loaded_data.texts)} samples from CSV")
    
    # Save to JSON
    json_path = "sample_multilabel_data.json"
    DataLoader.save_to_json(training_data, json_path)
    print(f"Saved data to {json_path}")
    
    # Load from JSON
    loaded_json_data = DataLoader.from_json(
        json_path,
        text_field='text',
        label_field='label',
        classification_type=ClassificationType.MULTI_LABEL
    )
    
    print(f"Loaded {len(loaded_json_data.texts)} samples from JSON")
    
    # Show statistics
    stats = get_data_statistics(loaded_data)
    print(f"\nData statistics:")
    print(f"  Classification type: {stats['classification_type']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Labels per sample: {stats['labels_per_sample']}")


def main():
    """Run all multi-label examples."""
    print("TextClassify Multi-label Classification Examples")
    print("=" * 50)
    
    try:
        # Data loading example
        example_data_loading()
        print()
        
        # LLM examples
        if os.getenv("OPENAI_API_KEY"):
            example_multilabel_llm()
            print()
        else:
            print("Skipping OpenAI example (no API key)")
            print()
        
        # RoBERTa example
        try:
            example_multilabel_roberta()
            print()
        except ImportError:
            print("Skipping RoBERTa example (transformers not installed)")
            print()
        
        # Ensemble examples
        if os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"):
            example_multilabel_ensemble()
            print()
        
        if os.getenv("OPENAI_API_KEY") and os.getenv("GEMINI_API_KEY"):
            example_class_routing_ensemble()
        else:
            print("Skipping class routing example (requires multiple API keys)")
    
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure to set your API keys as environment variables:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export GEMINI_API_KEY='your-key-here'")


if __name__ == "__main__":
    main()

