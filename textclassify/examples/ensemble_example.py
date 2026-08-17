"""
Advanced Ensemble Methods Example

This example demonstrates advanced ensemble strategies for combining
multiple classifiers to optimize performance.
"""

import os
import numpy as np
from textclassify import (
    OpenAIClassifier, ClaudeClassifier, GeminiClassifier, DeepSeekClassifier,
    RoBERTaClassifier, VotingEnsemble, WeightedEnsemble, ClassRoutingEnsemble,
    Config, ClassificationType, ModelType
)
from textclassify.utils import evaluate_predictions, compare_models, split_data
from textclassify.core.types import TrainingData, ModelConfig, EnsembleConfig


def create_comprehensive_dataset():
    """Create a more comprehensive dataset for ensemble testing."""
    texts = [
        # Technology
        "The new smartphone features advanced AI capabilities and 5G connectivity.",
        "Machine learning algorithms are revolutionizing data analysis.",
        "Cloud computing provides scalable infrastructure for businesses.",
        
        # Sports
        "The basketball team won the championship with an amazing final shot.",
        "The soccer match ended in a thrilling penalty shootout.",
        "Olympic athletes train for years to compete at the highest level.",
        
        # Entertainment
        "The movie premiere was attended by many Hollywood celebrities.",
        "The concert featured incredible live performances and stage effects.",
        "The TV series finale broke viewership records.",
        
        # Business
        "The company reported record profits in the quarterly earnings call.",
        "Stock markets showed volatility due to economic uncertainty.",
        "The startup secured significant funding from venture capitalists.",
        
        # Health
        "Regular exercise and healthy diet are essential for wellbeing.",
        "Medical research has led to breakthrough treatments for diseases.",
        "Mental health awareness is becoming increasingly important.",
        
        # Education
        "Online learning platforms have transformed modern education.",
        "University students are adapting to hybrid learning models.",
        "Educational technology is making learning more accessible."
    ]
    
    labels = [
        "technology", "technology", "technology",
        "sports", "sports", "sports",
        "entertainment", "entertainment", "entertainment",
        "business", "business", "business",
        "health", "health", "health",
        "education", "education", "education"
    ]
    
    return TrainingData(
        texts=texts,
        labels=labels,
        classification_type=ClassificationType.MULTI_CLASS
    )


def create_individual_classifiers():
    """Create individual classifiers for ensemble."""
    classifiers = {}
    
    # LLM Classifiers
    llm_configs = {
        "openai": {
            "model_name": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "classifier_class": OpenAIClassifier
        },
        "claude": {
            "model_name": "claude-3-haiku-20240307",
            "api_key": os.getenv("CLAUDE_API_KEY"),
            "classifier_class": ClaudeClassifier
        },
        "gemini": {
            "model_name": "gemini-1.5-flash",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "classifier_class": GeminiClassifier
        },
        "deepseek": {
            "model_name": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "classifier_class": DeepSeekClassifier
        }
    }
    
    for name, config_info in llm_configs.items():
        if config_info["api_key"]:
            config = ModelConfig(
                model_name=config_info["model_name"],
                model_type=ModelType.LLM,
                api_key=config_info["api_key"],
                parameters={
                    "temperature": 0.1,
                    "max_tokens": 50,
                    "max_examples": 3
                }
            )
            classifiers[name] = config_info["classifier_class"](config)
    
    # RoBERTa Classifier (if transformers available)
    try:
        roberta_config = ModelConfig(
            model_name="roberta-base",
            model_type=ModelType.TRADITIONAL_ML,
            parameters={
                "max_length": 128,
                "batch_size": 8,
                "num_epochs": 2,
                "learning_rate": 2e-5
            }
        )
        classifiers["roberta"] = RoBERTaClassifier(roberta_config)
    except ImportError:
        print("RoBERTa not available (transformers not installed)")
    
    return classifiers


def example_model_comparison():
    """Compare individual models before ensembling."""
    print("=== Individual Model Comparison ===")
    
    # Create dataset
    training_data = create_comprehensive_dataset()
    train_data, test_data = split_data(training_data, train_ratio=0.7)
    
    print(f"Training samples: {len(train_data.texts)}")
    print(f"Test samples: {len(test_data.texts)}")
    print()
    
    # Create classifiers
    classifiers = create_individual_classifiers()
    
    if not classifiers:
        print("No classifiers available (missing API keys or dependencies)")
        return None, None, None
    
    print(f"Available classifiers: {list(classifiers.keys())}")
    
    # Train and evaluate each classifier
    results = []
    model_names = []
    
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        try:
            classifier.fit(train_data)
            result = classifier.predict(test_data.texts)
            results.append(result)
            model_names.append(name)
            
            # Quick evaluation
            metrics = evaluate_predictions(result, test_data.labels)
            print(f"{name} accuracy: {metrics['accuracy']:.3f}")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
    
    # Compare all models
    if len(results) > 1:
        comparison = compare_models(results, test_data.labels, model_names)
        
        print("\n=== Model Comparison Summary ===")
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            print(f"\n{metric.upper()}:")
            for model, score in comparison['summary'][metric].items():
                print(f"  {model}: {score:.3f}")
            print(f"  Best: {comparison['best_models'][metric]}")
    
    return classifiers, train_data, test_data


def example_voting_ensemble(classifiers, train_data, test_data):
    """Demonstrate voting ensemble with different strategies."""
    print("\n=== Voting Ensemble Strategies ===")
    
    if len(classifiers) < 2:
        print("Need at least 2 classifiers for ensemble")
        return
    
    # Test different voting strategies
    voting_strategies = ['majority', 'plurality']
    
    for strategy in voting_strategies:
        print(f"\n--- {strategy.upper()} Voting ---")
        
        # Create ensemble config
        model_configs = []
        for name, classifier in classifiers.items():
            model_configs.append(classifier.config)
        
        ensemble_config = EnsembleConfig(
            models=model_configs,
            ensemble_method=strategy
        )
        
        # Create ensemble
        ensemble = VotingEnsemble(ensemble_config)
        
        # Add classifiers
        for name, classifier in classifiers.items():
            ensemble.add_model(classifier, name)
        
        # Train and test
        ensemble.fit(train_data)
        result = ensemble.predict(test_data.texts)
        
        # Evaluate
        metrics = evaluate_predictions(result, test_data.labels)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1: {metrics['macro_f1']:.3f}")


def example_weighted_ensemble(classifiers, train_data, test_data):
    """Demonstrate weighted ensemble with performance-based weights."""
    print("\n=== Weighted Ensemble ===")
    
    if len(classifiers) < 2:
        print("Need at least 2 classifiers for ensemble")
        return
    
    # Calculate individual model performance to determine weights
    model_performances = {}
    
    for name, classifier in classifiers.items():
        # Use a small validation set to calculate weights
        val_result = classifier.predict(test_data.texts[:5])  # Small sample
        val_metrics = evaluate_predictions(val_result, test_data.labels[:5])
        model_performances[name] = val_metrics['accuracy']
    
    print("Individual model performances:")
    for name, perf in model_performances.items():
        print(f"  {name}: {perf:.3f}")
    
    # Calculate weights based on performance
    total_perf = sum(model_performances.values())
    weights = [model_performances[name] / total_perf for name in classifiers.keys()]
    
    print(f"Calculated weights: {[f'{w:.3f}' for w in weights]}")
    
    # Create weighted ensemble
    model_configs = [classifier.config for classifier in classifiers.values()]
    ensemble_config = EnsembleConfig(
        models=model_configs,
        ensemble_method="weighted",
        weights=weights
    )
    
    ensemble = WeightedEnsemble(ensemble_config)
    
    # Add classifiers with weights
    for (name, classifier), weight in zip(classifiers.items(), weights):
        ensemble.add_model(classifier, name, weight)
    
    # Train and test
    ensemble.fit(train_data)
    result = ensemble.predict_proba(test_data.texts)
    
    # Evaluate
    metrics = evaluate_predictions(result, test_data.labels)
    print(f"\nWeighted ensemble performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    
    # Show final weights
    print(f"Final weights: {[f'{w:.3f}' for w in ensemble.weights]}")


def example_class_routing(classifiers, train_data, test_data):
    """Demonstrate class-specific routing ensemble."""
    print("\n=== Class Routing Ensemble ===")
    
    if len(classifiers) < 2:
        print("Need at least 2 classifiers for routing")
        return
    
    # Define routing rules based on domain expertise
    # For example: technical topics to one model, creative topics to another
    classifier_names = list(classifiers.keys())
    
    routing_rules = {
        "technology": classifier_names[0],
        "business": classifier_names[0],
        "sports": classifier_names[1] if len(classifier_names) > 1 else classifier_names[0],
        "entertainment": classifier_names[1] if len(classifier_names) > 1 else classifier_names[0],
        "health": classifier_names[0],
        "education": classifier_names[0]
    }
    
    print("Routing rules:")
    for class_name, model_name in routing_rules.items():
        print(f"  {class_name} -> {model_name}")
    
    # Create routing ensemble
    model_configs = [classifier.config for classifier in classifiers.values()]
    ensemble_config = EnsembleConfig(
        models=model_configs,
        ensemble_method="routing",
        routing_rules=routing_rules
    )
    
    ensemble = ClassRoutingEnsemble(ensemble_config)
    
    # Add classifiers
    for name, classifier in classifiers.items():
        ensemble.add_model(classifier, name)
    
    # Train and test
    ensemble.fit(train_data)
    result = ensemble.predict(test_data.texts)
    
    # Evaluate
    metrics = evaluate_predictions(result, test_data.labels)
    print(f"\nClass routing performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    
    # Show which model was used for each prediction
    print(f"\nRouting summary:")
    routing_summary = ensemble.get_routing_summary()
    for class_name, model_name in routing_summary.items():
        print(f"  {class_name} -> {model_name}")


def example_ensemble_comparison(classifiers, train_data, test_data):
    """Compare different ensemble methods."""
    print("\n=== Ensemble Method Comparison ===")
    
    if len(classifiers) < 2:
        print("Need at least 2 classifiers for ensemble comparison")
        return
    
    ensemble_results = []
    ensemble_names = []
    
    # Voting ensemble
    try:
        model_configs = [classifier.config for classifier in classifiers.values()]
        voting_config = EnsembleConfig(models=model_configs, ensemble_method="majority")
        voting_ensemble = VotingEnsemble(voting_config)
        
        for name, classifier in classifiers.items():
            voting_ensemble.add_model(classifier, name)
        
        voting_ensemble.fit(train_data)
        voting_result = voting_ensemble.predict(test_data.texts)
        ensemble_results.append(voting_result)
        ensemble_names.append("voting")
    except Exception as e:
        print(f"Voting ensemble error: {e}")
    
    # Weighted ensemble
    try:
        weights = [1.0 / len(classifiers)] * len(classifiers)  # Equal weights
        weighted_config = EnsembleConfig(
            models=model_configs, 
            ensemble_method="weighted", 
            weights=weights
        )
        weighted_ensemble = WeightedEnsemble(weighted_config)
        
        for name, classifier in classifiers.items():
            weighted_ensemble.add_model(classifier, name)
        
        weighted_ensemble.fit(train_data)
        weighted_result = weighted_ensemble.predict(test_data.texts)
        ensemble_results.append(weighted_result)
        ensemble_names.append("weighted")
    except Exception as e:
        print(f"Weighted ensemble error: {e}")
    
    # Compare ensemble methods
    if len(ensemble_results) > 1:
        comparison = compare_models(ensemble_results, test_data.labels, ensemble_names)
        
        print("Ensemble comparison:")
        for metric in ['accuracy', 'macro_f1']:
            print(f"\n{metric.upper()}:")
            for method, score in comparison['summary'][metric].items():
                print(f"  {method}: {score:.3f}")
            print(f"  Best: {comparison['best_models'][metric]}")


def main():
    """Run all ensemble examples."""
    print("TextClassify Advanced Ensemble Examples")
    print("=" * 50)
    
    try:
        # Compare individual models
        classifiers, train_data, test_data = example_model_comparison()
        
        if not classifiers:
            print("No classifiers available. Please set API keys:")
            print("  export OPENAI_API_KEY='your-key-here'")
            print("  export CLAUDE_API_KEY='your-key-here'")
            print("  export GEMINI_API_KEY='your-key-here'")
            print("  export DEEPSEEK_API_KEY='your-key-here'")
            return
        
        # Ensemble examples
        example_voting_ensemble(classifiers, train_data, test_data)
        example_weighted_ensemble(classifiers, train_data, test_data)
        example_class_routing(classifiers, train_data, test_data)
        example_ensemble_comparison(classifiers, train_data, test_data)
        
        print("\n=== Summary ===")
        print("Ensemble methods can significantly improve classification performance")
        print("by combining the strengths of different models:")
        print("- Voting: Simple majority/plurality voting")
        print("- Weighted: Performance-based model weighting")
        print("- Routing: Class-specific model assignment")
        print("Choose the method that best fits your use case and data!")
    
    except Exception as e:
        print(f"Error running ensemble examples: {str(e)}")


if __name__ == "__main__":
    main()

