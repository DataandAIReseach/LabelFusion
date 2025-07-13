#!/usr/bin/env python3
"""
🎯 TextClassify Demo Script

A simple demonstration of TextClassify's capabilities for sentiment analysis.
This script shows basic usage patterns and can be run without API keys using mock data.

Run with: python demo.py
"""

import time
from typing import List, Dict
from textclassify import (
    OpenAIClassifier, ClaudeClassifier, GeminiClassifier,
    VotingEnsemble, ModelConfig, ModelType, ClassificationType,
    EnsembleConfig
)
from textclassify.utils import DataLoader, evaluate_predictions


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")


def print_results(results, title: str = "Results"):
    """Print classification results in a formatted way."""
    print(f"\n📊 {title}:")
    for i, (pred, conf) in enumerate(zip(results.predictions, results.confidence_scores)):
        print(f"   {i+1}. {pred} (confidence: {conf:.2f})")


def demo_basic_classification():
    """Demonstrate basic multi-class sentiment classification."""
    print_header("Basic Sentiment Classification")
    
    # Sample data for demonstration
    sample_texts = [
        "I absolutely love this product! It's amazing! 🌟",
        "This is the worst thing I've ever bought. Terrible quality. 😞",
        "It's okay, nothing special but does the job. 😐",
        "Outstanding quality and fast delivery! Highly recommend! 🚀",
        "Poor customer service and broken on arrival. 😠"
    ]
    
    # Create training data
    training_texts = [
        "I love this movie! Great acting and plot.",
        "Terrible film, waste of time and money.",
        "It was okay, not bad but not great either.",
        "Amazing cinematography and soundtrack!",
        "Boring and predictable storyline."
    ]
    training_labels = ["positive", "negative", "neutral", "positive", "negative"]
    
    training_data = DataLoader.from_lists(
        texts=training_texts,
        labels=training_labels,
        classification_type=ClassificationType.MULTI_CLASS
    )
    
    print("📝 Training data created:")
    print(f"   - {len(training_texts)} samples")
    print(f"   - Classes: {set(training_labels)}")
    
    # Note: In a real scenario, you would set up API keys
    print("\n🤖 Classifier setup (mock mode for demo):")
    config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        temperature=0.0
    )
    
    # In demo mode, we'll simulate results
    print("   ⚠️  Demo mode: Using simulated results")
    print("   💡 Set API keys to use real LLM models")
    
    # Simulate classification results
    mock_predictions = ["positive", "negative", "neutral", "positive", "negative"]
    mock_confidences = [0.95, 0.88, 0.62, 0.91, 0.84]
    
    class MockResult:
        def __init__(self, predictions, confidence_scores):
            self.predictions = predictions
            self.confidence_scores = confidence_scores
            self.metadata = {"model": "gpt-3.5-turbo", "processing_time": 2.5}
    
    results = MockResult(mock_predictions, mock_confidences)
    print_results(results, "Classification Results")
    
    print(f"\n⏱️  Processing time: {results.metadata['processing_time']:.1f}s")
    print(f"🤖 Model used: {results.metadata['model']}")


def demo_multi_label_classification():
    """Demonstrate multi-label classification for movie genres."""
    print_header("Multi-Label Movie Genre Classification")
    
    # Sample movie descriptions
    movie_descriptions = [
        "Action-packed superhero film with romantic subplot",
        "Hilarious comedy with family-friendly adventure",
        "Terrifying horror movie with supernatural elements",
        "Romantic drama set in World War II",
        "Sci-fi thriller with dystopian future setting"
    ]
    
    # Simulate multi-label results
    mock_labels = [
        ["action", "romance"],
        ["comedy", "adventure", "family"],
        ["horror", "supernatural"],
        ["romance", "drama", "war"],
        ["sci-fi", "thriller", "dystopian"]
    ]
    
    print("🎬 Movie descriptions for classification:")
    for i, desc in enumerate(movie_descriptions):
        print(f"   {i+1}. {desc}")
    
    print("\n🏷️  Predicted genres:")
    for i, (desc, labels) in enumerate(zip(movie_descriptions, mock_labels)):
        print(f"   {i+1}. {', '.join(labels)}")


def demo_ensemble_methods():
    """Demonstrate ensemble classification methods."""
    print_header("Ensemble Methods Demonstration")
    
    # Simulate individual model predictions
    models_predictions = {
        "OpenAI GPT-4": ["positive", "negative", "neutral", "positive", "negative"],
        "Claude Sonnet": ["positive", "negative", "positive", "positive", "negative"],
        "Gemini Pro": ["positive", "negative", "neutral", "positive", "neutral"]
    }
    
    test_texts = [
        "This product exceeded my expectations!",
        "Completely disappointed with the quality.",
        "Average product, nothing special.",
        "Fantastic service and quick delivery!",
        "Poor design and overpriced."
    ]
    
    print("🎭 Individual model predictions:")
    for model, predictions in models_predictions.items():
        print(f"\n   {model}:")
        for i, pred in enumerate(predictions):
            print(f"      {i+1}. {pred}")
    
    # Simulate ensemble voting
    print("\n🗳️  Ensemble voting results:")
    ensemble_predictions = []
    
    for i in range(len(test_texts)):
        votes = [models_predictions[model][i] for model in models_predictions]
        # Simple majority voting simulation
        vote_counts = {label: votes.count(label) for label in set(votes)}
        winner = max(vote_counts, key=vote_counts.get)
        ensemble_predictions.append(winner)
        
        print(f"   {i+1}. Votes: {votes} → Ensemble: {winner}")
    
    print(f"\n📈 Ensemble typically improves accuracy by 2-5% over individual models")


def demo_performance_metrics():
    """Demonstrate performance evaluation."""
    print_header("Performance Metrics Demonstration")
    
    # Simulate true labels and predictions
    true_labels = ["positive", "negative", "neutral", "positive", "negative"]
    predictions = ["positive", "negative", "positive", "positive", "negative"]
    
    print("📊 Performance evaluation:")
    print(f"   True labels:  {true_labels}")
    print(f"   Predictions:  {predictions}")
    
    # Calculate basic metrics
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    accuracy = correct / len(true_labels)
    
    print(f"\n📈 Metrics:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Correct predictions: {correct}/{len(true_labels)}")
    
    # Show per-class breakdown
    classes = set(true_labels)
    print(f"\n🏷️  Per-class performance:")
    for cls in classes:
        true_count = true_labels.count(cls)
        pred_count = predictions.count(cls)
        correct_count = sum(1 for t, p in zip(true_labels, predictions) 
                          if t == cls and p == cls)
        
        precision = correct_count / pred_count if pred_count > 0 else 0
        recall = correct_count / true_count if true_count > 0 else 0
        
        print(f"   {cls:>8}: Precision={precision:.1%}, Recall={recall:.1%}")


def demo_configuration_options():
    """Demonstrate configuration and setup options."""
    print_header("Configuration Options")
    
    print("⚙️  Available configuration options:")
    
    config_examples = {
        "Temperature": "0.0 (deterministic) to 1.0 (creative)",
        "Max tokens": "50 (short) to 2000 (detailed responses)",
        "Timeout": "10s (fast) to 60s (patient)",
        "Batch size": "8 (memory efficient) to 64 (throughput)",
        "Async processing": "True (parallel) vs False (sequential)",
        "Caching": "Memory, Redis, or Disk caching"
    }
    
    for option, description in config_examples.items():
        print(f"   📋 {option:>15}: {description}")
    
    print("\n🔑 API Key setup options:")
    print("   1. Environment variables (recommended)")
    print("   2. Configuration files (YAML/JSON)")
    print("   3. Programmatic setup (for dynamic keys)")
    
    print("\n💡 Performance tips:")
    tips = [
        "Use async processing for >100 texts",
        "Enable caching for repeated classifications",
        "Choose appropriate models for speed vs accuracy tradeoff",
        "Use ensemble methods for critical applications",
        "Monitor API costs with built-in tracking"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"   {i}. {tip}")


def main():
    """Run the complete demonstration."""
    print("🚀 Welcome to TextClassify Demo!")
    print("This demonstration shows the key features and capabilities.")
    
    try:
        # Run all demonstrations
        demo_basic_classification()
        demo_multi_label_classification()
        demo_ensemble_methods()
        demo_performance_metrics()
        demo_configuration_options()
        
        print_header("Demo Complete!")
        print("🎉 Thank you for trying TextClassify!")
        print("\n📖 Next steps:")
        print("   1. Install: pip install textclassify")
        print("   2. Set up API keys for your preferred LLM providers")
        print("   3. Check out examples/ folder for real implementations")
        print("   4. Read the full documentation in README.md")
        print("   5. Join our community on GitHub Discussions")
        
        print("\n🔗 Useful links:")
        print("   📖 Documentation: https://github.com/your-org/textclassify")
        print("   🐛 Issues: https://github.com/your-org/textclassify/issues")
        print("   💬 Discussions: https://github.com/your-org/textclassify/discussions")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for trying TextClassify!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("This is a demonstration script with simulated results.")


if __name__ == "__main__":
    main()
