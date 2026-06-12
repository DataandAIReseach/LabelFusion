"""
Example: Using LLM Cache Management Functions

This script demonstrates how to use the new cache management functions
in LLM classifiers for fast prediction reuse and testing.
"""

import pandas as pd
from pathlib import Path
from textclassify.llm import OpenAIClassifier, GeminiClassifier, DeepSeekClassifier
from textclassify.config.model_config import ModelConfig


def example_1_discover_caches():
    """Example 1: Discover available cache files."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Discover Available Cache Files")
    print("="*60)
    
    # Use class method to discover caches without instantiating classifier
    discovered = OpenAIClassifier.discover_cached_predictions("cache")
    
    print(f"\nFound {len(discovered)} dataset types with cached predictions")
    for dataset_type, files in discovered.items():
        print(f"\n{dataset_type}:")
        for file_path in files[:3]:  # Show first 3
            print(f"  - {Path(file_path).name}")


def example_2_cache_status():
    """Example 2: Print comprehensive cache status."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Print Cache Status")
    print("="*60)
    
    # Create a classifier instance
    config = ModelConfig(
        model_type="llm",
        parameters={"model": "gpt-3.5-turbo"}
    )
    classifier = OpenAIClassifier(config)
    
    # Print status for default cache directory
    classifier.print_cache_status("cache")


def example_3_load_and_inspect():
    """Example 3: Load cache file and inspect contents."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Load and Inspect Cache File")
    print("="*60)
    
    config = ModelConfig(
        model_type="llm",
        parameters={"model": "gpt-3.5-turbo"}
    )
    classifier = OpenAIClassifier(config)
    
    # Discover caches
    discovered = OpenAIClassifier.discover_cached_predictions("cache")
    
    if not discovered:
        print("\nNo cache files found. Run predictions first to create cache files.")
        return
    
    # Get first available cache file
    first_dataset_type = list(discovered.keys())[0]
    cache_file = discovered[first_dataset_type][0]
    
    print(f"\nLoading cache file: {Path(cache_file).name}")
    
    # Get summary
    summary = classifier.get_cached_predictions_summary(cache_file)
    
    print("\nCache Summary:")
    print(f"  Provider: {summary.get('provider', 'unknown')}")
    print(f"  Model: {summary.get('model', 'unknown')}")
    print(f"  Total predictions: {summary.get('total_predictions', 0)}")
    print(f"  Timestamp: {summary.get('timestamp', 'unknown')}")
    
    if summary.get('has_metrics'):
        print("\n  Metrics:")
        metrics = summary.get('metrics', {})
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")


def example_4_reuse_cached_predictions():
    """Example 4: Reuse cached predictions for fast testing."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Reuse Cached Predictions")
    print("="*60)
    
    # Check if cache exists
    discovered = OpenAIClassifier.discover_cached_predictions("cache")
    
    if not discovered:
        print("\nNo cache files found.")
        print("Run predictions with caching enabled first:")
        print("   result = classifier.predict(train_df, test_df)")
        return
    
    # Get a test cache file
    if 'test_predictions' in discovered:
        cache_file = discovered['test_predictions'][0]
    elif 'validation_predictions' in discovered:
        cache_file = discovered['validation_predictions'][0]
    else:
        cache_file = list(discovered.values())[0][0]
    
    print(f"\nUsing cache file: {Path(cache_file).name}")
    
    # Load the cache to check structure
    config = ModelConfig(
        model_type="llm",
        parameters={"model": "gpt-3.5-turbo"}
    )
    classifier = OpenAIClassifier(config)
    
    cached_data = classifier.load_cached_predictions_for_dataset(cache_file)
    
    if not cached_data:
        print("\nCould not load cache file")
        return
    
    num_predictions = len(cached_data.get('predictions', []))
    print(f"\nLoaded {num_predictions} cached predictions")
    
    # Note: To actually use the predictions, you need a test_df with matching structure
    print("\nTo use cached predictions:")
    print("   result = classifier.predict_with_cached_predictions(test_df, cache_file)")
    print("   This is 1000-5000x faster than running inference!")


def example_5_compare_providers():
    """Example 5: Compare cached predictions from different providers."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Predictions from Different Providers")
    print("="*60)
    
    # Search for caches from different providers
    cache_dirs = {
        'openai': 'cache/fusion_openai_cache',
        'gemini': 'cache/fusion_gemini_cache',
        'deepseek': 'cache/fusion_deepseek_cache'
    }
    
    results = {}
    
    for provider_name, cache_dir in cache_dirs.items():
        if Path(cache_dir).exists():
            discovered = OpenAIClassifier.discover_cached_predictions(cache_dir)
            
            if discovered:
                # Get first available cache
                first_dataset = list(discovered.values())[0]
                cache_file = first_dataset[0]
                
                # Get summary
                config = ModelConfig(
                    model_type="llm",
                    parameters={"model": "dummy"}
                )
                classifier = OpenAIClassifier(config)
                summary = classifier.get_cached_predictions_summary(cache_file)
                
                results[provider_name] = summary
    
    if not results:
        print("\nNo provider cache directories found")
        print("Expected directories: cache/fusion_<provider>_cache/")
        return
    
    print(f"\nFound caches for {len(results)} provider(s)\n")
    
    # Display comparison
    for provider_name, summary in results.items():
        print(f"{provider_name.upper()}:")
        print(f"  Model: {summary.get('model', 'unknown')}")
        print(f"  Predictions: {summary.get('total_predictions', 0)}")
        
        if summary.get('has_metrics'):
            metrics = summary.get('metrics', {})
            if 'accuracy' in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if 'f1_macro' in metrics:
                print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        print()


def main():
    """Run all examples."""
    print("\nLLM Cache Management Examples")
    print("="*60)
    
    # Run examples
    try:
        example_1_discover_caches()
    except Exception as e:
        print(f"\nExample 1 error: {e}")
    
    try:
        example_2_cache_status()
    except Exception as e:
        print(f"\nExample 2 error: {e}")
    
    try:
        example_3_load_and_inspect()
    except Exception as e:
        print(f"\nExample 3 error: {e}")
    
    try:
        example_4_reuse_cached_predictions()
    except Exception as e:
        print(f"\nExample 4 error: {e}")
    
    try:
        example_5_compare_providers()
    except Exception as e:
        print(f"\nExample 5 error: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nSee docs/LLM_CACHE_MANAGEMENT.md for detailed documentation")


if __name__ == "__main__":
    main()
