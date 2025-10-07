"""Test script demonstrating LLM prediction caching and recovery capabilities."""

import pandas as pd
import numpy as np
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig

def test_llm_caching():
    """Test LLM caching functionality with sample data."""
    
    print("🚀 Testing LLM Prediction Caching System")
    print("=" * 50)
    
    # Create sample data
    train_data = {
        'text': [
            "I love this product, it's amazing!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special.",
            "Excellent quality and fast shipping.",
            "Poor quality, would not recommend.",
            "Average product, meets expectations."
        ],
        'positive': [1, 0, 0, 1, 0, 0],
        'negative': [0, 1, 0, 0, 1, 0],
        'neutral': [0, 0, 1, 0, 0, 1]
    }
    
    test_data = {
        'text': [
            "Great product, highly recommended!",
            "Disappointing purchase, waste of money.",
            "It's fine, does what it's supposed to do."
        ],
        'positive': [1, 0, 0],
        'negative': [0, 1, 0], 
        'neutral': [0, 0, 1]
    }
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print(f"📊 Training data: {len(train_df)} samples")
    print(f"📊 Test data: {len(test_df)} samples")
    
    # Create model configuration
    config = ModelConfig(
        parameters={
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'batch_size': 64,  # Use stratified sampling
            'max_completion_tokens': 100
        }
    )
    
    # Create classifier with caching enabled
    print(f"\n🤖 Creating OpenAI classifier with caching...")
    classifier = OpenAIClassifier(
        config=config,
        text_column='text',
        label_columns=['positive', 'negative', 'neutral'],
        multi_label=False,
        enable_cache=True,
        cache_dir="cache/tests"
    )
    
    # Check if cache is enabled
    print(f"💾 Cache enabled: {classifier.enable_cache}")
    if classifier.cache:
        print(f"📁 Cache directory: {classifier.cache.cache_dir}")
        print(f"🆔 Session ID: {classifier.cache.session_id}")
    
    try:
        # Run prediction (this will populate the cache)
        print(f"\n🔮 Running first prediction (will cache results)...")
        result1 = classifier.predict(train_df=train_df, test_df=test_df)
        
        print(f"✅ First prediction completed!")
        print(f"📋 Predictions: {result1.predictions}")
        
        # Get cache statistics
        if classifier.cache:
            cache_stats = classifier.get_cache_stats()
            print(f"\n📊 Cache Statistics:")
            print(f"   Total predictions: {cache_stats['total_predictions']}")
            print(f"   Successful: {cache_stats['successful_predictions']}")
            print(f"   Failed: {cache_stats['failed_predictions']}")
            print(f"   Cache size: {cache_stats['cache_size_mb']:.2f} MB")
            print(f"   Last updated: {cache_stats['last_updated']}")
        
        # Run prediction again (should use cache)
        print(f"\n🔮 Running second prediction (should use cache)...")
        result2 = classifier.predict(train_df=train_df, test_df=test_df)
        
        print(f"✅ Second prediction completed!")
        print(f"📋 Predictions: {result2.predictions}")
        
        # Verify results are the same
        if result1.predictions == result2.predictions:
            print(f"✅ Cache working correctly - results are identical!")
        else:
            print(f"⚠️  Results differ - cache may not be working as expected")
        
        # Export cache
        print(f"\n💾 Exporting cache to CSV...")
        classifier.export_cache("test_cache_export.csv", format="csv")
        print(f"📄 Cache exported to test_cache_export.csv")
        
        # Final cache stats
        final_stats = classifier.get_cache_stats()
        print(f"\n📊 Final Cache Statistics:")
        print(f"   Session: {final_stats['session_id']}")
        print(f"   Total predictions: {final_stats['total_predictions']}")
        print(f"   Cache directory: {final_stats['cache_directory']}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        
        # Even if prediction fails, we can check cache
        if classifier.cache:
            failed_predictions = classifier.cache.get_failed_predictions()
            print(f"💥 Failed predictions in cache: {len(failed_predictions)}")
            for i, failed in enumerate(failed_predictions[:3]):  # Show first 3
                print(f"   {i+1}. Error: {failed['error_message']}")
    
    print(f"\n🎯 Testing completed!")
    print(f"🔍 Check the 'cache/tests' directory for cached files")
    print(f"📄 Check 'test_cache_export.csv' for exported results")

if __name__ == "__main__":
    test_llm_caching()