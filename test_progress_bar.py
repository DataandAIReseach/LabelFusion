"""Test script to verify progress bar functionality."""

import pandas as pd
import asyncio
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, ModelType

async def test_progress_bar():
    """Test the enhanced progress bar functionality."""
    
    print("🧪 Testing LLM Progress Bar...")
    print("=" * 50)
    
    # Create test data
    test_data = {
        'text': [
            "I love this product, it's amazing!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special.",
            "Excellent quality and fast delivery.",
            "Poor quality, would not recommend.",
            "Average product, meets expectations.",
            "Great customer service experience.",
            "Disappointed with the quality.",
            "Exceeds my expectations completely.",
            "Not worth the money spent."
        ],
        'positive': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        'negative': [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        'neutral': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(test_data)
    print(f"📊 Created test dataset with {len(df)} samples")
    
    # Create model configuration
    config = ModelConfig(
        model_name="gpt-3.5-turbo",
        model_type=ModelType.LLM,
        parameters={
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'batch_size': 3,  # Small batches to see multiple progress updates
            'max_completion_tokens': 50
        }
    )
    
    # Create classifier with caching enabled
    classifier = OpenAIClassifier(
        config=config,
        text_column='text',
        label_columns=['positive', 'negative', 'neutral'],
        multi_label=False,
        enable_cache=True,
        cache_dir="cache/tests/progress_bar_test"
    )
    
    # Ensure verbose mode is enabled
    classifier.verbose = True
    
    print(f"🤖 Created classifier with cache enabled")
    print(f"📁 Cache directory: {classifier.cache.cache_dir}")
    
    # Mock the LLM call to avoid API costs during testing
    async def mock_llm_call(prompt):
        """Mock LLM call that simulates realistic API delay."""
        await asyncio.sleep(1.0)  # Simulate 1 second API delay
        return "positive"  # Return simple response
    
    # Replace the actual LLM call with our mock
    classifier._call_llm = mock_llm_call
    
    print(f"\n🔮 Starting prediction with mocked LLM calls...")
    print(f"⏱️  Each prediction will take ~1 second (simulated API delay)")
    print(f"🎯 Watch for progress bar below:\n")
    
    try:
        # Test the prediction generation with progress bar
        predictions = await classifier._generate_predictions(df, 'text')
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Generated {len(predictions)} predictions")
        
        # Test cache functionality
        print(f"\n🔄 Testing cache hits (should be much faster)...")
        predictions2 = await classifier._generate_predictions(df, 'text')
        
        print(f"✅ Cache test completed!")
        print(f"📊 Second run generated {len(predictions2)} predictions")
        
        # Show cache statistics
        if classifier.cache:
            stats = classifier.get_cache_stats()
            print(f"\n📊 Final Cache Statistics:")
            print(f"   Total predictions: {stats['total_predictions']}")
            print(f"   Successful: {stats['successful_predictions']}")
            print(f"   Failed: {stats['failed_predictions']}")
            print(f"   Cache size: {stats['cache_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_progress_bar())