"""
Example: Using Cached LLM Predictions with FusionEnsemble

This example demonstrates how to use the new caching functions to avoid
regenerating LLM predictions every time you train or test the fusion ensemble.
"""

import pandas as pd
from pathlib import Path
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.core.types import EnsembleConfig, ModelConfig, ModelType, ClassificationType


def example_with_cache_usage():
    """Example showing how to use cached LLM predictions effectively."""
    
    # 1. Setup data (replace with your actual data loading)
    print("📊 Loading example data...")
    
    # Create example dataframes (replace with your actual data)
    train_data = pd.DataFrame({
        'text': ['This is great!', 'This is terrible!', 'This is okay.'] * 100,
        'positive': [1, 0, 0] * 100,
        'negative': [0, 1, 0] * 100,
        'neutral': [0, 0, 1] * 100
    })
    
    val_data = pd.DataFrame({
        'text': ['Amazing product!', 'Worst ever!', 'It\'s fine.'] * 50,
        'positive': [1, 0, 0] * 50,
        'negative': [0, 1, 0] * 50,
        'neutral': [0, 0, 1] * 50
    })
    
    test_data = pd.DataFrame({
        'text': ['Love it!', 'Hate it!', 'Neutral opinion.'] * 30,
        'positive': [1, 0, 0] * 30,
        'negative': [0, 1, 0] * 30,
        'neutral': [0, 0, 1] * 30
    })
    
    # 2. Setup models
    print("🤖 Setting up models...")
    
    # Create ML model (RoBERTa)
    ml_config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.TRANSFORMER,
        classification_type=ClassificationType.MULTI_CLASS,
        max_length=512,
        batch_size=8,
        num_epochs=2
    )
    ml_model = RoBERTaClassifier(ml_config)
    ml_model.setup_for_training(
        text_column='text',
        label_columns=['positive', 'negative', 'neutral']
    )
    
    # Create LLM model (DeepSeek)
    llm_config = ModelConfig(
        model_name="deepseek-chat",
        model_type=ModelType.LLM,
        classification_type=ClassificationType.MULTI_CLASS
    )
    llm_model = DeepSeekClassifier(llm_config)
    llm_model.setup_for_training(
        text_column='text',
        label_columns=['positive', 'negative', 'neutral']
    )
    
    # 3. Setup Fusion Ensemble with cache paths
    print("🔗 Setting up Fusion Ensemble with caching...")
    
    cache_dir = Path("./llm_cache")
    cache_dir.mkdir(exist_ok=True)
    
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            'val_llm_cache_path': str(cache_dir / "validation_predictions"),
            'test_llm_cache_path': str(cache_dir / "test_predictions"),
            'fusion_epochs': 10,
            'hidden_dims': [64, 32]
        }
    )
    
    fusion_ensemble = FusionEnsemble(ensemble_config)
    fusion_ensemble.add_ml_model(ml_model)
    fusion_ensemble.add_llm_model(llm_model)
    
    # 4. Check cache status before training
    print("\n🔍 Checking cache status before training...")
    fusion_ensemble.print_cache_status()
    
    # 5. Training with automatic cache usage
    print("\n🚀 Training with automatic cache loading...")
    
    # Method 1: Automatic cache loading (recommended)
    # This will automatically try to load cached validation predictions
    training_results = fusion_ensemble.fit_with_cached_predictions(
        train_df=train_data,
        val_df=val_data,
        force_load_from_cache=False  # Set to True to require cached predictions
    )
    
    # Alternative Method 2: Manual cache loading
    # cached_val_predictions = fusion_ensemble.load_cached_predictions_for_dataset(
    #     val_data, dataset_type="validation"
    # )
    # if cached_val_predictions:
    #     training_results = fusion_ensemble.fit(train_data, val_data, cached_val_predictions)
    # else:
    #     training_results = fusion_ensemble.fit(train_data, val_data)
    
    # 6. Check cache status after training
    print("\n📈 Cache status after training:")
    fusion_ensemble.print_cache_status()
    
    # 7. Prediction with automatic cache usage
    print("\n🔮 Making predictions with automatic cache loading...")
    
    # Method 1: Automatic cache loading (recommended)
    test_results = fusion_ensemble.predict_with_cached_predictions(
        test_df=test_data,
        force_load_from_cache=False  # Set to True to require cached predictions
    )
    
    # Alternative Method 2: Manual cache loading
    # cached_test_predictions = fusion_ensemble.load_cached_predictions_for_dataset(
    #     test_data, dataset_type="test"
    # )
    # test_results = fusion_ensemble.predict(test_data, test_llm_predictions=cached_test_predictions)
    
    print(f"✅ Test accuracy: {test_results.metadata.get('accuracy', 'N/A')}")
    
    return fusion_ensemble, training_results, test_results


def example_cache_discovery():
    """Example showing how to discover and inspect cached predictions."""
    
    print("\n🔍 CACHE DISCOVERY EXAMPLE")
    print("="*50)
    
    # Discover all cached prediction files in a directory
    cache_directory = "./llm_cache"
    cached_files = FusionEnsemble.discover_cached_predictions(cache_directory)
    
    if cached_files:
        print(f"Found cached prediction files in {cache_directory}:")
        for base_name, files in cached_files.items():
            print(f"\n📁 {base_name}:")
            for i, file_path in enumerate(files):
                file_name = Path(file_path).name
                creation_time = Path(file_path).stat().st_ctime
                print(f"   {i+1}. {file_name} (created: {creation_time})")
    else:
        print(f"No cached prediction files found in {cache_directory}")


def example_forced_cache_usage():
    """Example showing how to enforce cache usage and handle errors."""
    
    print("\n🔒 FORCED CACHE USAGE EXAMPLE")
    print("="*50)
    
    # This example shows how to ensure that only cached predictions are used
    # and how to handle the case when cached predictions are not available
    
    # Setup (same as previous example)
    val_data = pd.DataFrame({
        'text': ['Test text'] * 10,
        'positive': [1] * 10,
        'negative': [0] * 10,
        'neutral': [0] * 10
    })
    
    # Create a fusion ensemble with cache paths
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[],
        parameters={
            'val_llm_cache_path': "./llm_cache/validation_predictions",
            'test_llm_cache_path': "./llm_cache/test_predictions"
        }
    )
    
    fusion_ensemble = FusionEnsemble(ensemble_config)
    
    try:
        # Try to load cached predictions with force=True
        cached_predictions = fusion_ensemble.load_cached_predictions_for_dataset(
            val_data, dataset_type="validation"
        )
        
        if cached_predictions:
            print(f"✅ Successfully loaded {len(cached_predictions)} cached predictions")
        else:
            print("❌ No cached predictions available")
            
    except Exception as e:
        print(f"⚠️ Error loading cached predictions: {e}")


if __name__ == "__main__":
    print("🚀 LLM Cache Usage Examples")
    print("="*60)
    
    # Run the main example
    try:
        fusion_ensemble, training_results, test_results = example_with_cache_usage()
        print("\n✅ Main example completed successfully!")
    except Exception as e:
        print(f"❌ Main example failed: {e}")
    
    # Run cache discovery example
    try:
        example_cache_discovery()
        print("\n✅ Cache discovery example completed!")
    except Exception as e:
        print(f"❌ Cache discovery example failed: {e}")
    
    # Run forced cache usage example
    try:
        example_forced_cache_usage()
        print("\n✅ Forced cache usage example completed!")
    except Exception as e:
        print(f"❌ Forced cache usage example failed: {e}")
    
    print("\n" + "="*60)
    print("📚 SUMMARY OF NEW CACHE FUNCTIONS:")
    print("="*60)
    print("1. load_cached_predictions_for_dataset() - Load cached predictions for a specific dataset")
    print("2. fit_with_cached_predictions() - Train with automatic cache loading")
    print("3. predict_with_cached_predictions() - Predict with automatic cache loading")
    print("4. get_cached_predictions_summary() - Get cache status information")
    print("5. print_cache_status() - Print detailed cache status")
    print("6. discover_cached_predictions() - Find all cached files in a directory")
    print("\n💡 These functions help you avoid regenerating LLM predictions and speed up experimentation!")