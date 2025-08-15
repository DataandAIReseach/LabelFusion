#!/usr/bin/env python3
"""
Test AutoFusion classifier with Kaggle ecommerce dataset.

This test demonstrates the simplified AutoFusion interface using real data,
showing how the config is passed directly to the fusion class.
"""

import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import textclassify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from textclassify.ensemble.auto_fusion import AutoFusionClassifier

# Load environment variables
load_dotenv()

def prepare_kaggle_data():
    """Load and prepare the Kaggle ecommerce dataset."""
    print("📂 Loading Kaggle ecommerce dataset...")
    
    # Load data
    df = pd.read_csv('data/ecommerceDataset.csv', encoding='latin1')
    print(f"   📊 Original dataset shape: {df.shape}")
    
    # Drop first column and reorder/rename columns
    cols = df.columns.tolist()
    df = df[[cols[1], cols[0]]]  # Swap columns
    df.columns = ['text', 'label']  # Rename columns
    
    # Convert label column to dummy variables
    label_dummies = pd.get_dummies(df['label'], prefix='')
    df = pd.concat([df[['text']], label_dummies], axis=1)
    
    # Get label columns
    label_columns = label_dummies.columns.tolist()
    print(f"   🏷️  Available labels: {label_columns}")
    
    # Shuffle and split data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Use more data for AutoFusion test (20 for training, 10 for testing)
    train_df = df_shuffled.iloc[:20]
    test_df = df_shuffled.iloc[20:25]
    
    print(f"   ✅ Training set: {len(train_df)} samples")
    print(f"   ✅ Test set: {len(test_df)} samples")
    
    # Verify DataFrame structure
    print(f"   📋 Train DataFrame columns: {train_df.columns.tolist()}")
    print(f"   📋 Test DataFrame columns: {test_df.columns.tolist()}")
    print(f"   📋 Test DataFrame shape: {test_df.shape}")
    print(f"   📋 Test DataFrame sample:")
    print(f"       Text: {test_df['text'].iloc[0][:50]}...")
    print(f"       Labels: {[col for col in label_columns if test_df[col].iloc[0] == 1]}")
    
    return train_df, test_df, label_columns

def test_autofusion_with_kaggle():
    """Test AutoFusion classifier with Kaggle dataset."""
    print("\n🚀 TESTING AUTOFUSION WITH KAGGLE DATASET")
    print("=" * 60)
    
    # Prepare data
    train_df, test_df, label_columns = prepare_kaggle_data()
    
    # Create AutoFusion configuration (this is what the CLI would create)
    config = {
        'llm_provider': 'deepseek',  # Using DeepSeek as it's the most stable
        'label_columns': label_columns,
        'multi_label': False,  # Multi-class classification
        'text_column': 'text',
        'batch_size': 4,  # Small batch for quick testing
        'num_epochs': 2,  # Reduced epochs for quick testing
        'fusion_epochs': 3,  # Reduced fusion epochs
        'output_dir': 'autofusion_kaggle_test',
        'ml_model': 'roberta-base',
        'max_length': 256  # Shorter for quick testing
    }
    
    print("\n🔧 AutoFusion Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Create AutoFusion classifier (config passed as argument)
        print("\n🤖 Creating AutoFusion classifier...")
        classifier = AutoFusionClassifier(config=config)
        
        print("\n✅ AutoFusion classifier created successfully!")
        print(f"   📊 LLM Provider: {classifier.llm_provider}")
        print(f"   🏷️  Labels: {classifier.label_columns}")
        print(f"   📝 Classification Type: {classifier.classification_type}")
        print(f"   🤖 ML Model: {classifier.ml_model_name}")
        
        # Train the classifier
        print("\n🎯 Training AutoFusion classifier...")
        print("   This will automatically:")
        print("   1. Set up RoBERTa ML model")
        print("   2. Set up DeepSeek LLM model") 
        print("   3. Train fusion ensemble (including ML training)")
        
        classifier.fit(train_df)
        
        print("\n✅ AutoFusion training completed!")
        
        # Make predictions on test data
        print("\n📊 Making predictions on test data...")
        result = classifier.predict(test_df)
        
        # Show results
        print("\n📋 Sample Predictions:")
        print("-" * 80)
        for i in range(min(5, len(test_df))):
            text = test_df['text'].iloc[i]
            pred = result.predictions[i]
            
            # Get true labels
            true_labels = []
            for label in label_columns:
                if test_df[label].iloc[i] == 1:
                    true_labels.append(label)
            
            print(f"\nSample {i + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"True Label: {true_labels}")
            print(f"Prediction: {pred}")
        
        # Show metrics if available
        if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
            metrics = result.metadata['metrics']
            print("\n📈 Performance Metrics:")
            print("-" * 40)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
        
        # Show fusion details
        if hasattr(classifier, 'fusion_ensemble') and classifier.fusion_ensemble:
            print("\n🔬 Fusion Ensemble Details:")
            print(f"   Models in ensemble: {len(classifier.fusion_ensemble.models)}")
            print(f"   Fusion method: {type(classifier.fusion_ensemble).__name__}")
        
        print("\n✅ AutoFusion test completed successfully!")
        return result
        
    except Exception as e:
        print(f"\n❌ Error during AutoFusion test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_individual_models():
    """Compare AutoFusion with individual ML model performance."""
    print("\n🔍 COMPARISON WITH INDIVIDUAL ML MODEL")
    print("=" * 50)
    
    # This would test just the ML model for comparison
    print("ℹ️  Individual model comparison can be run separately")
    print("   See test_kaggle_data.py for ML-only testing")

def main():
    """Main function to run AutoFusion test with Kaggle data."""
    print("🧪 AUTOFUSION KAGGLE DATASET TEST")
    print("=" * 60)
    print("Testing the simplified AutoFusion interface with real ecommerce data")
    print("This demonstrates how config is passed directly to the fusion class")
    
    # Check if data file exists
    if not os.path.exists('data/ecommerceDataset.csv'):
        print("❌ Error: data/ecommerceDataset.csv not found")
        print("   Please ensure the Kaggle dataset is in the data/ directory")
        return
    
    # Run AutoFusion test
    result = test_autofusion_with_kaggle()
    
    # Show additional information
    compare_with_individual_models()
    
    if result:
        print("\n🎉 Test completed successfully!")
        print("   AutoFusion successfully combined ML and LLM models")
        print("   Configuration was properly passed to the fusion class")
    else:
        print("\n⚠️  Test encountered issues")

if __name__ == "__main__":
    main()
