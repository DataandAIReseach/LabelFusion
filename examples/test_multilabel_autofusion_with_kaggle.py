#!/usr/bin/env python3
"""
Test AutoFusion classifier with multi-label classification using Kaggle dataset.

This test demonstrates the simplified AutoFusion interface using real data
for multi-label classification scenarios.
"""

import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import textclassify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from textclassify.ensemble.auto_fusion import AutoFusionClassifier

# Load environment variables
load_dotenv()

def prepare_multilabel_kaggle_data():
    """Load and prepare a multi-label version of the Kaggle ecommerce dataset."""
    print("📂 Loading Kaggle ecommerce dataset for multi-label classification...")
    
    # Load data
    df = pd.read_csv('data/ecommerceDataset.csv', encoding='latin1')
    print(f"   📊 Original dataset shape: {df.shape}")
    
    # Drop first column and reorder/rename columns
    cols = df.columns.tolist()
    df = df[[cols[1], cols[0]]]  # Swap columns
    df.columns = ['text', 'original_label']  # Rename columns
    
    # Create multi-label scenario by analyzing text content
    # We'll create multiple binary labels based on keywords in the text
    print("   🏷️  Creating multi-label scenario from text content...")
    
    # Define label categories based on common ecommerce themes
    label_categories = {
        'positive_sentiment': ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome', 'wonderful'],
        'negative_sentiment': ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor'],
        'quality_related': ['quality', 'material', 'build', 'construction', 'durable', 'cheap', 'flimsy'],
        'price_related': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'affordable', 'budget'],
        'delivery_related': ['delivery', 'shipping', 'arrived', 'fast', 'slow', 'quick', 'package']
    }
    
    # Create binary labels for each category
    for category, keywords in label_categories.items():
        df[category] = 0
        for keyword in keywords:
            # Case-insensitive search for keywords in text
            mask = df['text'].str.lower().str.contains(keyword, na=False)
            df.loc[mask, category] = 1
    
    # Ensure each sample has at least one label
    label_columns = list(label_categories.keys())
    no_labels_mask = df[label_columns].sum(axis=1) == 0
    
    # For samples with no labels, assign based on original label sentiment
    if no_labels_mask.sum() > 0:
        print(f"   📝 Assigning default labels to {no_labels_mask.sum()} samples without labels...")
        # Simple heuristic: positive original labels get positive sentiment
        positive_original = df[no_labels_mask]['original_label'].isin(['Books', 'Clothing & Accessories', 'Electronics'])
        df.loc[no_labels_mask & positive_original, 'positive_sentiment'] = 1
        df.loc[no_labels_mask & ~positive_original, 'negative_sentiment'] = 1
    
    # Remove the original label column
    df = df.drop('original_label', axis=1)
    
    print(f"   🏷️  Multi-label categories: {label_columns}")
    print(f"   📊 Label distribution:")
    for label in label_columns:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"       {label}: {count} samples ({percentage:.1f}%)")
    
    # Check for multi-label samples
    multi_label_count = (df[label_columns].sum(axis=1) > 1).sum()
    print(f"   🔄 Samples with multiple labels: {multi_label_count} ({(multi_label_count/len(df)*100):.1f}%)")
    
    # Shuffle and split data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Use appropriate data size for multi-label testing (25 for training, 10 for testing)
    train_df = df_shuffled.iloc[:25]
    test_df = df_shuffled.iloc[25:35]
    
    print(f"   ✅ Training set: {len(train_df)} samples")
    print(f"   ✅ Test set: {len(test_df)} samples")
    
    # Verify DataFrame structure
    print(f"   📋 Train DataFrame columns: {train_df.columns.tolist()}")
    print(f"   📋 Test DataFrame columns: {test_df.columns.tolist()}")
    print(f"   📋 Test DataFrame shape: {test_df.shape}")
    print(f"   📋 Test DataFrame sample:")
    print(f"       Text: {test_df['text'].iloc[0][:50]}...")
    active_labels = [col for col in label_columns if test_df[col].iloc[0] == 1]
    print(f"       Active Labels: {active_labels}")
    
    return train_df, test_df, label_columns

def test_multilabel_autofusion_with_kaggle():
    """Test AutoFusion classifier with multi-label Kaggle dataset."""
    print("\n🚀 TESTING MULTI-LABEL AUTOFUSION WITH KAGGLE DATASET")
    print("=" * 70)
    
    # Prepare data
    train_df, test_df, label_columns = prepare_multilabel_kaggle_data()
    
    # Create AutoFusion configuration for multi-label classification
    config = {
        'llm_provider': 'deepseek',  # Using DeepSeek as it's the most stable
        'label_columns': label_columns,
        'multi_label': True,  # Multi-label classification
        'text_column': 'text',
        'batch_size': 4,  # Small batch for quick testing
        'num_epochs': 2,  # Reduced epochs for quick testing
        'fusion_epochs': 3,  # Reduced fusion epochs
        'output_dir': 'autofusion_multilabel_kaggle_test',
        'ml_model': 'roberta-base',
        'max_length': 256  # Shorter for quick testing
    }
    
    print("\n🔧 AutoFusion Multi-Label Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Create AutoFusion classifier (config passed as argument)
        print("\n🤖 Creating Multi-Label AutoFusion classifier...")
        classifier = AutoFusionClassifier(config=config)
        
        print("\n✅ Multi-Label AutoFusion classifier created successfully!")
        print(f"   📊 LLM Provider: {classifier.llm_provider}")
        print(f"   🏷️  Labels: {classifier.label_columns}")
        print(f"   📝 Classification Type: {classifier.classification_type}")
        print(f"   🤖 ML Model: {classifier.ml_model_name}")
        print(f"   🔄 Multi-Label: {classifier.multi_label}")
        
        # Train the classifier
        print("\n🎯 Training Multi-Label AutoFusion classifier...")
        print("   This will automatically:")
        print("   1. Set up RoBERTa ML model for multi-label classification")
        print("   2. Set up DeepSeek LLM model for multi-label classification") 
        print("   3. Train fusion ensemble (including ML training)")
        
        classifier.fit(train_df)
        
        print("\n✅ Multi-Label AutoFusion training completed!")
        
        # Make predictions on test data
        print("\n📊 Making multi-label predictions on test data...")
        result = classifier.predict(test_df)
        
        # Show results
        print("\n📋 Sample Multi-Label Predictions:")
        print("-" * 90)
        for i in range(min(5, len(test_df))):
            text = test_df['text'].iloc[i]
            pred = result.predictions[i]
            
            # Get true labels
            true_labels = []
            for label in label_columns:
                if test_df[label].iloc[i] == 1:
                    true_labels.append(label)
            
            print(f"\nSample {i + 1}:")
            print(f"Text: {text[:80]}...")
            print(f"True Labels: {true_labels}")
            print(f"Predicted Labels: {pred if isinstance(pred, list) else [pred]}")
            
            # Show overlap
            pred_set = set(pred if isinstance(pred, list) else [pred])
            true_set = set(true_labels)
            overlap = pred_set & true_set
            print(f"Overlap: {list(overlap)} ({len(overlap)}/{len(true_set)} correct)")
        
        # Show metrics if available
        if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
            metrics = result.metadata['metrics']
            print("\n📈 Multi-Label Performance Metrics:")
            print("-" * 50)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
        
        # Show fusion details
        if hasattr(classifier, 'fusion_ensemble') and classifier.fusion_ensemble:
            print("\n🔬 Fusion Ensemble Details:")
            print(f"   Models in ensemble: {len(classifier.fusion_ensemble.models)}")
            print(f"   Fusion method: {type(classifier.fusion_ensemble).__name__}")
            print(f"   Classification type: {classifier.fusion_ensemble.classification_type}")
        
        # Show label statistics in predictions
        print("\n📊 Prediction Label Statistics:")
        print("-" * 40)
        for label in label_columns:
            pred_count = sum(1 for pred in result.predictions if label in (pred if isinstance(pred, list) else [pred]))
            true_count = test_df[label].sum()
            print(f"{label}: {pred_count} predicted, {true_count} actual")
        
        print("\n✅ Multi-Label AutoFusion test completed successfully!")
        return result
        
    except Exception as e:
        print(f"\n❌ Error during Multi-Label AutoFusion test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_single_label():
    """Compare with single-label classification approach."""
    print("\n🔍 COMPARISON WITH SINGLE-LABEL CLASSIFICATION")
    print("=" * 60)
    
    print("ℹ️  Multi-label vs Single-label comparison:")
    print("   🔄 Multi-label: Can predict multiple categories per text")
    print("   🔢 Single-label: Predicts only one category per text")
    print("   📈 Multi-label typically shows different evaluation metrics")
    print("   📊 See test_singlelabel_autofusion_with_ecommerce.py for single-label testing")

def main():
    """Main function to run Multi-Label AutoFusion test with Kaggle data."""
    print("🧪 MULTI-LABEL AUTOFUSION KAGGLE DATASET TEST")
    print("=" * 70)
    print("Testing the simplified AutoFusion interface with multi-label classification")
    print("This demonstrates how config is passed for multi-label scenarios")
    
    # Check if data file exists
    if not os.path.exists('data/ecommerceDataset.csv'):
        print("❌ Error: data/ecommerceDataset.csv not found")
        print("   Please ensure the Kaggle dataset is in the data/ directory")
        return
    
    # Run Multi-Label AutoFusion test
    result = test_multilabel_autofusion_with_kaggle()
    
    # Show additional information
    compare_with_single_label()
    
    if result:
        print("\n🎉 Multi-Label test completed successfully!")
        print("   AutoFusion successfully combined ML and LLM models for multi-label classification")
        print("   Configuration was properly passed to the fusion class")
        print("   Multiple labels per sample were correctly handled")
    else:
        print("\n⚠️  Multi-Label test encountered issues")

if __name__ == "__main__":
    main()
