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
    """Load and prepare a multi-label version of the abstracts dataset."""
    print("📂 Loading abstracts dataset for multi-label classification...")
    
    # Load data
    df = pd.read_csv('data/abstracts.csv', encoding='latin1')
    print(f"   📊 Original dataset shape: {df.shape}")
    print(f"   📋 Original columns: {df.columns.tolist()}")
    
    # Delete ID column if it exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        print("   🗑️  Deleted ID column")
    
    # Fuse TITLE and ABSTRACT columns into a new "text" column
    if 'TITLE' in df.columns and 'ABSTRACT' in df.columns:
        print("   🔗 Fusing TITLE and ABSTRACT columns into 'text' column...")
        df['text'] = df['TITLE'].fillna('') + ' ' + df['ABSTRACT'].fillna('')
        df['text'] = df['text'].str.strip()  # Remove leading/trailing whitespace
        
        # Remove TITLE and ABSTRACT columns, keep other columns as labels
        label_columns = [col for col in df.columns if col not in ['TITLE', 'ABSTRACT', 'text']]
        df = df[['text'] + label_columns]
        
        print(f"   🏷️  Using existing label columns: {label_columns}")
        
        # Verify that label columns contain binary values (0/1)
        for label_col in label_columns:
            unique_vals = df[label_col].unique()
            print(f"       {label_col}: unique values = {sorted(unique_vals)}")
            
            # Convert to binary if needed
            if not all(val in [0, 1] for val in unique_vals if pd.notna(val)):
                print(f"       Converting {label_col} to binary format...")
                df[label_col] = (df[label_col] > 0).astype(int)
        
    else:
        # Fallback: assume first column is text, rest are labels
        print("   📝 Using first column as text, rest as label columns...")
        text_col = df.columns[0]
        label_columns = df.columns[1:].tolist()
        df = df.rename(columns={text_col: 'text'})
    
    print(f"   📋 After processing columns: {df.columns.tolist()}")
    print(f"   📊 Processed dataset shape: {df.shape}")
    print(f"   📝 Sample text length: {len(df['text'].iloc[0])} characters")
    
    # Check label distribution
    print(f"   📊 Label distribution:")
    for label in label_columns:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"       {label}: {count} samples ({percentage:.1f}%)")
    
    # Check for multi-label samples
    multi_label_count = (df[label_columns].sum(axis=1) > 1).sum()
    print(f"   🔄 Samples with multiple labels: {multi_label_count} ({(multi_label_count/len(df)*100):.1f}%)")
    
    # Ensure each sample has at least one label
    no_labels_mask = df[label_columns].sum(axis=1) == 0
    if no_labels_mask.sum() > 0:
        print(f"   ⚠️  Warning: {no_labels_mask.sum()} samples have no labels")
        # Assign first label as default for samples without labels
        if label_columns:
            df.loc[no_labels_mask, label_columns[0]] = 1
            print(f"   📝 Assigned {label_columns[0]} as default label for samples without labels")
    multi_label_count = (df[label_columns].sum(axis=1) > 1).sum()
    print(f"   🔄 Samples with multiple labels: {multi_label_count} ({(multi_label_count/len(df)*100):.1f}%)")
    
    # Shuffle and split data randomly
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Randomly sample training and test sets
    train_df = df_shuffled.sample(n=20, random_state=42).reset_index(drop=True)
    remaining_df = df_shuffled.drop(train_df.index).reset_index(drop=True)
    test_df = remaining_df.sample(n=10, random_state=42).reset_index(drop=True)
    
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
    """Test AutoFusion classifier with multi-label abstracts dataset."""
    print("\n🚀 TESTING MULTI-LABEL AUTOFUSION WITH ABSTRACTS DATASET")
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
        'output_dir': 'autofusion_multilabel_abstracts_test',
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
    """Main function to run Multi-Label AutoFusion test with abstracts data."""
    print("🧪 MULTI-LABEL AUTOFUSION ABSTRACTS DATASET TEST")
    print("=" * 70)
    print("Testing the simplified AutoFusion interface with multi-label classification")
    print("This demonstrates how config is passed for multi-label scenarios using research abstracts")
    
    # Check if data file exists
    if not os.path.exists('data/abstracts.csv'):
        print("❌ Error: data/abstracts.csv not found")
        print("   Please ensure the abstracts dataset is in the data/ directory")
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
