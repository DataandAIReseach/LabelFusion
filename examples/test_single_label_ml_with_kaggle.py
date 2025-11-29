#!/usr/bin/env python3
"""
Test single-label ML-only classifier with Kaggle ecommerce dataset.

This test demonstrates using only a traditional ML model (no LLM, no fusion) for single-label classification.
"""

import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import textclassify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from textclassify.ml.base import MLClassifier  # Assuming this is your ML-only classifier

# Load environment variables
load_dotenv()

def prepare_kaggle_data():
    """Load and prepare the Kaggle ecommerce dataset for ML-only classification."""
    print("📂 Loading Kaggle ecommerce dataset...")
    
    # Load data
    df = pd.read_csv('data/ecommerceDataset.csv', encoding='latin1')
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
        df['text'] = df['text'].str.strip()
        label_col = [col for col in df.columns if col not in ['TITLE', 'ABSTRACT', 'text']][0]
        df = df[['text', label_col]]
        df.columns = ['text', 'label']
    else:
        cols = df.columns.tolist()
        df = df[[cols[1], cols[0]]]
        df.columns = ['text', 'label']
    
    print(f"   📋 After processing columns: {df.columns.tolist()}")
    print(f"   📊 Processed dataset shape: {df.shape}")
    print(f"   📝 Sample text length: {len(df['text'].iloc[0])} characters")
    
    # Convert label column to dummy variables
    label_dummies = pd.get_dummies(df['label'], prefix='')
    df = pd.concat([df[['text']], label_dummies], axis=1)
    label_columns = label_dummies.columns.tolist()
    print(f"   🏷️  Available labels: {label_columns}")
    
    # Shuffle and split data randomly
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df_shuffled.sample(n=20, random_state=42).reset_index(drop=True)
    remaining_df = df_shuffled.drop(train_df.index).reset_index(drop=True)
    test_df = remaining_df.sample(n=10, random_state=42).reset_index(drop=True)
    
    print(f"   ✅ Training set: {len(train_df)} samples")
    print(f"   ✅ Test set: {len(test_df)} samples")
    print(f"   📋 Train DataFrame columns: {train_df.columns.tolist()}")
    print(f"   📋 Test DataFrame columns: {test_df.columns.tolist()}")
    print(f"   📋 Test DataFrame shape: {test_df.shape}")
    print(f"   📋 Test DataFrame sample:")
    print(f"       Text: {test_df['text'].iloc[0][:50]}...")
    print(f"       Labels: {[col for col in label_columns if test_df[col].iloc[0] == 1]}")
    
    return train_df, test_df, label_columns

def test_single_label_ml_with_kaggle():
    """Test ML-only classifier with Kaggle dataset."""
    print("\n🚀 TESTING ML-ONLY CLASSIFIER WITH KAGGLE DATASET")
    print("=" * 60)
    
    train_df, test_df, label_columns = prepare_kaggle_data()
    
    config = {
        'label_columns': label_columns,
        'multi_label': False,
        'text_column': 'text',
        'batch_size': 4,
        'num_epochs': 2,
        'output_dir': 'models/ml_singlelabel_ecommerce',
        'ml_model': 'roberta-base',
        'max_length': 256
    }
    print("\n🔧 ML-Only Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n🤖 Creating ML-only classifier...")
        classifier = MLClassifier(config=config)
        print("\n✅ ML-only classifier created successfully!")
        print(f"   🏷️  Labels: {classifier.label_columns}")
        print(f"   🤖 ML Model: {classifier.ml_model_name}")
        
        print("\n🎯 Training ML-only classifier...")
        classifier.fit(train_df)
        print("\n✅ ML-only training completed!")
        
        print("\n📊 Making predictions on test data...")
        result = classifier.predict(test_df)
        print("\n📋 Sample Predictions:")
        print("-" * 80)
        for i in range(min(5, len(test_df))):
            text = test_df['text'].iloc[i]
            pred = result.predictions[i]
            true_labels = [col for col in label_columns if test_df[col].iloc[i] == 1]
            print(f"\nSample {i + 1}:")
            print(f"Text: {text[:100]}...")
            print(f"True Label: {true_labels}")
            print(f"Prediction: {pred}")
        if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
            metrics = result.metadata['metrics']
            print("\n📈 Performance Metrics:")
            print("-" * 40)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
        print("\n✅ ML-only test completed successfully!")
        return result
    except Exception as e:
        print(f"\n❌ Error during ML-only test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧪 ML-ONLY KAGGLE DATASET TEST")
    print("=" * 60)
    if not os.path.exists('data/ecommerceDataset.csv'):
        print("❌ Error: data/ecommerceDataset.csv not found")
        print("   Please ensure the Kaggle dataset is in the data/ directory")
        return
    result = test_single_label_ml_with_kaggle()
    if result:
        print("\n🎉 ML-only test completed successfully!")
    else:
        print("\n⚠️  ML-only test encountered issues")

if __name__ == "__main__":
    main()
