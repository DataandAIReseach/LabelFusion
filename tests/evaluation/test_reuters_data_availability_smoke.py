#!/usr/bin/env python3
"""
Smoke test for Reuters data availability experiments.
Tests with minimal data to verify the pipeline works end-to-end.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import ModelConfig, ModelType, EnsembleConfig


def detect_label_columns(df: pd.DataFrame, text_column: str = 'text'):
    """Detect binary label columns."""
    candidate = [c for c in df.columns if c != text_column and not c.lower().startswith('id')]
    labels = []
    for c in candidate:
        ser = df[c]
        try:
            if pd.api.types.is_numeric_dtype(ser):
                uniques = set(pd.Series(ser).dropna().unique())
                if uniques.issubset({0, 1, 0.0, 1.0}):
                    labels.append(c)
        except Exception:
            continue
    if not labels:
        labels = [c for c in df.columns if c != text_column]
    return labels


def load_datasets(data_dir: str):
    """Load Reuters datasets."""
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_val, df_test


def main():
    """Run smoke test with minimal data."""
    print("\n" + "="*80)
    print("🧪 REUTERS DATA AVAILABILITY SMOKE TEST")
    print("="*80)
    
    # Configuration
    data_dir = '/scratch/users/u19147/LabelFusion/data/reuters'
    output_dir = 'tests/experiments/smoke_test'
    cache_dir = 'cache'
    
    # Use very small sample for smoke test
    train_pct = 0.01  # 1% of training data (~70 samples)
    val_pct = 0.01    # 1% of validation data (~20 samples)
    test_samples = 50  # Only 50 test samples
    
    print(f"\n📊 Running with {int(train_pct*100)}% train, {int(val_pct*100)}% val, {test_samples} test samples")
    
    # Load and sample datasets
    print("\n📂 Loading datasets...")
    df_train, df_val, df_test = load_datasets(data_dir)
    
    text_column = 'text'
    label_columns = detect_label_columns(df_train, text_column)
    
    # Sample data
    n_train = max(50, int(len(df_train) * train_pct))
    n_val = max(20, int(len(df_val) * val_pct))
    
    sampled_train = df_train.sample(n=n_train, random_state=42).reset_index(drop=True)
    sampled_val = df_val.sample(n=n_val, random_state=42).reset_index(drop=True)
    sampled_test = df_test.sample(n=test_samples, random_state=42).reset_index(drop=True)
    
    print(f"   Train: {len(sampled_train)}, Val: {len(sampled_val)}, Test: {len(sampled_test)}")
    print(f"   Labels: {label_columns[:3]}...")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(output_dir, f"smoke_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. Train ML Model
    print("\n🤖 Training ML Model (RoBERTa)...")
    ml_config = ModelConfig(
        model_name='distilroberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'distilroberta-base',
            'max_length': 256,
            'learning_rate': 2e-5,
            'num_epochs': 1,
            'batch_size': 8
        }
    )
    
    ml_model = RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        auto_save_results=True,
        output_dir=experiment_dir,
        experiment_name='smoke_roberta'
    )
    
    ml_model.fit(sampled_train, sampled_val)
    print("   ✅ ML model trained")
    
    # 2. Create LLM Model
    print("\n🧠 Creating LLM Model (OpenAI)...")
    llm_config = ModelConfig(
        model_name='gpt-4o-mini',
        model_type=ModelType.LLM,
        parameters={
            'model': 'gpt-4o-mini',
            'temperature': 0.1,
            'max_completion_tokens': 150,
            'top_p': 1.0
        }
    )
    
    llm_model = OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        few_shot_mode=10,  # Just 10 examples for smoke test
        auto_use_cache=True,
        cache_dir=cache_dir,
        auto_save_results=True,
        output_dir=experiment_dir,
        experiment_name='smoke_openai'
    )
    print("   ✅ LLM model created")
    
    # 3. Train Fusion Ensemble
    print("\n🔗 Training Fusion Ensemble...")
    fusion_config = EnsembleConfig(
        ensemble_method='fusion',
        models=[ml_model, llm_model],
        parameters={
            'fusion_hidden_dims': [64, 32],
            'ml_lr': 1e-5,
            'fusion_lr': 5e-4,
            'num_epochs': 50,
            'batch_size': 8,
            'classification_type': 'multi_label',
            'output_dir': experiment_dir,
            'experiment_name': 'smoke_fusion',
            'val_llm_cache_path': os.path.join(cache_dir, 'val'),
            'test_llm_cache_path': os.path.join(cache_dir, 'test')
        }
    )
    
    fusion_model = FusionEnsemble(
        fusion_config,
        output_dir=experiment_dir,
        experiment_name='smoke_fusion',
        auto_save_results=True,
        save_intermediate_llm_predictions=False
    )
    
    fusion_model.add_ml_model(ml_model)
    fusion_model.add_llm_model(llm_model)
    
    fusion_model.fit(sampled_train, sampled_val)
    print("   ✅ Fusion ensemble trained")
    
    # 4. Evaluate all models on test set
    print("\n📊 Evaluating on test set...")
    
    # Fusion
    fusion_result = fusion_model.predict(sampled_test)
    fusion_metrics = fusion_result['metrics']
    
    # ML
    ml_result = ml_model.predict(sampled_test)
    ml_metrics = ml_result['metrics']
    
    # LLM
    llm_result = llm_model.predict(sampled_test)
    llm_metrics = llm_result['metrics']
    
    # Print results
    print("\n" + "="*80)
    print("📊 SMOKE TEST RESULTS")
    print("="*80)
    print("\n| Model        | Accuracy | F1-Score | Precision | Recall |")
    print("|--------------|----------|----------|-----------|--------|")
    
    models = [
        ('Fusion', fusion_metrics),
        ('RoBERTa', ml_metrics),
        ('OpenAI', llm_metrics)
    ]
    
    for model_name, metrics in models:
        accuracy = metrics.get('subset_accuracy', metrics.get('accuracy', 0.0))
        f1 = metrics.get('f1', metrics.get('f1_weighted', 0.0))
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        
        print(f"| {model_name:<12} | {accuracy:>8.3f} | {f1:>8.3f} | {precision:>9.3f} | {recall:>6.3f} |")
    
    print("\n✅ Smoke test completed successfully!")
    print(f"📁 Results saved to: {experiment_dir}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Smoke test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
