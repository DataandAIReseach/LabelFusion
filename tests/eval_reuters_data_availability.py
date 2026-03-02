"""Reuters data availability experiments.

Trains fusion models with different percentages of:
- Training data for ML model (1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9%, 10%)
- Validation LLM predictions for fusion training (subsets of val set)

Outputs a table showing performance at different data availability levels.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List

# ensure project root on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType


def load_datasets(data_dir: str = "/scratch/users/u19147/LabelFusion/data/reuters"):
    """Load train, val, test datasets."""
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(data_dir, "val.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_train, df_val, df_test


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


def train_ml_model(train_df: pd.DataFrame, val_df: pd.DataFrame, text_column: str, 
                   label_columns: list, output_dir: str, experiment_name: str, 
                   cache_path: str = None) -> RoBERTaClassifier:
    """Train or load ML model."""
    ml_config = ModelConfig(
        model_name='roberta-base', 
        model_type=ModelType.TRADITIONAL_ML, 
        parameters={
            'model_name': 'roberta-base', 
            'max_length': 256, 
            'learning_rate': 2e-5, 
            'num_epochs': 2, 
            'batch_size': 32
        }
    )
    ml_model = RoBERTaClassifier(
        config=ml_config, 
        text_column=text_column, 
        label_columns=label_columns, 
        multi_label=True, 
        auto_save_results=True, 
        output_dir=output_dir, 
        experiment_name=f"{experiment_name}_roberta"
    )
    
    # Try to load from cache if path provided
    if cache_path and os.path.exists(cache_path):
        try:
            ml_model.load_model(cache_path)
            print(f"✅ Loaded ML model from cache: {cache_path}")
            return ml_model
        except Exception as e:
            print(f"⚠️ Failed to load cached ML model: {e}")
    
    # Train new model
    print(f"🔧 Training ML model on {len(train_df)} samples...")
    ml_model.fit(train_df, val_df)
    
    # Save to cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ml_model.save_model(cache_path)
        print(f"💾 Saved ML model to cache: {cache_path}")
    
    return ml_model


def create_llm_model(text_column: str, label_columns: list, output_dir: str, 
                     experiment_name: str, cache_dir: str = 'cache', 
                     few_shot_examples: int = 20) -> OpenAIClassifier:
    """Create LLM model with caching."""
    llm_config = ModelConfig(
        model_name='gpt-5-nano', 
        model_type=ModelType.LLM, 
        parameters={
            'model': 'gpt-5-nano', 
            'temperature': 0.1, 
            'max_completion_tokens': 150, 
            'top_p': 1.0
        }
    )
    return OpenAIClassifier(
        config=llm_config, 
        text_column=text_column, 
        label_columns=label_columns, 
        multi_label=True, 
        few_shot_mode=few_shot_examples, 
        auto_use_cache=True, 
        cache_dir=cache_dir, 
        auto_save_results=True, 
        output_dir=output_dir, 
        experiment_name=f"{experiment_name}_openai"
    )


def train_fusion_model(ml_model: RoBERTaClassifier, llm_model: OpenAIClassifier, 
                       train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       output_dir: str, experiment_name: str, 
                       cache_dir: str = 'cache') -> FusionEnsemble:
    """Train fusion ensemble."""
    fusion_config = EnsembleConfig(
        ensemble_method='fusion', 
        models=[ml_model, llm_model], 
        parameters={
            'fusion_hidden_dims': [64, 32], 
            'ml_lr': 1e-5, 
            'fusion_lr': 5e-4, 
            'num_epochs': 10, 
            'batch_size': 8, 
            'classification_type': 'multi_label', 
            'output_dir': output_dir, 
            'experiment_name': experiment_name, 
            'val_llm_cache_path': os.path.join(cache_dir, 'val'), 
            'test_llm_cache_path': os.path.join(cache_dir, 'test')
        }
    )
    fusion = FusionEnsemble(
        fusion_config, 
        output_dir=output_dir, 
        experiment_name=experiment_name, 
        auto_save_results=True, 
        save_intermediate_llm_predictions=False
    )
    fusion.add_ml_model(ml_model)
    fusion.add_llm_model(llm_model)
    
    print(f"🔧 Training fusion on {len(train_df)} train samples and {len(val_df)} val samples...")

    fusion.fit(train_df, val_df)
    
    return fusion


def run_single_experiment(train_pct: float, df_train: pd.DataFrame, 
                         df_val: pd.DataFrame, df_test: pd.DataFrame, 
                         text_column: str, label_columns: list, 
                         base_output_dir: str, cache_dir: str) -> Dict:
    """Run a single experiment with specified train percentage.
    
    Note: val_df and test_df are kept constant (full datasets) across all experiments.
    Only the training data is sampled according to train_pct.
    This ensures consistent evaluation while testing different training data amounts.
    """
    
    # Sample training data for ML model
    n_train = max(100, int(len(df_train) * train_pct))
    sampled_train = df_train.sample(n=n_train, random_state=42).reset_index(drop=True)
    
    # Use FULL validation and test sets (not sampled)
    # This ensures consistent evaluation across all experiments
    # LLM predictions on these sets will be cached and reused
    
    # Create experiment directory organized by percentage
    pct_dir = os.path.join(base_output_dir, f"{int(train_pct*100)}%")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"reuters_train{int(train_pct*100)}_{timestamp}"
    output_dir = os.path.join(pct_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🧪 Experiment: {int(train_pct*100)}% train ({n_train} samples)")
    print(f"   Validation: {len(df_val)} samples (full set, constant)")
    print(f"   Test: {len(df_test)} samples (full set, constant)")
    print(f"{'='*80}")
    
    # Check for cached ML model
    ml_cache_path = os.path.join(cache_dir, f"roberta_train{int(train_pct*100)}pct.pt")
    
    # Train/load ML model
    ml_model = train_ml_model(
        sampled_train, df_val, text_column, label_columns, 
        output_dir, experiment_name, ml_cache_path
    )
    
    # Create LLM model (uses cached predictions automatically)
    llm_model = create_llm_model(
        text_column, label_columns, output_dir, experiment_name, cache_dir
    )
    
    # Train fusion model (uses full validation set for meta-learner training)
    fusion = train_fusion_model(
        ml_model, llm_model, sampled_train, df_val,  # Note: df_val is full set, not sampled
        output_dir, experiment_name, cache_dir
    )
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_result = fusion.predict(df_test, train_df=df_train)
    fusion_metrics = test_result.metadata.get('metrics', {}) if test_result.metadata else {}
    
    # Also get individual model metrics on test set
    print("📊 Evaluating ML model on test set...")
    ml_test_result = ml_model.predict(df_test)
    ml_metrics = ml_test_result.metadata.get('metrics', {}) if ml_test_result.metadata else {}
    
    print("📊 Evaluating LLM model on test set...")
    llm_test_result = llm_model.predict(train_df=df_train, test_df=df_test)
    llm_metrics = llm_test_result.metadata.get('metrics', {}) if llm_test_result.metadata else {}
    
    return {
        'train_pct': train_pct,
        'train_samples': n_train,
        'val_samples': len(df_val),  # Always full validation set
        'test_samples': len(df_test),  # Always full test set
        'fusion_metrics': fusion_metrics,
        'ml_metrics': ml_metrics,
        'llm_metrics': llm_metrics
    }


def format_results_table(results: List[Dict]) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append("")
    lines.append("| Training Data | Model    | Accuracy | F1-Score | Precision | Recall |")
    lines.append("|---------------|----------|----------|----------|-----------|--------|")
    
    for result in results:
        train_size = result['train_samples']
        train_pct = int(result['train_pct'] * 100)
        
        # Fusion metrics
        fusion_m = result['fusion_metrics']
        fusion_acc = fusion_m.get('subset_accuracy', 0.0)
        fusion_f1 = fusion_m.get('f1', 0.0)
        fusion_prec = fusion_m.get('precision', 0.0)
        fusion_rec = fusion_m.get('recall', 0.0)
        
        lines.append(f"| {train_pct}% ({train_size}) | **Fusion**   | {fusion_acc:.1%} | {fusion_f1:.3f} | {fusion_prec:.3f} | {fusion_rec:.3f} |")
        
        # ML metrics
        ml_m = result['ml_metrics']
        ml_acc = ml_m.get('exact_match_accuracy', ml_m.get('subset_accuracy', 0.0))
        ml_f1 = ml_m.get('f1_weighted', ml_m.get('f1', 0.0))
        ml_prec = ml_m.get('precision_weighted', ml_m.get('precision', 0.0))
        ml_rec = ml_m.get('recall_weighted', ml_m.get('recall', 0.0))
        
        lines.append(f"| {train_pct}% ({train_size}) | RoBERTa  | {ml_acc:.1%} | {ml_f1:.3f} | {ml_prec:.3f} | {ml_rec:.3f} |")
        
        # LLM metrics
        llm_m = result['llm_metrics']
        llm_acc = llm_m.get('subset_accuracy', 0.0)
        llm_f1 = llm_m.get('f1', 0.0)
        llm_prec = llm_m.get('precision', 0.0)
        llm_rec = llm_m.get('recall', 0.0)
        
        lines.append(f"| {train_pct}% ({train_size}) | OpenAI   | {llm_acc:.1%} | {llm_f1:.3f} | {llm_prec:.3f} | {llm_rec:.3f} |")
    
    lines.append("")
    return "\n".join(lines)


def main():
    """Run data availability experiments."""
    # Configuration
    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    base_output_dir = os.getenv('OUTPUT_DIR', 'tests/experiments')
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    
    # Training data percentages to test (for ML model)
    train_percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]  # 1%-10%
    
    # Note: Validation and test sets are kept constant (100% of data)
    # Only training data is sampled to test impact of training data size
    # LLM predictions on val/test will be cached once and reused across experiments
    
    # Load datasets
    print("📂 Loading datasets...")
    df_train, df_val, df_test = load_datasets(data_dir)
    text_column = 'text'
    label_columns = detect_label_columns(df_train, text_column)
    
    print(f"📊 Dataset sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"🏷️  Labels: {label_columns}")
    
    # Run experiments
    results = []
    for train_pct in train_percentages:
        
        try:
            result = run_single_experiment(
                train_pct, df_train, df_val, df_test,
                text_column, label_columns, base_output_dir, cache_dir
            )
            results.append(result)
            
            print(f"\n✅ Completed: {int(train_pct*100)}% experiment")
            print(f"   Fusion F1: {result['fusion_metrics'].get('f1', 0.0):.3f}")
            print(f"   ML F1: {result['ml_metrics'].get('f1_weighted', result['ml_metrics'].get('f1', 0.0)):.3f}")
            print(f"   LLM F1: {result['llm_metrics'].get('f1', 0.0):.3f}")
            
        except Exception as e:
            print(f"\n❌ Failed {int(train_pct*100)}% experiment: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results table
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    table = format_results_table(results)
    print(table)
    
    # Save results to file
    results_file = os.path.join(base_output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(results_file, 'w') as f:
        f.write(table)
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == '__main__':
    main()
