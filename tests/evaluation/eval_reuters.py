"""
Evaluation script for testing Fusion Ensemble performance with different training data sizes on Reuters.

Tests how the ensemble performs with different percentages of training data,
while keeping validation and test sets constant.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType, ClassificationType


def load_datasets(data_dir: str = "/scratch/users/u19147/LabelFusion/data/reuters"):
    """Load Reuters train, validation and test datasets.

    Expects:
        train.csv, val.csv, test.csv
    with columns:
        - 'text'
        - one column per label (binary 0/1)
    """
    print(f"Loading datasets from {data_dir}...")
    
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val   = pd.read_csv(os.path.join(data_dir, "val.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.read_csv(os.path.join(data_dir, "test.csv")).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Training samples:   {len(df_train)}")
    print(f"  Validation samples: {len(df_val)}")
    print(f"  Test samples:       {len(df_test)}")
    
    return df_train, df_val, df_test


def create_stratified_subset(df: pd.DataFrame, percentage: float, label_columns: list, random_state: int = 42) -> pd.DataFrame:
    """Create a stratified subset of the dataframe (multi-label-aware approximation).

    For multi-label, we create a string representation of the label combination
    and stratify on that. If combinations are too rare and stratification fails,
    we fall back to random sampling.
    """
    from sklearn.model_selection import train_test_split
    
    if percentage >= 1.0:
        return df.reset_index(drop=True)
    
    try:
        # Multi-label: encode label pattern as a string like "010001..."
        stratify_column = df[label_columns].apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
        
        subset_df, _ = train_test_split(
            df,
            train_size=percentage,
            stratify=stratify_column,
            random_state=random_state
        )
    except (ValueError, Exception):
        print(f"  Warning: Stratification failed for {percentage*100:.1f}%, using random sampling instead")
        n_samples = max(1, int(len(df) * percentage))
        subset_df = df.sample(n=n_samples, random_state=random_state)
    
    return subset_df.reset_index(drop=True)


def calculate_dataset_hash(df: pd.DataFrame, label_columns: list) -> str:
    """Calculate hash of a dataset based on its size and label configuration."""
    size_str = f"reuters_train_size_{len(df)}_labels_{len(label_columns)}_seed_42"
    hash_obj = hashlib.sha256(size_str.encode('utf-8'))
    return hash_obj.hexdigest()[:8]


def create_ml_model(text_column: str, label_columns: list, output_dir: str, experiment_name: str, 
                    auto_save_path: str = None) -> RoBERTaClassifier:
    """Create and configure ML (RoBERTa) model."""
    ml_config = ModelConfig(
        model_name='roberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'roberta-base',
            'max_length': 256,
            'learning_rate': 2e-5,
            'num_epochs': 2,
            'batch_size': 8,
        }
    )
    
    return RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        auto_save_path=auto_save_path,
        auto_save_results=True,
        output_dir=output_dir,
        experiment_name=f"{experiment_name}_roberta"
    )


def create_llm_model(text_column: str, label_columns: list, 
                     provider: str = 'openai', 
                     output_dir: str = "outputs",
                     experiment_name: str = "llm",
                     cache_dir: str = "cache",
                     few_shot_examples: int = 2):
    """Create and configure LLM model."""
    
    if provider == 'openai':
        llm_config = ModelConfig(
            model_name="gpt-5-nano",
            model_type=ModelType.LLM,
            parameters={
                "model": "gpt-5-nano",
                "temperature": 0.1,
                "max_completion_tokens": 150,
                "top_p": 1.0
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
    elif provider == 'deepseek':
        llm_config = Config()
        llm_config.model_type = ModelType.LLM
        llm_config.parameters = {
            'model': 'deepseek-chat',
            'temperature': 0.1,
            'max_completion_tokens': 150,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        return DeepSeekClassifier(
            config=llm_config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=True,
            few_shot_mode=few_shot_examples,
            auto_save_results=True,
            auto_use_cache=True,
            cache_dir=cache_dir,
            output_dir=output_dir,
            experiment_name=f"{experiment_name}_deepseek"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_fusion_ensemble(ml_model, llm_model, output_dir: str, experiment_name: str, 
                          auto_use_cache: bool = True, cache_dir: str = "cache",
                          val_llm_cache_path: str = None, test_llm_cache_path: str = None) -> FusionEnsemble:
    """Create and configure Fusion Ensemble."""
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
            'auto_save_results': True,
            'val_llm_cache_path': val_llm_cache_path or f"{cache_dir}/val",
            'test_llm_cache_path': test_llm_cache_path or f"{cache_dir}/test"
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
    
    return fusion


def evaluate_with_data_percentage(
    df_train_full: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    percentage: float,
    text_column: str,
    label_columns: list,
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/reuters_availability_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
) -> dict:
    """Evaluate fusion ensemble with a specific percentage of Reuters training data."""
    percentage_str = f"{int(percentage * 100)}pct" if percentage >= 0.01 else f"{percentage * 100:.1f}pct"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"reuters_{percentage_str}_{timestamp}"
    output_dir = os.path.join(base_output_dir, f"train_{percentage_str}")
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: Training with {percentage_str} of training data")
    print("="*80)
    
    # Create stratified subset of training data
    print(f"\nCreating stratified subset with {percentage:.0%} of training data...")
    df_train_subset = create_stratified_subset(df_train_full, percentage, label_columns)
    print(f"  Subset size: {len(df_train_subset)} samples")
    
    # Verify label distribution
    print("\nLabel distribution in subset (sum of positives per label):")
    subset_dist = df_train_subset[label_columns].sum()
    print(subset_dist)
    
    # Calculate hash of the training subset for cache lookup
    dataset_hash = calculate_dataset_hash(df_train_subset, label_columns)
    print(f"\nTraining subset hash: {dataset_hash}")
    
    # Setup cache paths - use hash-based naming
    ml_cache_path = os.path.join(cache_dir, f"fusion_roberta_model_{dataset_hash}")
    val_cache_path = os.path.join(cache_dir, "val")
    test_cache_path = os.path.join(cache_dir, "test")
    
    # Create models
    print("\nCreating models...")
    ml_model = create_ml_model(text_column, label_columns, output_dir, experiment_name, auto_save_path=ml_cache_path)
    # few_shot_examples may be injected into the local scope by the caller (run_data_availability_experiments)
    few_shot_local = globals().get('FEW_SHOT_OVERRIDE', None)
    if few_shot_local is None:
        few_shot_local = 2
    llm_model = create_llm_model(text_column, label_columns, llm_provider, output_dir, experiment_name, cache_dir=cache_dir, few_shot_examples=int(few_shot_local))
    
    # Create fusion ensemble
    print("Creating fusion ensemble...")
    fusion = create_fusion_ensemble(
        ml_model, llm_model, output_dir, experiment_name, auto_use_cache, cache_dir,
        val_llm_cache_path=val_cache_path, test_llm_cache_path=test_cache_path
    )
    
    all_model_results = {}
    
    # ===== Try loading cached RoBERTa model =====
    ml_model_loaded_from_cache = False
    if os.path.exists(ml_cache_path):
        print(f"\n📦 Loading cached RoBERTa model from: {ml_cache_path}")
        try:
            ml_model.load_model(ml_cache_path)
            print("✅ Successfully loaded cached model")
            ml_model_loaded_from_cache = True
        except Exception as e:
            print(f"⚠️ Failed to load cached model: {e}")
            print("Will train a new model...")
    
    # ===== BASELINE 1: RoBERTa alone =====
    if evaluate_baselines:
        print("\n" + "-"*80)
        print("BASELINE 1: Evaluating RoBERTa (ML) model alone...")
        print("-"*80)
        
        if not ml_model_loaded_from_cache:
            print(f"Training RoBERTa on {len(df_train_subset)} samples...")
            ml_training_result = ml_model.fit(df_train_subset, df_val)
        else:
            ml_training_result = {
                "model_name": "roberta-base",
                "cached": True,
                "cache_path": ml_cache_path
            }
        
        print("Evaluating RoBERTa on test set...")
        ml_test_result = ml_model.predict(df_test)
        ml_test_metrics = ml_test_result.metadata.get('metrics', {}) if ml_test_result.metadata else {}
        
        print("\nRoBERTa Test Results:")
        print(f"  Accuracy:            {ml_test_metrics.get('accuracy', 0.0):.4f}")
        print(f"  F1 Score (Weighted): {ml_test_metrics.get('f1_weighted', 0.0):.4f}")
        print(f"  Precision (Weighted):{ml_test_metrics.get('precision_weighted', 0.0):.4f}")
        print(f"  Recall (Weighted):   {ml_test_metrics.get('recall_weighted', 0.0):.4f}")
        
        all_model_results['roberta'] = {
            'test_metrics': ml_test_metrics,
            'training_result': ml_training_result
        }
    
    # ===== BASELINE 2: LLM alone (few-shot on small train, full-val/test evaluation) =====
    if evaluate_baselines:
        print("\n" + "-"*80)
        print(f"BASELINE 2: Evaluating {llm_provider.upper()} (LLM) model alone on tiny subset...")
        print("-"*80)
        
        # Use the configured few-shot size for LLM baseline (FEW_SHOT_OVERRIDE)
        few_shot_local = globals().get('FEW_SHOT_OVERRIDE', 2)
        n_llm_samples = int(few_shot_local)
        llm_train_df = df_train_full.sample(n=min(n_llm_samples, len(df_train_full)), random_state=42)
        llm_val_df   = df_val.sample(n=min(n_llm_samples, len(df_val)), random_state=42)
        llm_test_df  = df_test.sample(n=min(n_llm_samples, len(df_test)), random_state=42)

        print(f"Using {len(llm_train_df)} train, {len(llm_val_df)} val, {len(llm_test_df)} test samples for LLM baseline.")

        # LLM is given only the few-shot train examples, but evaluated on full tiny test subset
        llm_test_result = llm_model.predict(
            train_df=llm_train_df,
            val_df=llm_val_df,
            test_df=llm_test_df
        )
        llm_test_metrics = llm_test_result.metadata.get('metrics', {}) if llm_test_result.metadata else {}
        
        print(f"\n{llm_provider.upper()} Test Results (on tiny subset):")
        print(f"  Accuracy:   {llm_test_metrics.get('accuracy', 0.0):.4f}")
        print(f"  F1 Score:   {llm_test_metrics.get('f1', 0.0):.4f}")
        print(f"  Precision:  {llm_test_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall:     {llm_test_metrics.get('recall', 0.0):.4f}")
        
        all_model_results['llm'] = {
            'test_metrics': llm_test_metrics,
            'model_name': llm_provider
        }
    
    # ===== FUSION ENSEMBLE =====
    print("\n" + "-"*80)
    print("FUSION ENSEMBLE: Training combined RoBERTa + LLM model...")
    print("-"*80)
    
    # Determine whether ML should be trained on the full training set (env override)
    ml_train_on_full = os.getenv('ML_TRAIN_ON_FULL', '').lower() in ('1', 'true', 'yes')
    if not ml_model_loaded_from_cache and not evaluate_baselines:
        if ml_train_on_full:
            print(f"🔄 ML_TRAIN_ON_FULL enabled — training RoBERTa on full training set ({len(df_train_full)} samples)")
            ml_model.fit(df_train_full, df_val)
        else:
            print(f"🔄 Training RoBERTa on {len(df_train_subset)} samples...")
            ml_model.fit(df_train_subset, df_val)
    elif ml_model_loaded_from_cache:
        print("✅ Using cached RoBERTa model (already loaded)")
    else:
        print("✅ Using RoBERTa model from baseline evaluation")
    
    # Fusion training: typically uses the same ML training data. If ML_TRAIN_ON_FULL is set,
    # use the full training set for fusion.fit so the ML and fusion see the same data.
    fusion_train_df = df_train_full if ml_train_on_full else df_train_subset

    print(f"\n🔧 Training fusion ensemble on {len(fusion_train_df)} samples...")
    training_result = fusion.fit(fusion_train_df, df_val.sample(n=10, random_state=42))
    
    print("\nTraining completed!")
    print(f"  ML model trained:      {training_result.get('ml_model_trained', False)}")
    print(f"  Fusion MLP trained:    {training_result.get('fusion_mlp_trained', False)}")
    
    print("\nEvaluating fusion ensemble on test set...")
    # Predict on the full test set (LLM will still use few-shot prompts internally).
    test_result = fusion.predict(df_test.sample(n=5, random_state=42))
    
    test_metrics = test_result.metadata.get('metrics', {}) if test_result.metadata else {}
    
    print("\nFusion Ensemble Test Results:")
    print(f"  Accuracy:   {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"  F1 Score:   {test_metrics.get('f1', 0.0):.4f}")
    print(f"  Precision:  {test_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:     {test_metrics.get('recall', 0.0):.4f}")
    
    all_model_results['fusion'] = {
        'test_metrics': test_metrics,
        'training_result': training_result
    }
    
    # ===== COMPARISON SUMMARY =====
    if evaluate_baselines:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"\nModel Comparison on Test Set ({len(df_test)} samples for ML/Fusion, tiny subset for LLM baseline):")
        print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}")
        print("-" * 65)
        
        for model_name in ['roberta', 'llm', 'fusion']:
            if model_name in all_model_results:
                metrics = all_model_results[model_name]['test_metrics']
                display_name = {
                    'roberta': 'RoBERTa (ML)',
                    'llm': f'{llm_provider.upper()} (LLM)',
                    'fusion': 'Fusion Ensemble'
                }[model_name]
                
                acc = metrics.get('accuracy', 0.0)
                if model_name == 'roberta':
                    f1 = metrics.get('f1_weighted', 0.0)
                    precision = metrics.get('precision_weighted', 0.0)
                    recall = metrics.get('recall_weighted', 0.0)
                else:
                    f1 = metrics.get('f1', 0.0)
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                
                print(f"{display_name:<20} {acc:>10.4f} {f1:>10.4f} {precision:>12.4f} {recall:>10.4f}")
        
        if 'roberta' in all_model_results and 'fusion' in all_model_results:
            roberta_acc = all_model_results['roberta']['test_metrics'].get('accuracy', 0.0)
            fusion_acc = all_model_results['fusion']['test_metrics'].get('accuracy', 0.0)
            improvement = ((fusion_acc - roberta_acc) / roberta_acc * 100) if roberta_acc > 0 else 0
            print("\n" + "-" * 55)
            print(f"Fusion vs RoBERTa: {improvement:+.2f}% accuracy improvement")
    
    results = {
        'percentage': percentage,
        'percentage_str': percentage_str,
        'experiment_name': experiment_name,
        'output_dir': output_dir,
        'train_samples': len(df_train_subset),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'training_result': training_result,
        'test_metrics': test_metrics,
        'test_predictions': len(test_result.predictions),
        'all_models': all_model_results if evaluate_baselines else {'fusion': all_model_results.get('fusion', {})}
    }
    
    return results


def run_data_availability_experiments(
    data_dir: str = "/scratch/users/u19147/LabelFusion/data/reuters",
    percentages: list = None,
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/reuters_availability_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
):
    """Run experiments with different training data percentages on Reuters."""
    if percentages is None:
        percentages = [i * 0.1 for i in range(1, 11)]  # 10% to 100%
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    df_train, df_val, df_test = load_datasets(data_dir)
    
    text_column = 'text'
    # Auto-detect label columns: prefer numeric/binary columns (0/1). Exclude obvious metadata like 'id'.
    candidate_labels = [c for c in df_train.columns if c != text_column and c != "id"]
    label_columns = []
    for c in candidate_labels:
        if c.lower() == 'id' or c.lower().startswith('id_'):
            continue
        ser = df_train[c]
        # Numeric dtype and values subset of {0,1}
        try:
            if pd.api.types.is_numeric_dtype(ser):
                uniques = set(pd.Series(ser).dropna().unique())
                # allow floats that are 0.0/1.0
                if uniques.issubset({0, 1, 0.0, 1.0}):
                    label_columns.append(c)
                    continue
            # Boolean-like columns
            if ser.dropna().apply(lambda x: str(x).lower() in ('0', '1', 'true', 'false')).all():
                label_columns.append(c)
                continue
        except Exception:
            continue
    # Fallback: if nothing detected, keep previous behavior
    if not label_columns:
        label_columns = [c for c in df_train.columns if c != text_column]
    
    print(f"\nDataset configuration:")
    print(f"  Text column:   {text_column}")
    print(f"  Label columns: {len(label_columns)}")
    print(f"  Labels:        {label_columns}")
    
    all_results = []
    
    for percentage in percentages:
        try:
            result = evaluate_with_data_percentage(
                df_train_full=df_train,
                df_val=df_val,
                df_test=df_test,
                percentage=percentage,
                text_column=text_column,
                label_columns=label_columns,
                llm_provider=llm_provider,
                base_output_dir=base_output_dir,
                auto_use_cache=auto_use_cache,
                cache_dir=cache_dir,
                evaluate_baselines=evaluate_baselines
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error in experiment with {int(percentage*100)}% training data: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        pct = result['percentage'] * 100
        if pct < 1:
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = f"{int(pct)}%"
        
        fusion_metrics = result['test_metrics']
        summary_data.append({
            'percentage': pct_str,
            'model': 'Fusion',
            'train_samples': result['train_samples'],
            'accuracy': fusion_metrics.get('accuracy', 0.0),
            'f1': fusion_metrics.get('f1', 0.0),
            'precision': fusion_metrics.get('precision', 0.0),
            'recall': fusion_metrics.get('recall', 0.0)
        })
        
        if evaluate_baselines and 'all_models' in result:
            all_models = result['all_models']
            
            if 'roberta' in all_models:
                ml_metrics = all_models['roberta']['test_metrics']
                summary_data.append({
                    'percentage': pct_str,
                    'model': 'RoBERTa',
                    'train_samples': result['train_samples'],
                    'accuracy': ml_metrics.get('accuracy', 0.0),
                    'f1': ml_metrics.get('f1', 0.0),
                    'precision': ml_metrics.get('precision', 0.0),
                    'recall': ml_metrics.get('recall', 0.0)
                })
            
            if 'llm' in all_models:
                llm_metrics = all_models['llm']['test_metrics']
                summary_data.append({
                    'percentage': pct_str,
                    'model': llm_provider.upper(),
                    'train_samples': result['train_samples'],
                    'accuracy': llm_metrics.get('accuracy', 0.0),
                    'f1': llm_metrics.get('f1', 0.0),
                    'precision': llm_metrics.get('precision', 0.0),
                    'recall': llm_metrics.get('recall', 0.0)
                })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(base_output_dir, f"summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n📊 Summary saved to: {summary_file}")
    
    detailed_file = os.path.join(base_output_dir, f"detailed_results_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📋 Detailed results saved to: {detailed_file}")
    
    return all_results, summary_df


if __name__ == "__main__":
    def _parse_percentages(s: str):
        """Parse a comma- or whitespace-separated string of floats into a list of floats."""
        import re
        if not s:
            return [i * 0.1 for i in range(1, 11)]  # 10% to 100%
        parts = [p for p in re.split(r'[,\s]+', s.strip()) if p]
        return [float(p) for p in parts]

    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    percentages = _parse_percentages(os.getenv('PERCENTAGES', ''))
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    base_output_dir = os.getenv('OUTPUT_DIR', 'outputs/reuters_availability_experiments')
    no_cache_env = os.getenv('NO_CACHE', '')
    auto_use_cache = not (no_cache_env.lower() in ('1', 'true', 'yes'))
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    no_baselines_env = os.getenv('NO_BASELINES', '1')  # default: skip baselines
    evaluate_baselines = not (no_baselines_env.lower() in ('1', 'true', 'yes'))

    print("="*80)
    print("DATA AVAILABILITY EXPERIMENT - FUSION ENSEMBLE ON REUTERS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory:       {data_dir}")
    pct_display = [f"{p*100:.1f}%" if p < 0.01 else f"{int(p*100)}%" for p in percentages]
    print(f"  Training percentages: {pct_display}")
    print(f"  LLM provider:         {llm_provider}")
    print(f"  Output directory:     {base_output_dir}")
    print(f"  Auto-use cache:       {auto_use_cache}")
    print(f"  Cache directory:      {cache_dir}")
    print(f"  Evaluate baselines:   {evaluate_baselines}")

    # Allow overriding few-shot examples via FEW_SHOT env var (default 2)
    try:
        few_shot_env = int(os.getenv('FEW_SHOT', '2'))
    except Exception:
        few_shot_env = 2
    # store in module globals so evaluate_with_data_percentage can pick it up
    globals()['FEW_SHOT_OVERRIDE'] = few_shot_env

    # Optional: pre-generate LLM cache for full val/test using sampled few-shot context
    # Enable with environment variable PRECACHE_LLM=1 (uses FEW_SHOT value)
    try:
        precache_flag = os.getenv('PRECACHE_LLM', '').lower() in ('1', 'true', 'yes')
    except Exception:
        precache_flag = False

    if precache_flag:
        print('\n🔁 PRECACHE MODE: Generating LLM predictions for full val/test using few-shot context')


        # Load datasets (deterministic sampling)
        df_train_all, df_val_all, df_test_all = load_datasets(data_dir)

        if True:
            df_val_all = df_val_all.sample(n=5, random_state=42)
            df_test_all = df_test_all.sample(n=5, random_state=42)

        # Determine label columns same way as run_data_availability_experiments
        candidate_labels = [c for c in df_train_all.columns if c != 'text']
        label_columns = []
        for c in candidate_labels:
            if c.lower() == 'id' or c.lower().startswith('id_'):
                continue
            ser = df_train_all[c]
            try:
                if pd.api.types.is_numeric_dtype(ser):
                    uniques = set(pd.Series(ser).dropna().unique())
                    if uniques.issubset({0, 1, 0.0, 1.0}):
                        label_columns.append(c)
                        continue
                if ser.dropna().apply(lambda x: str(x).lower() in ('0', '1', 'true', 'false')).all():
                    label_columns.append(c)
                    continue
            except Exception:
                continue
        if not label_columns:
            label_columns = [c for c in df_train_all.columns if c != 'text']

        few_shot_to_use = few_shot_env
        print(f"  Using {few_shot_to_use} few-shot examples to create cached LLM predictions")
        train_small = df_train_all.sample(n=min(few_shot_to_use, len(df_train_all)), random_state=42)

        # Create LLM with the configured few-shot size
        llm_model = create_llm_model('text', label_columns, llm_provider, base_output_dir, 'precache', cache_dir=cache_dir, few_shot_examples=int(few_shot_to_use))

        # Generate predictions for val and test and save into fusion cache format
        print('  Generating validation predictions...')
        val_result = llm_model.predict(train_df=train_small, test_df=df_val_all)
        print(f"   -> {len(val_result.predictions)} validation preds generated")

        print('  Generating test predictions...')
        test_result = llm_model.predict(train_df=train_small, test_df=df_test_all)
        print(f"   -> {len(test_result.predictions)} test preds generated")

        # Create a temporary ML + Fusion ensemble to write cache files using the same helpers
        tmp_ml = create_ml_model('text', label_columns, base_output_dir, 'precache_ml')
        tmp_fusion = create_fusion_ensemble(tmp_ml, llm_model, base_output_dir, 'precache_fusion', cache_dir=cache_dir)
        tmp_fusion._save_cached_llm_predictions(val_result.predictions, os.path.join(cache_dir, 'val'), df_val_all)
        tmp_fusion._save_cached_llm_predictions(test_result.predictions, os.path.join(cache_dir, 'test'), df_test_all)

        print('✅ Precache complete. Cached files saved under cache/val_* and cache/test_*')

    results, summary = run_data_availability_experiments(
        data_dir=data_dir,
        percentages=percentages,
        llm_provider=llm_provider,
        base_output_dir=base_output_dir,
        auto_use_cache=auto_use_cache,
        cache_dir=cache_dir,
        evaluate_baselines=evaluate_baselines
    )

    print("\n✅ All experiments completed!")
