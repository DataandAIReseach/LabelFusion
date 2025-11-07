"""
Evaluation script for testing Fusion Ensemble performance with different training data sizes on GoEmotions.

Tests how the ensemble performs with 0.1% to 1% (in 0.1% steps) and 10% to 100% (in 10% steps) of training data,
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


def load_datasets(data_dir: str = "data/goemotions"):
    """Load GoEmotions train, validation and test datasets."""
    print(f"Loading datasets from {data_dir}...")
    
    df_train = pd.read_csv(os.path.join(data_dir, "goemotions_all_train_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(data_dir, "goemotions_all_val_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(data_dir, "goemotions_all_test_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)

    
    print(f"  Training samples: {len(df_train)}")
    print(f"  Validation samples: {len(df_val)}")
    print(f"  Test samples: {len(df_test)}")
    
    return df_train, df_val, df_test


def create_stratified_subset(df: pd.DataFrame, percentage: float, label_columns: list, random_state: int = 42) -> pd.DataFrame:
    """Create a stratified subset of the dataframe.
    
    Args:
        df: Input DataFrame
        percentage: Percentage of data to keep (0.0 to 1.0)
        label_columns: List of label column names
        random_state: Random seed for reproducibility
        
    Returns:
        Stratified subset DataFrame
    """
    from sklearn.model_selection import train_test_split
    
    # If percentage is 1.0, return the full dataframe (no split needed)
    if percentage >= 1.0:
        return df.reset_index(drop=True)
    
    # Split to get the desired percentage
    try:
        # For multi-label, create a string representation of the label combination for stratification
        stratify_column = df[label_columns].apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
        
        subset_df, _ = train_test_split(
            df,
            train_size=percentage,
            stratify=stratify_column,
            random_state=random_state
        )
    except (ValueError, Exception) as e:
        # If stratification fails (some combinations too rare), do random sampling
        print(f"  Warning: Stratification failed for {percentage*100:.1f}%, using random sampling instead")
        n_samples = max(1, int(len(df) * percentage))
        subset_df = df.sample(n=n_samples, random_state=random_state)
    
    return subset_df.reset_index(drop=True)


def calculate_dataset_hash(df: pd.DataFrame, label_columns: list) -> str:
    """Calculate hash of a dataset based on its content.
    
    Uses a simple deterministic approach based on dataset size and random_state.
    This ensures that subsets created with the same percentage and random_state
    always get the same hash, regardless of implementation details.
    
    Args:
        df: DataFrame to hash
        label_columns: List of label column names
        
    Returns:
        8-character hex hash
    """
    # Use dataset size as the primary identifier
    # This is deterministic: same percentage → same size → same hash
    size_str = f"goemotions_train_size_{len(df)}_labels_{len(label_columns)}_seed_42"
    
    # Calculate hash
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
            'num_epochs': 2,  # Faster training for experiments
            'batch_size': 32,  # Larger batch size for efficiency
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
                     cache_dir: str = "cache"):
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
            auto_use_cache=True,
            cache_dir=cache_dir,  # Where to search for existing cache
            multi_label=True,
            auto_save_results=True,  # Save experiment results to output_dir
            output_dir=output_dir,  # Experiment outputs go here
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
            auto_save_results=True,
            cache_dir=cache_dir,  # Where to search for existing cache
            output_dir=output_dir,  # Experiment outputs go here
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
            'fusion_hidden_dims': [64, 32],  # Smaller network to prevent overfitting
            'ml_lr': 1e-5,  # Low LR for stable RoBERTa fine-tuning
            'fusion_lr': 5e-4,  # Lower LR for more stable fusion training
            'num_epochs': 10,  # Fewer epochs (early stopping at ~6-7 would be ideal)
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
        save_intermediate_llm_predictions=False  # Disable intermediate LLM prediction saving
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
    base_output_dir: str = "outputs/goemotions_availability_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
) -> dict:
    """Evaluate fusion ensemble with a specific percentage of training data.
    
    Args:
        df_train_full: Full training DataFrame
        df_val: Validation DataFrame (kept constant)
        df_test: Test DataFrame (kept constant)
        percentage: Percentage of training data to use (0.0 to 1.0)
        text_column: Name of text column
        label_columns: List of label column names
        llm_provider: LLM provider to use ('openai' or 'deepseek')
        base_output_dir: Base output directory for experiments
        auto_use_cache: Whether to use cached LLM predictions
        cache_dir: Directory for LLM prediction caches
        evaluate_baselines: Whether to evaluate individual models (RoBERTa, LLM) separately
        
    Returns:
        Dictionary with evaluation results
    """
    percentage_str = f"{int(percentage * 100)}pct" if percentage >= 0.01 else f"{percentage * 100:.1f}pct"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"goemotions_{percentage_str}_{timestamp}"
    output_dir = os.path.join(base_output_dir, f"train_{percentage_str}")
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: Training with {percentage_str} of training data")
    print("="*80)
    
    # Create stratified subset of training data
    print(f"\nCreating stratified subset with {percentage:.0%} of training data...")
    df_train_subset = create_stratified_subset(df_train_full, percentage, label_columns)
    print(f"  Subset size: {len(df_train_subset)} samples")
    
    # Verify class distribution
    print("\nClass distribution in subset:")
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
    llm_model = create_llm_model(text_column, label_columns, llm_provider, output_dir, experiment_name, cache_dir=cache_dir)
    
    # Create fusion ensemble
    print("Creating fusion ensemble...")
    fusion = create_fusion_ensemble(ml_model, llm_model, output_dir, experiment_name, auto_use_cache, cache_dir,
                                   val_llm_cache_path=val_cache_path, test_llm_cache_path=test_cache_path)
    
    # Dictionary to store all results
    all_model_results = {}
    
    # ===== Check and load cached RoBERTa model if available =====
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
    
    # ===== BASELINE 1: Evaluate RoBERTa alone (if requested) =====
    if evaluate_baselines:
        print("\n" + "-"*80)
        print("BASELINE 1: Evaluating RoBERTa (ML) model alone...")
        print("-"*80)
        
        # Train RoBERTa if not loaded from cache
        if not ml_model_loaded_from_cache:
            print(f"Training RoBERTa on {len(df_train_subset)} samples...")
            ml_training_result = ml_model.fit(df_train_subset, df_val)
        else:
            ml_training_result = {
                "model_name": "roberta-base",
                "cached": True,
                "cache_path": ml_cache_path
            }
        
        # Evaluate on test set
        print("Evaluating RoBERTa on test set...")
        ml_test_result = ml_model.predict(df_test)
        ml_test_metrics = ml_test_result.metadata.get('metrics', {}) if ml_test_result.metadata else {}
        
        print("\nRoBERTa Test Results:")
        print(f"  Accuracy: {ml_test_metrics.get('accuracy', 0.0):.4f}")
        print(f"  F1 Score (Weighted): {ml_test_metrics.get('f1_weighted', 0.0):.4f}")
        print(f"  Precision (Weighted): {ml_test_metrics.get('precision_weighted', 0.0):.4f}")
        print(f"  Recall (Weighted): {ml_test_metrics.get('recall_weighted', 0.0):.4f}")
        
        all_model_results['roberta'] = {
            'test_metrics': ml_test_metrics,
            'training_result': ml_training_result
        }
    
    # ===== BASELINE 2: Evaluate LLM alone (if requested) =====
    # NOTE: LLM predictions are cached and identical for all experiments (same test set)
    # LLM always uses FULL training data for consistency across all experiments
    if evaluate_baselines:
        print("\n" + "-"*80)
        print(f"BASELINE 2: Evaluating {llm_provider.upper()} (LLM) model alone...")
        print("-"*80)
        
        # LLM predictions are cached (test set is constant across all experiments)
        # Using FULL training data (not subset) for LLM baseline
        print(f"Evaluating {llm_provider.upper()} on test set with FULL training data (with caching enabled)...")
        llm_test_result = llm_model.predict(train_df=df_train_full, test_df=df_test.sample(n=2))
        llm_test_metrics = llm_test_result.metadata.get('metrics', {}) if llm_test_result.metadata else {}
        
        print(f"\n{llm_provider.upper()} Test Results (trained on {len(df_train_full)} samples):")
        print(f"  Accuracy: {llm_test_metrics.get('accuracy', 0.0):.4f}")
        print(f"  F1 Score: {llm_test_metrics.get('f1', 0.0):.4f}")
        print(f"  Precision: {llm_test_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {llm_test_metrics.get('recall', 0.0):.4f}")
        
        all_model_results['llm'] = {
            'test_metrics': llm_test_metrics,
            'model_name': llm_provider
        }
    
    # ===== MAIN: Train and evaluate Fusion Ensemble =====
    print("\n" + "-"*80)
    print("FUSION ENSEMBLE: Training combined RoBERTa + LLM model...")
    print("-"*80)
    
    # Train RoBERTa if not already loaded/trained from cache or baseline evaluation
    if not ml_model_loaded_from_cache and not evaluate_baselines:
        print(f"🔄 Training RoBERTa on {len(df_train_subset)} samples...")
        ml_model.fit(df_train_subset, df_val)
    elif ml_model_loaded_from_cache:
        print(f"✅ Using cached RoBERTa model (already loaded)")
    else:
        print(f"✅ Using RoBERTa model from baseline evaluation")
    
    # Train fusion ensemble
    print(f"\n🔧 Training fusion ensemble on {len(df_train_subset)} samples...")
    training_result = fusion.fit(df_train_subset, df_val)
    
    print("\nTraining completed!")
    print(f"  ML model trained: {training_result.get('ml_model_trained', False)}")
    print(f"  Fusion MLP trained: {training_result.get('fusion_mlp_trained', False)}")
    
    # Evaluate on test set
    print("\nEvaluating fusion ensemble on test set...")
    test_result = fusion.predict(df_test)
    
    # Extract metrics
    test_metrics = test_result.metadata.get('metrics', {}) if test_result.metadata else {}
    
    print("\nFusion Ensemble Test Results:")
    print(f"  Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"  F1 Score: {test_metrics.get('f1', 0.0):.4f}")
    print(f"  Precision: {test_metrics.get('precision', 0.0):.4f}")
    print(f"  Recall: {test_metrics.get('recall', 0.0):.4f}")
    
    all_model_results['fusion'] = {
        'test_metrics': test_metrics,
        'training_result': training_result
    }
    
    # ===== COMPARISON SUMMARY =====
    if evaluate_baselines:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        print(f"\nModel Comparison on Test Set ({len(df_test)} samples):")
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
                # RoBERTa uses weighted metrics, others use regular metrics
                if model_name == 'roberta':
                    f1 = metrics.get('f1_weighted', 0.0)
                    precision = metrics.get('precision_weighted', 0.0)
                    recall = metrics.get('recall_weighted', 0.0)
                else:
                    f1 = metrics.get('f1', 0.0)
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                
                print(f"{display_name:<20} {acc:>10.4f} {f1:>10.4f} {precision:>12.4f} {recall:>10.4f}")
        
        # Calculate improvement
        if 'roberta' in all_model_results and 'fusion' in all_model_results:
            roberta_acc = all_model_results['roberta']['test_metrics'].get('accuracy', 0.0)
            fusion_acc = all_model_results['fusion']['test_metrics'].get('accuracy', 0.0)
            improvement = ((fusion_acc - roberta_acc) / roberta_acc * 100) if roberta_acc > 0 else 0
            
            print("\n" + "-" * 55)
            print(f"Fusion vs RoBERTa: {improvement:+.2f}% accuracy improvement")
    
    # Compile results
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
    data_dir: str = "data/goemotions",
    percentages: list = None,
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/goemotions_availability_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
):
    """Run experiments with different training data percentages.
    
    Args:
        data_dir: Directory containing GoEmotions data
        percentages: List of training data percentages to test (default: 10%-100% in 10% steps)
        llm_provider: LLM provider ('openai' or 'deepseek')
        base_output_dir: Base directory for experiment outputs
        auto_use_cache: Whether to use cached LLM predictions
        cache_dir: Directory for LLM prediction caches
        evaluate_baselines: Whether to evaluate individual models (RoBERTa, LLM) separately
    """
    # Default percentages: 10% to 100% in 10% steps
    if percentages is None:
        percentages = [i * 0.1 for i in range(1, 11)]    # 10% to 100%
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load datasets
    df_train, df_val, df_test = load_datasets(data_dir)
    
    # Detect text and label columns (GoEmotions uses 'text' and emotion labels)
    text_column = 'text'
    # For GoEmotions: label columns are the emotion names
    label_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    print(f"\nDataset configuration:")
    print(f"  Text column: {text_column}")
    print(f"  Label columns: {len(label_columns)} emotions")
    
    # Store all results
    all_results = []
    
    # Run experiments for each percentage
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
    
    # Save consolidated results
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    
    summary_data = []
    for result in all_results:
        # Format percentage for display (0.1% or 1% or 10%)
        pct = result['percentage'] * 100
        if pct < 1:
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = f"{int(pct)}%"
        
        # Add fusion results
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
        
        # Add baseline results if available
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
    
    # Save summary to CSV
    summary_file = os.path.join(base_output_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n📊 Summary saved to: {summary_file}")
    
    # Save detailed results to JSON
    detailed_file = os.path.join(base_output_dir, f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"📋 Detailed results saved to: {detailed_file}")
    
    return all_results, summary_df


if __name__ == "__main__":
    # Environment-variable based runner (no argparse required).
    # Supported env vars:
    #   DATA_DIR - path to GoEmotions data (default: data/goemotions)
    #   PERCENTAGES - comma- or whitespace-separated list of percentages (e.g. "0.001,0.002,0.1,0.2")
    #   LLM_PROVIDER - 'openai' or 'deepseek' (default: openai)
    #   OUTPUT_DIR - base output directory (default: outputs/goemotions_availability_experiments)
    #   NO_CACHE - set to '1' or 'true' to disable cache (default: not set)
    #   CACHE_DIR - cache directory (default: cache)
    #   NO_BASELINES - set to '1' or 'true' to skip baseline evaluations

    def _parse_percentages(s: str):
        """Parse a comma- or whitespace-separated string of floats into a list of floats."""
        import re
        if not s:
            # Default: 10% to 100% in 10% steps
            percentages = [i * 0.1 for i in range(1, 11)]    # 10% to 100%
            return percentages
        parts = [p for p in re.split('[,\s]+', s.strip()) if p]
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"Invalid PERCENTAGES value: {s}")

    data_dir = os.getenv('DATA_DIR', 'data/goemotions')
    percentages = _parse_percentages(os.getenv('PERCENTAGES', ''))
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    base_output_dir = os.getenv('OUTPUT_DIR', 'outputs/goemotions_availability_experiments')
    no_cache_env = os.getenv('NO_CACHE', '')
    auto_use_cache = not (no_cache_env.lower() in ('1', 'true', 'yes'))
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    no_baselines_env = os.getenv('NO_BASELINES', '1')  # Default to skipping baselines for testing
    evaluate_baselines = not (no_baselines_env.lower() in ('1', 'true', 'yes'))

    print("="*80)
    print("DATA AVAILABILITY EXPERIMENT - FUSION ENSEMBLE ON GOEMOTIONS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    pct_display = [f"{p*100:.1f}%" if p < 0.01 else f"{int(p*100)}%" for p in percentages]
    print(f"  Training percentages: {pct_display}")
    print(f"  LLM provider: {llm_provider}")
    print(f"  Output directory: {base_output_dir}")
    print(f"  Auto-use cache: {auto_use_cache}")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Evaluate baselines: {evaluate_baselines}")

    # Run experiments
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
