"""
Evaluation script for testing Fusion Ensemble performance with different training data sizes.

Tests how the ensemble performs with 20%, 40%, 60%, and 80% of training data,
while keeping validation and test sets constant.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.deepseek_classifier import DeepSeekClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.config.settings import Config
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType, ClassificationType


def load_datasets(data_dir: str = "data/ag_news"):
    """Load AG News train, validation and test datasets."""
    print(f"Loading datasets from {data_dir}...")
    
    df_train = pd.read_csv(os.path.join(data_dir, "ag_train_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(data_dir, "ag_val_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(data_dir, "ag_test_balanced.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    
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
    
    # For multi-class, use the first label column for stratification
    stratify_column = df[label_columns].idxmax(axis=1)
    
    # Split to get the desired percentage
    subset_df, _ = train_test_split(
        df,
        train_size=percentage,
        stratify=stratify_column,
        random_state=random_state
    )
    
    return subset_df.reset_index(drop=True)


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
            'batch_size': 8,
        }
    )
    
    return RoBERTaClassifier(
        config=ml_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=False,
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
            enable_cache=True,
            cache_dir=cache_dir,
            multi_label=False,
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
            multi_label=False,
            auto_save_results=True,
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
            'fusion_hidden_dims': [64, 32],  # Smaller network to prevent overfitting
            'ml_lr': 1e-5,  # Low LR for stable RoBERTa fine-tuning
            'fusion_lr': 5e-4,  # Lower LR for more stable fusion training
            'num_epochs': 10,  # Fewer epochs (early stopping at ~6-7 would be ideal)
            'batch_size': 8,
            'classification_type': 'multi_class',
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
    base_output_dir: str = "outputs/data_availability_experiments",
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
    percentage_str = f"{int(percentage * 100)}pct"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"ag_news_{percentage_str}_{timestamp}"
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
    
    # Setup cache paths
    ml_cache_path = os.path.join(cache_dir, "experimente", f"fusion_roberta_model_{percentage_str}")
    llm_cache_base = os.path.join(cache_dir, "experimente", f"fusion_{llm_provider}_cache_{percentage_str}")
    val_cache_path = os.path.join(llm_cache_base, "val")
    test_cache_path = os.path.join(llm_cache_base, "test")
    
    # Create models
    print("\nCreating models...")
    ml_model = create_ml_model(text_column, label_columns, output_dir, experiment_name, auto_save_path=ml_cache_path)
    llm_model = create_llm_model(text_column, label_columns, llm_provider, output_dir, experiment_name, cache_dir=llm_cache_base)
    
    # Create fusion ensemble
    print("Creating fusion ensemble...")
    fusion = create_fusion_ensemble(ml_model, llm_model, output_dir, experiment_name, auto_use_cache, cache_dir,
                                   val_llm_cache_path=val_cache_path, test_llm_cache_path=test_cache_path)
    
    # Dictionary to store all results
    all_model_results = {}
    
    # ===== BASELINE 1: Evaluate RoBERTa alone (if requested) =====
    if evaluate_baselines:
        print("\n" + "-"*80)
        print("BASELINE 1: Evaluating RoBERTa (ML) model alone...")
        print("-"*80)
        
        # Train RoBERTa
        print(f"Training RoBERTa on {len(df_train_subset)} samples...")
        ml_training_result = ml_model.fit(df_train_subset, df_val)
        
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
    if evaluate_baselines:
        print("\n" + "-"*80)
        print(f"BASELINE 2: Evaluating {llm_provider.upper()} (LLM) model alone...")
        print("-"*80)
        
        # LLM doesn't need training, just predict
        print(f"Generating {llm_provider.upper()} predictions on test set...")
        llm_test_result = llm_model.predict(train_df=df_train_subset, test_df=df_test)
        llm_test_metrics = llm_test_result.metadata.get('metrics', {}) if llm_test_result.metadata else {}
        
        print(f"\n{llm_provider.upper()} Test Results:")
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
    
    # Train fusion ensemble
    print(f"\nTraining fusion ensemble on {len(df_train_subset)} samples...")
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
    data_dir: str = "data/ag_news",
    percentages: list = [0.2, 0.4, 0.6, 0.8, 1.0],
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/data_availability_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
):
    """Run experiments with different training data percentages.
    
    Args:
        data_dir: Directory containing AG News data
        percentages: List of training data percentages to test
        llm_provider: LLM provider ('openai' or 'deepseek')
        base_output_dir: Base directory for experiment outputs
        auto_use_cache: Whether to use cached LLM predictions
        cache_dir: Directory for LLM prediction caches
        evaluate_baselines: Whether to evaluate individual models (RoBERTa, LLM) separately
    """
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load datasets
    df_train, df_val, df_test = load_datasets(data_dir)
    
    # Detect text and label columns (AG News uses 'description' and 'label_1-4')
    text_column = 'description' if 'description' in df_train.columns else 'text'
    # For AG News: label columns are label_1, label_2, label_3, label_4
    label_columns = [col for col in df_train.columns if col.startswith('label_')]
    if not label_columns:
        label_columns = [col for col in df_train.columns if col != text_column]
    
    print(f"\nDataset configuration:")
    print(f"  Text column: {text_column}")
    print(f"  Label columns: {label_columns}")
    
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
        # Add fusion results
        # Fusion uses 'f1', 'precision', 'recall' (without suffixes)
        fusion_metrics = result['test_metrics']
        summary_data.append({
            'percentage': f"{int(result['percentage']*100)}%",
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
                # RoBERTa uses 'f1_weighted', 'precision_weighted', 'recall_weighted'
                ml_metrics = all_models['roberta']['test_metrics']
                summary_data.append({
                    'percentage': f"{int(result['percentage']*100)}%",
                    'model': 'RoBERTa',
                    'train_samples': result['train_samples'],
                    'accuracy': ml_metrics.get('accuracy', 0.0),
                    'f1': ml_metrics.get('f1_weighted', 0.0),
                    'precision': ml_metrics.get('precision_weighted', 0.0),
                    'recall': ml_metrics.get('recall_weighted', 0.0)
                })
            
            if 'llm' in all_models:
                # LLM uses 'f1', 'precision', 'recall' (without suffixes)
                llm_metrics = all_models['llm']['test_metrics']
                summary_data.append({
                    'percentage': f"{int(result['percentage']*100)}%",
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
    #   DATA_DIR - path to AG News data (default: data/ag_news)
    #   PERCENTAGES - comma- or whitespace-separated list of percentages (e.g. "0.2,0.4,1.0")
    #   LLM_PROVIDER - 'openai' or 'deepseek' (default: openai)
    #   OUTPUT_DIR - base output directory (default: outputs/data_availability_experiments)
    #   NO_CACHE - set to '1' or 'true' to disable cache (default: not set)
    #   CACHE_DIR - cache directory (default: cache)
    #   NO_BASELINES - set to '1' or 'true' to skip baseline evaluations

    def _parse_percentages(s: str):
        """Parse a comma- or whitespace-separated string of floats into a list of floats."""
        import re
        if not s:
            return [0.2, 0.4, 0.6, 0.8, 1.0]
        parts = [p for p in re.split('[,\s]+', s.strip()) if p]
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"Invalid PERCENTAGES value: {s}")

    data_dir = os.getenv('DATA_DIR', 'data/ag_news')
    percentages = _parse_percentages(os.getenv('PERCENTAGES', '0.2,0.4,0.6,0.8,1.0'))
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    base_output_dir = os.getenv('OUTPUT_DIR', 'outputs/data_availability_experiments')
    no_cache_env = os.getenv('NO_CACHE', '')
    auto_use_cache = not (no_cache_env.lower() in ('1', 'true', 'yes'))
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    no_baselines_env = os.getenv('NO_BASELINES', '')
    evaluate_baselines = not (no_baselines_env.lower() in ('1', 'true', 'yes'))

    print("="*80)
    print("DATA AVAILABILITY EXPERIMENT - FUSION ENSEMBLE (env-vars)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Training percentages: {[f'{int(p*100)}%' for p in percentages]}")
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
