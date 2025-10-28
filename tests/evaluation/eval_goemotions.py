"""
Evaluation script for testing Fusion Ensemble performance on GoEmotions dataset.

This script evaluates the fusion ensemble on the GoEmotions multi-label emotion classification task
with different training data sizes (starting with 1% for pipeline testing).
"""

import sys
import os
import pandas as pd
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


def load_datasets(data_dir: str = "data/goemotions"):
    """Load GoEmotions train, validation and test datasets."""
    print(f"Loading datasets from {data_dir}...")
    
    train_path = Path(data_dir) / "goemotions_all_train_balanced.csv"
    val_path = Path(data_dir) / "goemotions_all_val_balanced.csv"
    test_path = Path(data_dir) / "goemotions_all_test_balanced.csv"
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"✅ Loaded {len(df_train)} train, {len(df_val)} val, {len(df_test)} test samples")
    return df_train, df_val, df_test


def get_emotion_label_columns():
    """Return the list of emotion label columns in GoEmotions dataset."""
    # Based on the CSV structure, these are the emotion labels (excluding metadata columns)
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    return emotion_labels


def create_stratified_subset(df: pd.DataFrame, percentage: float = None, n_samples: int = None, label_columns: list = None, random_state: int = 42) -> pd.DataFrame:
    """Create a stratified subset of the dataframe for multi-label data.
    
    For multi-label classification, we use a simple random sample while trying to
    maintain some balance across the most common labels.
    
    Args:
        df: Input dataframe
        percentage: Percentage of data to sample (0-1)
        n_samples: Exact number of samples to take (overrides percentage)
        label_columns: List of label columns for statistics
        random_state: Random seed
    """
    if percentage is not None and percentage >= 1.0:
        return df.copy()
    
    if n_samples is not None:
        target_size = min(n_samples, len(df))
        size_description = f"{target_size} samples"
    else:
        target_size = int(len(df) * percentage)
        size_description = f"{percentage*100:.1f}% = {target_size} samples"
    
    # For multi-label, simple random sampling is often sufficient
    # Stratification is complex for multi-label, so we use random sampling
    subset_df = df.sample(n=target_size, random_state=random_state)
    
    print(f"Created subset: {size_description}")
    
    # Print label distribution statistics if label_columns provided
    if label_columns:
        label_counts = subset_df[label_columns].sum()
        print(f"Label distribution in subset:")
        print(f"  - Total labels: {label_counts.sum()}")
        print(f"  - Avg labels per sample: {label_counts.sum() / len(subset_df):.2f}")
        print(f"  - Most common: {label_counts.nlargest(3).to_dict()}")
    
    return subset_df


def create_ml_model(text_column: str, label_columns: list, output_dir: str, experiment_name: str, 
                    auto_save_path: str = None) -> RoBERTaClassifier:
    """Create RoBERTa classifier for multi-label emotion classification."""
    config = ModelConfig(
        model_name="roberta-base",
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "max_length": 128,
            "batch_size": 16,
            "num_epochs": 3,
            "learning_rate": 2e-5,
            "classification_type": "multi_label"
        }
    )
    # Explicitly mark model config as multi-label so components that read
    # `config.label_type` (e.g. PromptEngineer) detect the correct task.
    try:
        config.label_type = "multiple"
    except Exception:
        # Be defensive: if ModelConfig is immutable or doesn't accept new attrs,
        # write into parameters as a fallback (already present above).
        config.parameters["classification_type"] = "multi_label"
    return RoBERTaClassifier(
        config=config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,  # Multiclass now
        output_dir=output_dir,
        experiment_name=experiment_name,
        auto_save_path=auto_save_path
    )


def create_llm_model(text_column: str, label_columns: list, 
                     provider: str = 'openai', 
                     output_dir: str = "outputs",
                     experiment_name: str = "llm",
                     cache_dir: str = "cache",
                     multi_label: bool = True):
    """Create LLM classifier for multi-label or multi-class emotion classification.

    Args:
        text_column: name of the text column
        label_columns: list of label column names
        provider: 'openai' or 'deepseek'
        output_dir: output directory
        experiment_name: experiment name
        cache_dir: cache directory
        multi_label: whether this is a multi-label task (True) or single-label/multi-class (False)
    """
    config = Config()
    
    llm_config = ModelConfig(
        model_name="gpt-4o-mini" if provider == 'openai' else "deepseek-chat",
        model_type=ModelType.LLM,
        parameters={
            "model": "gpt-4o-mini" if provider == 'openai' else "deepseek-chat",
            "temperature": 0.0,
            "max_tokens": 500,
            "few_shot_examples": 10,
            "classification_type": "multi_label" if multi_label else "multi_class"
        }
    )
    # Ensure config exposes label_type for prompt engineering
    try:
        llm_config.label_type = "multiple" if multi_label else "single"
    except Exception:
        llm_config.parameters["classification_type"] = "multi_label" if multi_label else "multi_class"
    
    if provider == 'openai':
        return OpenAIClassifier(
            config=llm_config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode="few_shot",
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_use_cache=True,
            cache_dir=cache_dir
        )
    elif provider == 'deepseek':
        return DeepSeekClassifier(
            config=llm_config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode="few_shot",
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_use_cache=True,
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_fusion_ensemble(ml_model, llm_model, output_dir: str, experiment_name: str, 
                          auto_use_cache: bool = True, cache_dir: str = "cache",
                          val_llm_cache_path: str = None, test_llm_cache_path: str = None) -> FusionEnsemble:
    """Create Fusion Ensemble for multi-label emotion classification."""
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            "fusion_hidden_dims": [64, 32],
            "fusion_epochs": 10,
            "fusion_batch_size": 32,
            "fusion_learning_rate": 0.001,
            "early_stopping_patience": 3,
            "use_hidden_states": True,
            "val_llm_cache_path": val_llm_cache_path or "",
            "test_llm_cache_path": test_llm_cache_path or "",
            "classification_type": "multi_label"
        }
    )
    
    return FusionEnsemble(
        ensemble_config=ensemble_config,
        output_dir=output_dir,
        experiment_name=experiment_name,
        auto_save_results=True,
        save_intermediate_llm_predictions=True,
        auto_use_cache=auto_use_cache,
        cache_dir=cache_dir
    )


def evaluate_with_data_percentage(
    df_train_full: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    percentage: float = None,
    n_samples: int = None,
    text_column: str = None,
    label_columns: list = None,
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/goemotions_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True
) -> dict:
    """
    Evaluate fusion ensemble and baselines with a specific percentage or number of samples.
    
    Args:
        df_train_full: Full training dataset
        df_val: Validation dataset
        df_test: Test dataset
        percentage: Percentage of training data to use (optional)
        n_samples: Exact number of samples to use (optional, overrides percentage)
        text_column: Name of text column
        label_columns: List of label columns
        llm_provider: LLM provider to use ('openai' or 'deepseek')
        base_output_dir: Base directory for outputs
        auto_use_cache: Whether to use automatic caching
        cache_dir: Cache directory
        evaluate_baselines: Whether to evaluate baseline models
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    if n_samples is not None:
        print(f"🚀 EVALUATING WITH {n_samples} SAMPLES")
    else:
        print(f"🚀 EVALUATING WITH {percentage*100:.1f}% OF TRAINING DATA")
    print("="*80)
    
    # Create subset of training data
    if n_samples is not None:
        df_train = create_stratified_subset(df_train_full, n_samples=n_samples, label_columns=label_columns)
        df_val = create_stratified_subset(df_val, n_samples=n_samples, label_columns=label_columns, random_state=43)
        df_test = create_stratified_subset(df_test, n_samples=n_samples, label_columns=label_columns, random_state=44)
        size_str = f"{n_samples}samples"
    else:
        df_train = create_stratified_subset(df_train_full, percentage=percentage, label_columns=label_columns)
        
        # Also subset validation and test for small test runs (≤ 1%)
        if percentage <= 0.01:  # For 1% or less training, also subset val and test
            print(f"\n📊 Creating {percentage*100:.1f}% subsets for validation and test sets as well...")
            df_val = create_stratified_subset(df_val, percentage=percentage, label_columns=label_columns, random_state=43)
            df_test = create_stratified_subset(df_test, percentage=percentage, label_columns=label_columns, random_state=44)
        size_str = f"{int(percentage*100)}pct"
    
    # Create experiment directory
    experiment_name = f"goemotions_{llm_provider}_{size_str}"
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup cache paths
    cache_base = os.path.join(cache_dir, f"goemotions_{llm_provider}_cache_{size_str}")
    val_cache_path = os.path.join(cache_base, "val")
    test_cache_path = os.path.join(cache_base, "test")
    os.makedirs(cache_base, exist_ok=True)
    
    results = {
        'percentage': percentage if percentage is not None else None,
        'n_samples': n_samples if n_samples is not None else None,
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_test),
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Train and evaluate ML model (baseline)
    if evaluate_baselines:
        print("\n" + "-"*80)
        print("1️⃣  TRAINING ML MODEL (RoBERTa baseline)")
        print("-"*80)
        
        ml_model = create_ml_model(
            text_column=text_column,
            label_columns=label_columns,
            output_dir=os.path.join(output_dir, "ml_model"),
            experiment_name=f"ml_{experiment_name}"
        )
        
        ml_train_result = ml_model.fit(df_train, df_val)
        print(f"✅ ML training completed")
        
        # Evaluate on test set
        ml_test_result = ml_model.predict(df_test)
        ml_metrics = ml_test_result.metadata.get('metrics', {})
        
        print(f"\n📊 ML Model Test Results:")
        print(f"   Accuracy: {ml_metrics.get('accuracy', 0):.4f}")
        print(f"   Macro F1: {ml_metrics.get('macro_f1', 0):.4f}")
        print(f"   Micro F1: {ml_metrics.get('micro_f1', 0):.4f}")
        
        results['ml_model'] = {
            'test_accuracy': ml_metrics.get('accuracy', 0),
            'test_macro_f1': ml_metrics.get('macro_f1', 0),
            'test_micro_f1': ml_metrics.get('micro_f1', 0),
            'test_hamming_loss': ml_metrics.get('hamming_loss', 0)
        }
    else:
        print("\n⏩ Skipping ML baseline evaluation")
        ml_model = create_ml_model(
            text_column=text_column,
            label_columns=label_columns,
            output_dir=os.path.join(output_dir, "ml_model"),
            experiment_name=f"ml_{experiment_name}"
        )
    
    # 2. Evaluate LLM model (baseline)
    if evaluate_baselines:
        print("\n" + "-"*80)
        print(f"2️⃣  EVALUATING LLM MODEL ({llm_provider.upper()} baseline)")
        print("-"*80)
        
        llm_model = create_llm_model(
            text_column=text_column,
            label_columns=label_columns,
            provider=llm_provider,
            output_dir=os.path.join(output_dir, "llm_model"),
            experiment_name=f"llm_{experiment_name}",
            cache_dir=cache_dir,
            multi_label=True  # Multiclass now
        )
        
        # LLM prediction with auto-cache
        llm_test_result = llm_model.predict(train_df=df_train, test_df=df_test)
        llm_metrics = llm_test_result.metadata.get('metrics', {})
        
        print(f"\n📊 LLM Model Test Results:")
        print(f"   Accuracy: {llm_metrics.get('accuracy', 0):.4f}")
        print(f"   Macro F1: {llm_metrics.get('macro_f1', 0):.4f}")
        print(f"   Micro F1: {llm_metrics.get('micro_f1', 0):.4f}")
        
        results['llm_model'] = {
            'test_accuracy': llm_metrics.get('accuracy', 0),
            'test_macro_f1': llm_metrics.get('macro_f1', 0),
            'test_micro_f1': llm_metrics.get('micro_f1', 0),
            'test_hamming_loss': llm_metrics.get('hamming_loss', 0)
        }
    else:
        print("\n⏩ Skipping LLM baseline evaluation")
        llm_model = create_llm_model(
            text_column=text_column,
            label_columns=label_columns,
            provider=llm_provider,
            output_dir=os.path.join(output_dir, "llm_model"),
            experiment_name=f"llm_{experiment_name}",
            cache_dir=cache_dir
        )
    
    # 3. Train and evaluate Fusion Ensemble
    print("\n" + "-"*80)
    print("3️⃣  TRAINING FUSION ENSEMBLE")
    print("-"*80)
    
    fusion = create_fusion_ensemble(
        ml_model=ml_model,
        llm_model=llm_model,
        output_dir=os.path.join(output_dir, "fusion"),
        experiment_name=f"fusion_{experiment_name}",
        auto_use_cache=auto_use_cache,
        cache_dir=cache_dir,
        val_llm_cache_path=val_cache_path,
        test_llm_cache_path=test_cache_path
    )
    
    # Train fusion
    fusion_train_result = fusion.fit(df_train, df_val)
    print(f"✅ Fusion training completed")
    
    # Evaluate on test set
    print("\n🧪 Evaluating fusion on test set...")
    fusion_test_result = fusion.predict(df_test)
    fusion_metrics = fusion_test_result.metadata.get('metrics', {})
    
    print(f"\n📊 Fusion Ensemble Test Results:")
    print(f"   Accuracy: {fusion_metrics.get('accuracy', 0):.4f}")
    print(f"   Macro F1: {fusion_metrics.get('macro_f1', 0):.4f}")
    print(f"   Micro F1: {fusion_metrics.get('micro_f1', 0):.4f}")
    
    results['fusion'] = {
        'test_accuracy': fusion_metrics.get('accuracy', 0),
        'test_macro_f1': fusion_metrics.get('macro_f1', 0),
        'test_micro_f1': fusion_metrics.get('micro_f1', 0),
        'test_hamming_loss': fusion_metrics.get('hamming_loss', 0)
    }
    
    # Save results summary
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


def run_goemotions_experiments(
    data_dir: str = "data/goemotions",
    percentages: list = None,
    n_samples: int = 10,  # Default: 10 samples for quick testing
    llm_provider: str = 'openai',
    base_output_dir: str = "outputs/goemotions_experiments",
    auto_use_cache: bool = True,
    cache_dir: str = "cache",
    evaluate_baselines: bool = True,
    train_on_all: bool = False
):
    """
    Run experiments on GoEmotions dataset with different training data sizes.
    
    Args:
        data_dir: Directory containing GoEmotions CSV files
        percentages: List of training data percentages to evaluate (optional)
        n_samples: Number of samples to use (default: 10 for quick testing)
        llm_provider: LLM provider to use ('openai' or 'deepseek')
        base_output_dir: Base directory for all experiment outputs
        auto_use_cache: Whether to use automatic caching
        cache_dir: Cache directory
        evaluate_baselines: Whether to evaluate baseline models
    """
    print("="*80)
    print("🎭 GOEMOTIONS MULTI-LABEL EMOTION CLASSIFICATION EXPERIMENTS")
    print("="*80)
    print(f"📁 Data directory: {data_dir}")
    print(f"🤖 LLM Provider: {llm_provider}")
    if n_samples is not None:
        print(f"📊 Number of samples: {n_samples} (train, val, test)")
    else:
        print(f"📊 Training percentages: {percentages}")
    print(f"💾 Cache enabled: {auto_use_cache}")
    print(f"📈 Evaluate baselines: {evaluate_baselines}")
    
    # Load datasets
    df_train, df_val, df_test = load_datasets(data_dir)

    # Optionally merge train, val and test into the training set
    if train_on_all:
        print("\n⚠️  TRAIN-ON-ALL enabled: merging train, val and test into the training set.")
        print("This will use all available labeled data for training. Evaluation will still use the original val/test splits,")
        print("but note that evaluating on examples that were included in training will produce optimistic metrics.")
        df_train = pd.concat([df_train, df_val, df_test], ignore_index=True)
        print(f"✅ New training size: {len(df_train)} (train+val+test merged)")
    
    # Get label columns
    text_column = 'text'
    label_columns = get_emotion_label_columns()
    
    print(f"\n📋 Dataset Info:")
    print(f"   Text column: {text_column}")
    print(f"   Label columns: {len(label_columns)} emotions")
    print(f"   Labels: {', '.join(label_columns[:5])}... (showing first 5)")
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    
    if n_samples is not None:
        # Run with fixed number of samples
        try:
            results = evaluate_with_data_percentage(
                df_train_full=df_train,
                df_val=df_val,
                df_test=df_test,
                n_samples=n_samples,
                text_column=text_column,
                label_columns=label_columns,
                llm_provider=llm_provider,
                base_output_dir=base_output_dir,
                auto_use_cache=auto_use_cache,
                cache_dir=cache_dir,
                evaluate_baselines=evaluate_baselines
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error evaluating with {n_samples} samples: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run with percentages
        for pct in percentages:
            try:
                results = evaluate_with_data_percentage(
                    df_train_full=df_train,
                    df_val=df_val,
                    df_test=df_test,
                    percentage=pct,
                    text_column=text_column,
                    label_columns=label_columns,
                    llm_provider=llm_provider,
                    base_output_dir=base_output_dir,
                    auto_use_cache=auto_use_cache,
                    cache_dir=cache_dir,
                    evaluate_baselines=evaluate_baselines
                )
                all_results.append(results)
            except Exception as e:
                print(f"\n❌ Error evaluating with {pct*100:.1f}% data: {e}")
                import traceback
                traceback.print_exc()
    
    # Save overall summary
    summary_file = os.path.join(base_output_dir, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"📁 Results saved to: {base_output_dir}")
    print(f"📊 Summary file: {summary_file}")
    
    # Print comparison table
    print("\n📊 RESULTS COMPARISON:")
    print("-"*80)
    print(f"{'Data Size':<12} {'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Micro F1':<12}")
    print("-"*80)
    
    for res in all_results:
        if res.get('n_samples') is not None:
            size_str = f"{res['n_samples']} samples"
        else:
            size_str = f"{res['percentage']*100:.1f}%"
        
        if 'ml_model' in res:
            print(f"{size_str:<12} {'ML (RoBERTa)':<15} "
                  f"{res['ml_model']['test_accuracy']:<12.4f} "
                  f"{res['ml_model']['test_macro_f1']:<12.4f} "
                  f"{res['ml_model']['test_micro_f1']:<12.4f}")
        
        if 'llm_model' in res:
            print(f"{'':<12} {f'LLM ({llm_provider})':<15} "
                  f"{res['llm_model']['test_accuracy']:<12.4f} "
                  f"{res['llm_model']['test_macro_f1']:<12.4f} "
                  f"{res['llm_model']['test_micro_f1']:<12.4f}")
        
        if 'fusion' in res:
            print(f"{'':<12} {'Fusion':<15} "
                  f"{res['fusion']['test_accuracy']:<12.4f} "
                  f"{res['fusion']['test_macro_f1']:<12.4f} "
                  f"{res['fusion']['test_micro_f1']:<12.4f}")
        
        print("-"*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Fusion Ensemble on GoEmotions dataset")
    parser.add_argument("--data-dir", type=str, default="data/goemotions",
                        help="Directory containing GoEmotions CSV files")
    parser.add_argument("--percentages", type=float, nargs="+", default=None,
                        help="List of training data percentages to evaluate (e.g., 0.01 0.05 0.1 1.0)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of samples to use for train, val, and test (default: 10)")
    parser.add_argument("--full", action="store_true",
                        help="Use full datasets (overrides --n-samples and --percentages)")
    parser.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "deepseek"],
                        help="LLM provider to use")
    parser.add_argument("--output-dir", type=str, default="outputs/goemotions_experiments",
                        help="Base directory for experiment outputs")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable automatic caching")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Cache directory")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline model evaluation")
    parser.add_argument("--train-on-all", action="store_true",
                        help="Merge train, val and test into the training set before training")
    
    args = parser.parse_args()
    
    # Determine which mode to use
    if args.full:
        # Use full datasets
        percentages = [1.0]
        n_samples = None
    elif args.percentages is not None:
        # Use specified percentages
        percentages = args.percentages
        n_samples = None
    else:
        # Use n_samples
        percentages = None
        n_samples = args.n_samples
    
    run_goemotions_experiments(
        data_dir=args.data_dir,
        percentages=percentages,
        n_samples=n_samples,
        llm_provider=args.llm_provider,
        base_output_dir=args.output_dir,
        auto_use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        evaluate_baselines=not args.no_baselines,
        train_on_all=args.train_on_all
    )
