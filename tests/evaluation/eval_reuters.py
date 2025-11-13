"""
Evaluation script for testing Fusion Ensemble performance with different training data sizes on Reuters-21578 (ModApte-10).

Assumes CSV files:
  /scratch/users/u19147/LabelFusion/data/reuters/{train.csv,val.csv,test.csv}

Each CSV should contain:
  - 'text' column with the article text
  - optional 'id' column
  - 10 binary label columns for ModApte-10:
    ['earn','acq','money-fx','grain','crude','trade','interest','ship','wheat','corn']

The script keeps validation and test sets fixed and varies the training subset size.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import hashlib

# Add project root to path (adjust if needed)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
"""
Reconstructed evaluation script for Reuters including baselines and fusion flow.

This version restores the RoBERTa and LLM baseline training/evaluation sections
and uses a default FEW_SHOT of 2 (can be overridden by the FEW_SHOT env var).

It also provides helpers expected by `precache_llm.py`.
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
    """Load Reuters train/val/test CSVs from the data directory."""
    print(f"Loading datasets from {data_dir}...")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_path).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(val_path).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = pd.read_csv(test_path).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Training samples:   {len(df_train)}")
    print(f"  Validation samples: {len(df_val)}")
    print(f"  Test samples:       {len(df_test)}")

    return df_train, df_val, df_test


def create_stratified_subset(df: pd.DataFrame, percentage: float, label_columns: list, random_state: int = 42) -> pd.DataFrame:
    """Create a stratified subset of the dataframe.

    For multi-label we create a small string key per-row. If stratification fails
    (rare combinations) we fall back to random sampling.
    """
    from sklearn.model_selection import train_test_split

    if percentage >= 1.0:
        return df.reset_index(drop=True)

    try:
        stratify_column = df[label_columns].apply(lambda row: ''.join(row.fillna(0).astype(int).astype(str)), axis=1)
        subset_df, _ = train_test_split(
            df,
            train_size=percentage,
            stratify=stratify_column,
            random_state=random_state
        )
    except Exception:
        print(f"  Warning: Stratification failed for {percentage*100:.1f}%, using random sampling instead")
        n_samples = max(1, int(len(df) * percentage))
        subset_df = df.sample(n=n_samples, random_state=random_state)

    return subset_df.reset_index(drop=True)


def calculate_dataset_hash(df: pd.DataFrame, label_columns: list) -> str:
    """Deterministic short hash for a dataset subset based on size and labels."""
    size_str = f"reuters_train_size_{len(df)}_labels_{len(label_columns)}_seed_42"
    return hashlib.sha256(size_str.encode('utf-8')).hexdigest()[:8]


def create_ml_model(text_column: str, label_columns: list, output_dir: str, experiment_name: str,
                    auto_save_path: str = None) -> RoBERTaClassifier:
    ml_config = ModelConfig(
        model_name='roberta-base',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'roberta-base',
            'max_length': 256,
            'learning_rate': 2e-5,
            'num_epochs': 2,
            'batch_size': 32,
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

    # Label distribution
    print("\nLabel distribution in subset (sum of positives per label):")
    subset_dist = df_train_subset[label_columns].sum()
    print(subset_dist)

    dataset_hash = calculate_dataset_hash(df_train_subset, label_columns)
    print(f"\nTraining subset hash: {dataset_hash}")

    ml_cache_path = os.path.join(cache_dir, f"fusion_roberta_model_{dataset_hash}")
    val_cache_path = os.path.join(cache_dir, "val")
    test_cache_path = os.path.join(cache_dir, "test")

    print("\nCreating models...")
    ml_model = create_ml_model(text_column, label_columns, output_dir, experiment_name, auto_save_path=ml_cache_path)

    few_shot_local = globals().get('FEW_SHOT_OVERRIDE', None)
    if few_shot_local is None:
        few_shot_local = 20
    llm_model = create_llm_model(text_column, label_columns, llm_provider, output_dir, experiment_name, cache_dir=cache_dir, few_shot_examples=int(few_shot_local))

    print("Creating fusion ensemble...")
    fusion = create_fusion_ensemble(
        ml_model, llm_model, output_dir, experiment_name, auto_use_cache, cache_dir,
        val_llm_cache_path=val_cache_path, test_llm_cache_path=test_cache_path
    )

    all_model_results = {}

    # RoBERTa baseline removed — only Fusion ensemble will be trained below.

    # ===== Baseline: LLM-only few-shot =====
    if False:
        print("\n" + "-"*80)
        print("BASELINE: LLM few-shot - evaluating on tiny sample + full test")
        print("-"*80)

        # For LLM baseline use a tiny random train sample of few_shot_local
        train_small = df_train_subset.sample(n=min(int(few_shot_local), max(1, len(df_train_subset))), random_state=42)
        llm_result_val = llm_model.predict(train_df=train_small, test_df=df_val)
        llm_result_test = llm_model.predict(train_df=train_small, test_df=df_test)

        llm_metrics = llm_result_test.metadata.get('metrics', {}) if llm_result_test.metadata else {}
        all_model_results['llm'] = {'test_metrics': llm_metrics}

    # ===== Fusion =====
    #print("\n" + "-"*80)
    #print("FUSION ENSEMBLE: Training combined RoBERTa + LLM model...")
    #print("-"*80)

    ml_train_on_full = os.getenv('ML_TRAIN_ON_FULL', '').lower() in ('1', 'true', 'yes')

    fusion_train_df = df_train_full if ml_train_on_full else df_train_subset
    training_result = fusion.fit(fusion_train_df, df_val)

    test_result = fusion.predict(df_test)
    test_metrics = test_result.metadata.get('metrics', {}) if test_result.metadata else {}

    all_model_results['fusion'] = {'test_metrics': test_metrics, 'training_result': training_result}

    # Summary printing
    print("\nFusion Ensemble Test Results:")
    print(f"  Accuracy:   {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"  F1 Score:   {test_metrics.get('f1', 0.0):.4f}")

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
        'test_predictions': len(getattr(test_result, 'predictions', [])),
        'all_models': all_model_results
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
    if percentages is None:
        percentages = [i * 0.1 for i in range(1, 11)]

    os.makedirs(base_output_dir, exist_ok=True)
    df_train, df_val, df_test = load_datasets(data_dir)

    text_column = 'text'
    candidate_labels = [c for c in df_train.columns if c != text_column and c != 'id']
    label_columns = []
    for c in candidate_labels:
        if c.lower() == 'id' or c.lower().startswith('id_'):
            continue
        ser = df_train[c]
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

    summary_data = []
    for result in all_results:
        pct = result['percentage'] * 100
        pct_str = f"{pct:.1f}%" if pct < 1 else f"{int(pct)}%"
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

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(base_output_dir, f"summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    detailed_file = os.path.join(base_output_dir, f"detailed_results_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results, summary_df


if __name__ == "__main__":
    def _parse_percentages(s: str):
        import re
        if not s:
            return [i * 0.1 for i in range(1, 11)]
        parts = [p for p in re.split(r'[,\s]+', s.strip()) if p]
        return [float(p) for p in parts]

    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    percentages = _parse_percentages(os.getenv('PERCENTAGES', ''))
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    base_output_dir = os.getenv('OUTPUT_DIR', 'outputs/reuters_availability_experiments')
    no_cache_env = os.getenv('NO_CACHE', '')
    auto_use_cache = not (no_cache_env.lower() in ('1', 'true', 'yes'))
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    no_baselines_env = os.getenv('NO_BASELINES', '0')
    evaluate_baselines = not (no_baselines_env.lower() in ('1', 'true', 'yes'))

    print("="*80)
    print("DATA AVAILABILITY EXPERIMENT - REUTERS (reconstructed)")
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

    try:
        few_shot_env = int(os.getenv('FEW_SHOT', '15'))
    except Exception:
        few_shot_env = 15
    globals()['FEW_SHOT_OVERRIDE'] = few_shot_env

    try:
        precache_flag = os.getenv('PRECACHE_LLM', '').lower() in ('1', 'true', 'yes')
    except Exception:
        precache_flag = False

    if precache_flag:
        print('\n🔁 PRECACHE MODE: Generating LLM predictions for full val/test using few-shot context')
        df_train_all, df_val_all, df_test_all = load_datasets(data_dir)
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

        llm_model = create_llm_model('text', label_columns, llm_provider, base_output_dir, 'precache', cache_dir=cache_dir, few_shot_examples=int(few_shot_to_use))
        print('  Generating validation predictions...')
        val_result = llm_model.predict(train_df=train_small, test_df=df_val_all)
        print(f"   -> {len(val_result.predictions)} validation preds generated")

        print('  Generating test predictions...')
        test_result = llm_model.predict(train_df=train_small, test_df=df_test_all)
        print(f"   -> {len(test_result.predictions)} test preds generated")

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
