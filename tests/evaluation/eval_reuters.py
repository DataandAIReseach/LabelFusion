"""Trimmed Reuters fusion runner.

Performs a single FusionEnsemble training + evaluation run.
Environment variables:
  - PERCENTAGE: fraction of training data to use (default 0.001)
  - FEW_SHOT: few-shot examples for LLM (default 10)
  - DATA_DIR: path to Reuters data (default /scratch/users/u19147/LabelFusion/data/reuters)
  - OUTPUT_DIR: path to write outputs (default outputs/reuters_fusion)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# ensure project root on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType


def load_datasets(data_dir: str = "/scratch/users/u19147/LabelFusion/data/reuters"):
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(data_dir, "val.csv")).sample(frac=0.1, random_state=42).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv")).sample(frac=0.1, random_state=42).reset_index(drop=True)
    return df_train, df_val, df_test


def detect_label_columns(df: pd.DataFrame, text_column: str = 'text'):
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


def create_ml_model(text_column: str, label_columns: list, output_dir: str, experiment_name: str, auto_save_path: str = None):
    ml_config = ModelConfig(model_name='roberta-base', model_type=ModelType.TRADITIONAL_ML, parameters={'model_name': 'roberta-base', 'max_length': 256, 'learning_rate': 2e-5, 'num_epochs': 2, 'batch_size': 8})
    return RoBERTaClassifier(config=ml_config, text_column=text_column, label_columns=label_columns, multi_label=True, auto_save_path=auto_save_path, auto_save_results=True, output_dir=output_dir, experiment_name=f"{experiment_name}_roberta")


def create_llm_model(text_column: str, label_columns: list, output_dir: str, experiment_name: str, cache_dir: str = 'cache', few_shot_examples: int = 2):
    llm_config = ModelConfig(model_name='gpt-5-nano', model_type=ModelType.LLM, parameters={'model': 'gpt-5-nano', 'temperature': 0.1, 'max_completion_tokens': 150, 'top_p': 1.0})
    return OpenAIClassifier(config=llm_config, text_column=text_column, label_columns=label_columns, multi_label=True, few_shot_mode=few_shot_examples, auto_use_cache=True, cache_dir=cache_dir, auto_save_results=True, output_dir=output_dir, experiment_name=f"{experiment_name}_openai")


def create_fusion_ensemble(ml_model, llm_model, output_dir: str, experiment_name: str, cache_dir: str = 'cache'):
    fusion_config = EnsembleConfig(ensemble_method='fusion', models=[ml_model, llm_model], parameters={'fusion_hidden_dims': [64, 32], 'ml_lr': 1e-5, 'fusion_lr': 5e-4, 'num_epochs': 10, 'batch_size': 8, 'classification_type': 'multi_label', 'output_dir': output_dir, 'experiment_name': experiment_name, 'val_llm_cache_path': os.path.join(cache_dir, 'val'), 'test_llm_cache_path': os.path.join(cache_dir, 'test')})
    fusion = FusionEnsemble(fusion_config, output_dir=output_dir, experiment_name=experiment_name, auto_save_results=True, save_intermediate_llm_predictions=False)
    fusion.add_ml_model(ml_model)
    fusion.add_llm_model(llm_model)
    return fusion


def run_once(data_dir: str, output_dir: str, percentage: float = 0.001, few_shot: int = 10, ml_train_on_full: bool = False):
    df_train, df_val, df_test = load_datasets(data_dir)
    text_column = 'text'
    label_columns = detect_label_columns(df_train, text_column)

    if ml_train_on_full:
        fusion_train_df = df_train
    else:
        n = max(1, int(len(df_train) * percentage))
        fusion_train_df = df_train.sample(n=n, random_state=42).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    experiment_name = f"reuters_fusion_{int(percentage*100000)}ppm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ml_model = create_ml_model(text_column, label_columns, output_dir, experiment_name, auto_save_path=None)
    llm_model = create_llm_model(text_column, label_columns, output_dir, experiment_name, few_shot_examples=few_shot)
    fusion = create_fusion_ensemble(ml_model, llm_model, output_dir, experiment_name)

    print(f"Training fusion on {len(fusion_train_df)} samples (labels: {label_columns})")
    fusion.fit(fusion_train_df, df_val)

    print("Predicting on test set...")
    result = fusion.predict(df_test)
    metrics = result.metadata.get('metrics', {}) if result.metadata else {}
    print(f"Test metrics: {metrics}")


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    output_dir = os.getenv('OUTPUT_DIR', 'outputs/reuters_fusion')
    try:
        percentage = float(os.getenv('PERCENTAGE', '0.001'))
    except Exception:
        percentage = 0.001
    try:
        few_shot = int(os.getenv('FEW_SHOT', '10'))
    except Exception:
        few_shot = 10
    ml_train_on_full = os.getenv('ML_TRAIN_ON_FULL', '').lower() in ('1', 'true', 'yes')

    run_once(data_dir=data_dir, output_dir=output_dir, percentage=percentage, few_shot=few_shot, ml_train_on_full=ml_train_on_full)
