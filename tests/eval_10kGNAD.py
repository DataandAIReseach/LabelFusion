"""10kGNAD Fusion Runner.

Loads train.csv + test.csv from 10kGNAD, trains a FusionEnsemble 
with RoBERTa transformer and OpenAI LLM models, and evaluates predictions.

Environment variables:
  - DATA_DIR: 10kGNAD folder (default: Dataset_Descriptives/data/10kGNAD)
  - OUTPUT_DIR: output folder (default: outputs/10kgnad_fusion)
  - FEW_SHOT: number of few-shot examples for LLM (default: 5)
  - MODEL_NAME: OpenAI model name (default: gpt-5-mini)
  - CACHE_DIR: cache folder for LLM calls (default: cache/10kgnad_fusion)
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd

# ensure project root on path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import ModelConfig, EnsembleConfig, ModelType


def load_datasets(data_dir: str):
    columns = ["label", "text", "publishingDate"]
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(
        train_path,
        sep=";",
        quotechar="'",
        header=None,
        names=columns,
    ).dropna(subset=["label", "text"]).reset_index(drop=True)

    df_test = pd.read_csv(
        test_path,
        sep=";",
        quotechar="'",
        header=None,
        names=columns,
    ).dropna(subset=["text"]).reset_index(drop=True)

    return df_train, df_test


def one_hot_encode(df_train: pd.DataFrame, df_test: pd.DataFrame, label_col: str):
    """
    One-hot encode a single string label column into binary columns.

    e.g. label="Web" → Web=1, Sport=0, Inland=0, ...

    Classes are determined from training data only.

    Returns:
        df_train_encoded, df_test_encoded, label_columns (list of class names)
    """
    classes = sorted(df_train[label_col].dropna().unique().tolist())

    df_train = df_train.copy()
    df_test  = df_test.copy()

    for cls in classes:
        df_train[cls] = (df_train[label_col] == cls).astype(int)
        df_test[cls]  = (df_test[label_col]  == cls).astype(int) if label_col in df_test.columns else 0

    return df_train, df_test, classes


def create_llm_model(
    label_columns: list,
    output_dir: str,
    experiment_name: str,
    cache_dir: str,
    few_shot_examples: int,
    model_name: str,
):
    llm_config = ModelConfig(
        model_name=model_name,
        model_type=ModelType.LLM,
        parameters={
            "model": model_name,
            "temperature": 0.1,
            "max_completion_tokens": 120,
            "top_p": 1.0,
        },
    )
    return OpenAIClassifier(
        config=llm_config,
        text_column="text",
        label_columns=label_columns,
        multi_label=False,
        few_shot_mode=few_shot_examples,
        auto_use_cache=True,
        cache_dir=cache_dir,
        auto_save_results=True,
        output_dir=output_dir,
        experiment_name=experiment_name,
        use_nearest_neighbours=True,  # For simplicity, we won't use nearest neighbors in this test
    )


def create_ml_model(
    label_columns: list,
    output_dir: str,
    experiment_name: str,
    auto_save_path: str = None,
):
    ml_config = ModelConfig(
        model_name='benjamin/roberta-base-wechsel-german',
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            'model_name': 'benjamin/roberta-base-wechsel-german',
            'max_length': 256,
            'learning_rate': 2e-5,
            'num_epochs': 2,
            'batch_size': 32
        }
    )
    return RoBERTaClassifier(
        config=ml_config,
        text_column='text',
        label_columns=label_columns,
        multi_label=False,
        auto_save_path=auto_save_path,
        auto_save_results=True,
        output_dir=output_dir,
        experiment_name=f"{experiment_name}_german_roberta"
    )


def create_fusion_ensemble(
    ml_model,
    llm_model,
    output_dir: str,
    experiment_name: str,
    cache_dir: str = 'cache',
):
    fusion_config = EnsembleConfig(
        ensemble_method='fusion',
        models=[ml_model, llm_model],
        parameters={
            'fusion_hidden_dims': [64, 32],
            'ml_lr': 1e-5,
            'fusion_lr': 5e-4,
            'num_epochs': 10,
            'batch_size': 8,
            'classification_type': 'multi_class',
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
    return fusion


def _extract_pred_series(result):
    # Works with common result container styles.
    if hasattr(result, "predictions"):
        preds = result.predictions
        if isinstance(preds, pd.Series):
            return preds.reset_index(drop=True)
        if isinstance(preds, pd.DataFrame):
            if "label" in preds.columns:
                return preds["label"].reset_index(drop=True)
            if "pred_label" in preds.columns:
                return preds["pred_label"].reset_index(drop=True)
            if len(preds.columns) == 1:
                return preds.iloc[:, 0].reset_index(drop=True)
        if isinstance(preds, (list, tuple)):
            return pd.Series(list(preds))
    if isinstance(result, pd.DataFrame):
        if "label" in result.columns:
            return result["label"].reset_index(drop=True)
        if "pred_label" in result.columns:
            return result["pred_label"].reset_index(drop=True)
        if len(result.columns) == 1:
            return result.iloc[:, 0].reset_index(drop=True)
    if isinstance(result, (list, tuple, pd.Series)):
        return pd.Series(list(result))
    raise RuntimeError("Could not extract predictions from OpenAIClassifier result object.")


def run_once(data_dir: str, output_dir: str, few_shot: int, model_name: str, cache_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    raw_train_df, raw_test_df = load_datasets(data_dir)

    # One-hot encode string labels into binary columns per class
    # e.g. "Web" → Web=1, Sport=0, Inland=0, ...
    # Classes are determined from training data only
    df_train_enc, df_test_enc, label_columns = one_hot_encode(
        raw_train_df, raw_test_df, label_col="label"
    )
    print(f"Classes detected: {label_columns}")

    val_size = max(1, int(len(df_train_enc) * 0.1))
    val_df = df_train_enc.sample(n=val_size, random_state=42)
    train_df = df_train_enc.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = df_test_enc

    experiment_name = f"10kgnad_fusion_german_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Try to discover a cached RoBERTa model in the `cache/` folder.
    # Looks for files starting with `10kgnad_fusion_german_roberta_model_...` and loads the newest one if available.
    cache_base_pattern = os.path.join('cache', '10kgnad_fusion_german_roberta_model_*')
    cached_files = sorted(glob.glob(cache_base_pattern), key=os.path.getctime, reverse=True)
    ml_model = None
    if cached_files:
        latest_cache = cached_files[0]
        print(f"Found candidate cached ML model: {latest_cache} -> attempting to load")
        # Create model instance (no auto_save_path required to load)
        ml_model = create_ml_model(label_columns, output_dir, experiment_name, auto_save_path=None)
        try:
            ml_model.load_model(latest_cache)
            print(f"✅ Loaded RoBERTa model from cache: {latest_cache}")
        except Exception as e:
            print(f"⚠️ Failed to load cached RoBERTa model ({latest_cache}): {e}")
            print("Will train a new RoBERTa model instead.")
            ml_model = create_ml_model(label_columns, output_dir, experiment_name, auto_save_path=None)
    else:
        ml_model = create_ml_model(label_columns, output_dir, experiment_name, auto_save_path=None)

    llm_model = create_llm_model(
        label_columns=label_columns,
        output_dir=output_dir,
        experiment_name=experiment_name,
        cache_dir=cache_dir,
        few_shot_examples=few_shot,
        model_name=model_name,
    )

    fusion = create_fusion_ensemble(ml_model, llm_model, output_dir, experiment_name, cache_dir)

    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    print("Training FusionEnsemble with RoBERTa transformer and OpenAI LLM...")

    # For 10kGNAD, we use the encoded training data for fusion training
    fusion.fit(train_df, val_df)

    print("Predicting test labels with FusionEnsemble...")
    result = fusion.predict(test_df, train_df=train_df)
    pred_series = _extract_pred_series(result)

    if len(pred_series) != len(raw_test_df):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(pred_series)}, expected {len(raw_test_df)}"
        )

    # pred_series contains predicted class names (e.g. "Web", "Sport")
    # compare against original string labels for accuracy
    out_df = raw_test_df.copy()
    out_df["pred_label"] = pred_series.values

    out_file = os.path.join(output_dir, f"{experiment_name}_fusion_predictions.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved fusion predictions: {out_file}")

    if "label" in out_df.columns and out_df["label"].notna().any():
        acc = (out_df["label"].astype(str) == out_df["pred_label"].astype(str)).mean()
        print(f"Test accuracy (fusion): {acc:.4f}")


if __name__ == "__main__":
    data_dir = os.getenv(
        "DATA_DIR",
        
        "/scratch1/users/u19147/LabelFusion/Dataset_Descriptives/data/10kGNAD",
    )
    output_dir = os.getenv("OUTPUT_DIR", "outputs/10kgnad_fusion")
    cache_dir  = os.getenv("CACHE_DIR", "cache/10kgnad_fusion")
    model_name = os.getenv("MODEL_NAME", "gpt-5-mini")

    try:
        few_shot = int(os.getenv("FEW_SHOT", "5"))
    except Exception:
        few_shot = 20

    run_once(
        data_dir=data_dir,
        output_dir=output_dir,
        few_shot=few_shot,
        model_name=model_name,
        cache_dir=cache_dir,
    )