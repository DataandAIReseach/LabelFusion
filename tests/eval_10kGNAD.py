"""10kGNAD OpenAIClassifier runner.

Loads train.csv + test.csv from 10kGNAD, predicts labels for test texts,
and writes predictions to OUTPUT_DIR.

Environment variables:
  - DATA_DIR: 10kGNAD folder (default: Dataset_Descriptives/data/10kGNAD)
  - OUTPUT_DIR: output folder (default: outputs/10kgnad_openai)
  - FEW_SHOT: number of few-shot examples (default: 20)
  - MODEL_NAME: OpenAI model name (default: gpt-5-nano)
  - CACHE_DIR: cache folder for LLM calls (default: cache/10kgnad_openai)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# ensure project root on path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, ModelType


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


def create_llm_model(
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
        label_columns=["label"],   # single-label task
        multi_label=False,
        few_shot_mode=few_shot_examples,
        auto_use_cache=True,
        cache_dir=cache_dir,
        auto_save_results=True,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )


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

    df_train, df_test = load_datasets(data_dir)
    experiment_name = f"10kgnad_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    llm_model = create_llm_model(
        output_dir=output_dir,
        experiment_name=experiment_name,
        cache_dir=cache_dir,
        few_shot_examples=few_shot,
        model_name=model_name,
    )

    print(f"Train size: {len(df_train)} | Test size: {len(df_test)}")
    print("Predicting test labels with OpenAIClassifier...")

    result = llm_model.predict(df_test, train_df=df_train)
    pred_series = _extract_pred_series(result)

    if len(pred_series) != len(df_test):
        raise RuntimeError(f"Prediction length mismatch: got {len(pred_series)}, expected {len(df_test)}")

    out_df = df_test.copy()
    out_df["pred_label"] = pred_series.values

    out_file = os.path.join(output_dir, f"{experiment_name}_predictions.csv")
    out_df.to_csv(out_file, index=False)

    print(f"Saved predictions: {out_file}")

    if "label" in out_df.columns and out_df["label"].notna().any():
        acc = (out_df["label"].astype(str) == out_df["pred_label"].astype(str)).mean()
        print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    data_dir = os.getenv(
        "DATA_DIR",
        "/home/michaelschlee/ownCloud/GIT/LabelFusion/Dataset_Descriptives/data/10kGNAD",
    )
    output_dir = os.getenv("OUTPUT_DIR", "outputs/10kgnad_openai")
    cache_dir = os.getenv("CACHE_DIR", "cache/10kgnad_openai")
    model_name = os.getenv("MODEL_NAME", "gpt-5-nano")

    try:
        few_shot = int(os.getenv("FEW_SHOT", "20"))
    except Exception:
        few_shot = 20

    run_once(
        data_dir=data_dir,
        output_dir=output_dir,
        few_shot=few_shot,
        model_name=model_name,
        cache_dir=cache_dir,
    )