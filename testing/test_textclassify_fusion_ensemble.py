"""Integration test for the textclassify RoBERTa text expert.

Text-only pipeline over the FOMC hawkish/dovish/neutral task:

  1. Load data/training_data/lab-manual-mm-split-train-5768.xlsx and split it
     80/20 (stratified by label) into train/val.
  2. Load data/test_data/lab-manual-mm-split-test-5768.xlsx as the held-out
     test set.
  3. Train a RoBERTa text expert on the train split, validate on val, and
     evaluate on the held-out test split. No LLM, no market/time-series
     branch -- text only.

The test is intentionally guarded so it does not run during a normal pytest
session unless RUN_TEXTCLASSIFY_FUSION_TEST=1 is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import pytest
except ImportError:  # pragma: no cover - optional for direct script execution
    pytest = None


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from textclassify import RoBERTaClassifier
from textclassify.core.types import ModelConfig, ModelType


TRAIN_PATH = REPO_ROOT / "data" / "training_data" / "lab-manual-mm-split-train-5768.xlsx"
TEST_PATH = REPO_ROOT / "data" / "test_data" / "lab-manual-mm-split-test-5768.xlsx"
OUTPUT_DIR = REPO_ROOT / "testing" / "outputs"

# The benchmark encodes the stance as a single int column ("label": 0/1/2).
# RoBERTaClassifier expects one column per class (num_labels = len(LABEL_COLUMNS)),
# so we expand it into one-hot columns and use those as LABEL_COLUMNS everywhere.
LABEL_MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_COLUMNS = list(LABEL_MAP.values())
TEXT_COLUMN = "sentence"
RANDOM_STATE = 5768
SAMPLE_SIZE = 10

ROBERTA_MODEL = os.getenv("TEXTCLASSIFY_ROBERTA_MODEL", "roberta-base")

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        os.getenv("RUN_TEXTCLASSIFY_FUSION_TEST") != "1",
        reason="Set RUN_TEXTCLASSIFY_FUSION_TEST=1 to run the fusion integration test.",
    )


def _add_onehot_labels(df: pd.DataFrame) -> pd.DataFrame:
    for label_idx, col_name in LABEL_MAP.items():
        df[col_name] = (df["label"] == label_idx).astype(int)
    return df


def load_train_test(sample_size: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the lab-manual-mm-split train and test benchmark files."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train dataset not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)

    for name, df in (("train", train_df), ("test", test_df)):
        missing = [c for c in [TEXT_COLUMN, "label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {missing}")

    train_df = _add_onehot_labels(train_df)
    test_df = _add_onehot_labels(test_df)

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if sample_size > len(train_df):
            raise ValueError(f"sample_size={sample_size} exceeds train dataset size={len(train_df)}")
        train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_articles(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the train file 80/20 into train/validation, stratified by label."""
    stratify = train_df["label"] if len(train_df) >= 20 else None
    train_split, val_split = train_test_split(
        train_df, train_size=0.8, random_state=RANDOM_STATE, stratify=stratify
    )

    for split_name, split_df in (("train", train_split), ("val", val_split)):
        counts = split_df["label"].value_counts().to_dict()
        print(f"{split_name}: {len(split_df)} rows | label counts: {counts}")

    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def build_text_model() -> RoBERTaClassifier:
    """Build the RoBERTa text classifier."""
    ml_config = ModelConfig(
        model_name=ROBERTA_MODEL,
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "model_name": ROBERTA_MODEL,
            "max_length": 256,
            "batch_size": 8,
            "num_epochs": 1,
            "learning_rate": 2e-5,
        },
    )
    return RoBERTaClassifier(
        config=ml_config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_roberta_test",
    )


def run_roberta_only(sample_size: int | None = None):
    """Train the RoBERTa text expert on the train split, validate on val,
    and evaluate on the held-out test split."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_full_df, test_df = load_train_test(sample_size=sample_size)
    train_df, val_df = split_articles(train_full_df)

    ml_model = build_text_model()

    print("\nTraining RoBERTa on train split (validating on val split)...")
    ml_model.fit(train_df, val_df)

    print("\nEvaluating on test split...")
    result = ml_model.predict(test_df)

    metrics = result.metadata.get("metrics", {}) if result.metadata else {}
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    print("\nSample predictions:")
    for text, pred, true in list(zip(test_df[TEXT_COLUMN], result.predictions, test_df["label"].tolist()))[:5]:
        print(f"  pred={pred}  true={true}  text={text[:80]}...")

    return result


def test_textclassify_fusion_ensemble():
    """Pytest entry point for the integration test. Uses a toy SAMPLE_SIZE-row
    sample of the lab-manual-mm-split train file so the pipeline runs quickly
    as a smoke test rather than training over the full dataset."""
    result = run_roberta_only(sample_size=SAMPLE_SIZE)
    assert result is not None
    assert result.predictions


if __name__ == "__main__":
    sample_size_env = os.getenv("TEXTCLASSIFY_SAMPLE_SIZE")
    sample_size = int(sample_size_env) if sample_size_env else None
    run_roberta_only(sample_size=sample_size)
