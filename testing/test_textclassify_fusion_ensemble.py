"""Integration test for the textclassify fusion ensemble.

This test uses the article CSV as the source dataset, splits it into
train/validation/test, loads the corresponding commodity time-series files,
and exercises the FusionEnsemble end to end.

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

from textclassify import FusionEnsemble, OpenAIClassifier, RoBERTaClassifier
from textclassify.core.types import EnsembleConfig, ModelConfig, ModelType
from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder


DATA_PATH = REPO_ROOT / "data" / "articles" / "csvs" / "articles_all_commodities_baseline_seed7_20260722_190748.csv"
TS_DATA_DIR = REPO_ROOT / "data" / "TSs"
OUTPUT_DIR = REPO_ROOT / "testing" / "outputs"

LABEL_COLUMNS = ["gold", "silver", "oil", "gas"]
TEXT_COLUMN = "text"
DATE_COLUMN = "article_date"
TS_SERIES_NAMES = LABEL_COLUMNS
TS_WINDOW_DAYS = 21
RANDOM_STATE = 7
SAMPLE_SIZE = 10

LLM_MODEL = os.getenv("TEXTCLASSIFY_LLM_MODEL", "gpt-4o-mini")
ROBERTA_MODEL = os.getenv("TEXTCLASSIFY_ROBERTA_MODEL", "roberta-base")

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        os.getenv("RUN_TEXTCLASSIFY_FUSION_TEST") != "1",
        reason="Set RUN_TEXTCLASSIFY_FUSION_TEST=1 to run the fusion integration test.",
    )


def load_articles(sample_size: int | None = None) -> pd.DataFrame:
    """Load and normalize the article dataset for the fusion pipeline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Article dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_labels = [label for label in LABEL_COLUMNS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Article dataset is missing label columns: {missing_labels}")

    if "headline" in df.columns and "body" in df.columns:
        df[TEXT_COLUMN] = df["headline"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    elif TEXT_COLUMN not in df.columns:
        raise ValueError("Article dataset must contain either headline/body columns or a text column")

    if DATE_COLUMN not in df.columns:
        if "date" in df.columns:
            df[DATE_COLUMN] = df["date"]
        else:
            raise ValueError(f"Article dataset must contain {DATE_COLUMN} or date")

    df = df[[TEXT_COLUMN, DATE_COLUMN] + LABEL_COLUMNS].copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if sample_size > len(df):
            raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(df)}")
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def split_articles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/validation/test sets."""
    train_df, temp_df = train_test_split(df, train_size=0.8, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=RANDOM_STATE)

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = split_df[LABEL_COLUMNS].sum().to_dict()
        print(f"{split_name}: {len(split_df)} rows | label counts: {counts}")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_text_model() -> RoBERTaClassifier:
    """Build the RoBERTa branch used by the fusion ensemble."""
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
        multi_label=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_roberta_test",
    )


def build_llm_model() -> OpenAIClassifier:
    """Build the LLM branch used by the fusion ensemble."""
    llm_config = ModelConfig(
        model_name=LLM_MODEL,
        model_type=ModelType.LLM,
        parameters={
            "model": LLM_MODEL,
            "temperature": 0.1,
            "max_completion_tokens": 100,
        },
    )
    return OpenAIClassifier(
        config=llm_config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_openai_test",
        cache_dir=str(OUTPUT_DIR / "llm_cache"),
    )


def build_ts_branch():
    """Build the TS loader, per-series embedders, and cross-series transformer."""
    ts_loader = TS(
        data_dir=str(TS_DATA_DIR),
        stock_symbols=TS_SERIES_NAMES,
        date_column="Date",
        price_column="Stock_Price",
    )
    ts_loader.load_all()

    ts_embedders = {name: TSEmbedder(pooling="mean") for name in TS_SERIES_NAMES}
    hidden_size = next(iter(ts_embedders.values())).hidden_size
    ts_transformer = CrossTSTransformer(
        series_names=TS_SERIES_NAMES,
        hidden_size=hidden_size,
        output_dim=256,
    )

    return ts_loader, ts_embedders, ts_transformer


def build_fusion_ensemble(ml_model, llm_model, ts_branch=None) -> FusionEnsemble:
    """Create the fusion ensemble with the text and time-series branches."""
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            "fusion_hidden_dims": [64, 32],
            "ml_lr": 1e-5,
            "fusion_lr": 1e-3,
            "num_epochs": 3,
            "batch_size": 16,
            "classification_type": "multi_label",
            "output_dir": str(OUTPUT_DIR),
            "experiment_name": "fusion_ensemble_test",
            "auto_save_results": True,
        },
    )

    fusion_ensemble = FusionEnsemble(
        ensemble_config,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_ensemble_test",
    )
    fusion_ensemble.add_ml_model(ml_model)
    fusion_ensemble.add_llm_model(llm_model)

    if ts_branch is not None:
        ts_loader, ts_embedders, ts_transformer = ts_branch
        fusion_ensemble.add_ts_model(
            ts_loader,
            ts_embedders,
            ts_transformer,
            date_column=DATE_COLUMN,
            window_days=TS_WINDOW_DAYS,
        )

    return fusion_ensemble


def run_fusion_integration(sample_size: int | None = None):
    """Run the full fusion pipeline and return the prediction result."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_articles(sample_size=sample_size)
    train_df, val_df, test_df = split_articles(df)

    ml_model = build_text_model()
    llm_model = build_llm_model()
    ts_branch = build_ts_branch()
    fusion_ensemble = build_fusion_ensemble(ml_model, llm_model, ts_branch=ts_branch)

    print("\nTraining fusion ensemble...")
    fusion_ensemble.fit(train_df, val_df)

    print("\nEvaluating on test split...")
    result = fusion_ensemble.predict(test_df, train_df=train_df)

    metrics = result.metadata.get("metrics", {}) if result.metadata else {}
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    print("\nSample predictions:")
    for text, pred, true in list(zip(test_df[TEXT_COLUMN], result.predictions, test_df[LABEL_COLUMNS].values.tolist()))[:5]:
        true_labels = [LABEL_COLUMNS[i] for i, value in enumerate(true) if value == 1]
        print(f"  pred={pred}  true={true_labels}  text={text[:80]}...")

    return result


def test_textclassify_fusion_ensemble():
    """Pytest entry point for the integration test. Uses a toy SAMPLE_SIZE-row
    sample of the article dataset so the fusion pipeline runs quickly
    as a smoke test rather than training over the full dataset."""
    result = run_fusion_integration(sample_size=SAMPLE_SIZE)
    assert result is not None
    assert result.predictions


if __name__ == "__main__":
    sample_size_env = os.getenv("TEXTCLASSIFY_SAMPLE_SIZE")
    sample_size = int(sample_size_env) if sample_size_env else None
    run_fusion_integration(sample_size=sample_size)