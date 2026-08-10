"""Integration test for the textclassify fusion ensemble.

Continuous end-to-end pipeline over the FOMC hawkish/dovish/neutral task:

  1. Normalize the 6 FRED macro series (data/TSs/fred_*.csv) into a uniform
     Date/value schema so they can be read by the shared TS loader.
  2. Load data/gold_dated.csv -- the dated, all-doctype (mm/sp/pc) version of
     the lab-manual benchmark -- since the market branch needs a real
     per-sentence date to look up price windows (the lab-manual-*.xlsx files
     only carry a bare year).
  3. Time-split it: the most recent 20% of sentences (by date) is held out
     as the test set; the earlier 80% is split again 80/20 (stratified by
     label) into train/val.
  4. Train a RoBERTa text expert, an LLM expert, and a market expert (reads
     the 6 macro series around each sentence's date) and fuse all three via
     FusionEnsemble, mirroring the paper's u = [h; z; m] fusion.

The test is intentionally guarded so it does not run during a normal pytest
session unless RUN_TEXTCLASSIFY_FUSION_TEST=1 is set (the market branch
downloads a pretrained TimesFM checkpoint and needs a GPU to be practical).
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


DATA_PATH = REPO_ROOT / "for_michael" / "data" / "gold_dated.csv"
FRED_DIR = REPO_ROOT / "data" / "TSs"
MACRO_TS_DIR = REPO_ROOT / "data" / "TSs_macro"
OUTPUT_DIR = REPO_ROOT / "testing" / "outputs"

MACRO_SERIES = ["CPIAUCSL", "DFF", "DGS2", "DGS10", "NASDAQCOM", "UNRATE"]

# gold_dated.csv encodes the stance as a single int column ("label": 0/1/2).
# RoBERTaClassifier expects one column per class (num_labels = len(LABEL_COLUMNS)),
# so we expand it into one-hot columns and use those as LABEL_COLUMNS everywhere.
LABEL_MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_COLUMNS = list(LABEL_MAP.values())
TEXT_COLUMN = "sentence"
DATE_COLUMN = "date"
TS_WINDOW_DAYS = 126  # matches the paper's market window
RANDOM_STATE = 5768
SAMPLE_SIZE = 10
TEST_FRACTION = 0.2

LLM_MODEL = os.getenv("TEXTCLASSIFY_LLM_MODEL", "gpt-5o-mini")
ROBERTA_MODEL = os.getenv("TEXTCLASSIFY_ROBERTA_MODEL", "roberta-base")

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        os.getenv("RUN_TEXTCLASSIFY_FUSION_TEST") != "1",
        reason="Set RUN_TEXTCLASSIFY_FUSION_TEST=1 to run the fusion integration test.",
    )


def normalize_macro_ts() -> None:
    """Rewrite each fred_*.csv into MACRO_TS_DIR/<SERIES>.csv with columns
    Date,value so TS() can load all 6 series uniformly (their raw FRED
    columns differ: observation_date/CPIAUCSL, observation_date/DFF, ...).
    Idempotent: skipped if the normalized files already exist."""
    MACRO_TS_DIR.mkdir(parents=True, exist_ok=True)
    for series in MACRO_SERIES:
        out_path = MACRO_TS_DIR / f"{series}.csv"
        if out_path.exists():
            continue
        df = pd.read_csv(FRED_DIR / f"fred_{series}.csv")
        df = df.rename(columns={"observation_date": "Date", series: "value"})
        df[["Date", "value"]].to_csv(out_path, index=False)


def load_articles(sample_size: int | None = None) -> pd.DataFrame:
    """Load and normalize gold_dated.csv for the fusion pipeline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in [TEXT_COLUMN, DATE_COLUMN, "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    for label_idx, col_name in LABEL_MAP.items():
        df[col_name] = (df["label"] == label_idx).astype(int)

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if sample_size > len(df):
            raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(df)}")
        # Sample from the front 80% only, so the held-out chronological test
        # tail below stays untouched by the smoke-test's sub-sampling.
        cutoff = int(len(df) * (1 - TEST_FRACTION))
        df = pd.concat([
            df.iloc[:cutoff].sample(n=sample_size, random_state=RANDOM_STATE),
            df.iloc[cutoff:],
        ]).sort_values(DATE_COLUMN).reset_index(drop=True)

    return df


def split_articles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-split into train/val/test: the most recent TEST_FRACTION of
    sentences (by date) is the test set; the earlier portion is split
    80/20 (stratified by label) into train/val."""
    cutoff = int(len(df) * (1 - TEST_FRACTION))
    train_full_df, test_df = df.iloc[:cutoff], df.iloc[cutoff:]

    stratify = train_full_df["label"] if len(train_full_df) >= 20 else None
    train_df, val_df = train_test_split(
        train_full_df, train_size=0.8, random_state=RANDOM_STATE, stratify=stratify
    )

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = split_df["label"].value_counts().to_dict()
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
        multi_label=False,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_roberta_test",
    )


def run_roberta_only(sample_size: int | None = None):
    """Train the RoBERTa text expert alone (no LLM, no market branch) on the
    train split, validate on val, and evaluate on the held-out test split."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_articles(sample_size=sample_size)
    train_df, val_df, test_df = split_articles(df)

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
        multi_label=False,
        output_dir=str(OUTPUT_DIR),
        experiment_name="fusion_openai_test",
        cache_dir=str(OUTPUT_DIR / "llm_cache"),
    )


def build_ts_branch():
    """Build the TS loader, per-series embedders, and cross-series transformer
    for the 6 normalized macro series."""
    ts_loader = TS(
        data_dir=str(MACRO_TS_DIR),
        stock_symbols=MACRO_SERIES,
        date_column="Date",
        price_column="value",
    )
    ts_loader.load_all()

    ts_embedders = {name: TSEmbedder(pooling="mean") for name in MACRO_SERIES}
    hidden_size = next(iter(ts_embedders.values())).hidden_size
    ts_transformer = CrossTSTransformer(
        series_names=MACRO_SERIES,
        hidden_size=hidden_size,
        output_dim=256,
    )

    return ts_loader, ts_embedders, ts_transformer


def build_fusion_ensemble(ml_model, llm_model, ts_branch=None) -> FusionEnsemble:
    """Create the fusion ensemble with the text, LLM, and (optional) market branches."""
    ensemble_config = EnsembleConfig(
        ensemble_method="fusion",
        models=[ml_model, llm_model],
        parameters={
            "fusion_hidden_dims": [64, 32],
            "ml_lr": 1e-5,
            "fusion_lr": 1e-3,
            "num_epochs": 3,
            "batch_size": 16,
            "classification_type": "multi_class",
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
    """Run the full fusion pipeline (text + LLM + market experts) and return
    the prediction result."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normalize_macro_ts()

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
    for text, pred, true in list(zip(test_df[TEXT_COLUMN], result.predictions, test_df["label"].tolist()))[:5]:
        print(f"  pred={pred}  true={true}  text={text[:80]}...")

    return result


def test_textclassify_fusion_ensemble():
    """Pytest entry point for the integration test. Uses a toy SAMPLE_SIZE-row
    sample of the training portion of gold_dated.csv so the fusion pipeline
    runs quickly as a smoke test rather than training over the full dataset."""
    result = run_fusion_integration(sample_size=SAMPLE_SIZE)
    assert result is not None
    assert result.predictions


if __name__ == "__main__":
    sample_size_env = os.getenv("TEXTCLASSIFY_SAMPLE_SIZE")
    sample_size = int(sample_size_env) if sample_size_env else None

    if os.getenv("TEXTCLASSIFY_ROBERTA_ONLY") == "1":
        run_roberta_only(sample_size=sample_size)
    else:
        run_fusion_integration(sample_size=sample_size)