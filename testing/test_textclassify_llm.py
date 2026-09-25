"""Integration test for the textclassify LLM (few-shot) expert.

Text-only pipeline over the FOMC hawkish/dovish/neutral task, mirroring
test_textclassify_fusion_ensemble.py but using the LLM expert instead of
RoBERTa. No RoBERTa, no market/time-series branch -- LLM only.

Two modes, selected by TEXTCLASSIFY_SAMPLE_SIZE (default 0 = zero-shot):

  * Zero-shot (SAMPLE_SIZE = 0): the train split is not touched at all. The
    LLM sees only the label names and the test sentences and classifies
    data/test_data/lab-manual-mm-split-test-5768.xlsx.
  * Few-shot (SAMPLE_SIZE > 0): take SAMPLE_SIZE rows of
    data/training_data/lab-manual-mm-split-train-5768.xlsx as the few-shot
    example pool, "fit" the LLM expert (this only stores the examples -- no
    training happens) and evaluate on the test split.

The LLM is called through OpenRosuter. Put the key in the repo-root .env file:

    OPENROUTER_API_KEY=sk-or-v1-...

and optionally pick a model (OpenRouter "<vendor>/<model>" id):

    TEXTCLASSIFY_LLM_MODEL=anthropic/claude-sonnet-4.5

The test is intentionally guarded so it does not run during a normal pytest
session unless RUN_TEXTCLASSIFY_FUSION_TEST=1 is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    import pytest
except ImportError:  # pragma: no cover - optional for direct script execution
    pytest = None


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load OPENROUTER_API_KEY (and optional TEXTCLASSIFY_LLM_MODEL) from the repo-root .env
load_dotenv(REPO_ROOT / ".env")

from textclassify import OpenRouterClassifier
from textclassify.core.types import ModelConfig, ModelType


TRAIN_PATH = REPO_ROOT / "data" / "training_data" / "lab-manual-mm-split-train-5768.xlsx"
TEST_PATH = REPO_ROOT / "data" / "test_data" / "lab-manual-mm-split-test-5768.xlsx"
OUTPUT_DIR = REPO_ROOT / "testing" / "outputs"

# The benchmark encodes the stance as a single int column ("label": 0/1/2).
# Keep the same one-hot label columns as the RoBERTa test for consistency.
LABEL_MAP = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_COLUMNS = list(LABEL_MAP.values())
TEXT_COLUMN = "sentence"
RANDOM_STATE = 5768
# Number of few-shot training examples; 0 = zero-shot (no training examples at all)
SAMPLE_SIZE = int(os.getenv("TEXTCLASSIFY_SAMPLE_SIZE", "0"))

LLM_MODEL = os.getenv("TEXTCLASSIFY_LLM_MODEL", "openai/gpt-4o-mini")

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
    """Load the lab-manual-mm-split train and test benchmark files. The
    train file is subsampled to sample_size rows -- that's the few-shot
    example pool, not a training set."""
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


def build_llm_model(sample_size: int = SAMPLE_SIZE) -> OpenRouterClassifier:
    """Build the LLM classifier (via OpenRouter)."""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            f"OPENROUTER_API_KEY is not set. Add OPENROUTER_API_KEY=sk-or-v1-... to {REPO_ROOT / '.env'}"
        )
    llm_config = ModelConfig(
        model_name=LLM_MODEL,
        model_type=ModelType.LLM,
        parameters={
            "model": LLM_MODEL,
            "temperature": 0.1,
            # Reasoning models (gpt-5*) spend this budget on hidden reasoning too
            "max_completion_tokens": int(os.getenv("TEXTCLASSIFY_MAX_TOKENS", "100")),
        },
    )
    # Separate experiment name and cache dir per mode: the auto-cache matches on the
    # test set only, so a shared cache would return few-shot predictions for a
    # zero-shot run (and vice versa).
    # The model is part of the name too, so different models never share a cache.
    model_slug = LLM_MODEL.replace("/", "_")
    run_name = f"{model_slug}_" + ("zero_shot" if sample_size == 0 else f"few_shot_{sample_size}")
    return OpenRouterClassifier(
        config=llm_config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        few_shot_mode="zero_shot" if sample_size == 0 else sample_size,
        output_dir=str(OUTPUT_DIR),
        experiment_name=f"llm_only_{run_name}",
        cache_dir=str(OUTPUT_DIR / "llm_cache" / run_name),
    )


def run_llm_only(sample_size: int = SAMPLE_SIZE):
    """Evaluate the LLM expert on the held-out test split.

    sample_size == 0: zero-shot -- no training examples are stored or passed.
    sample_size > 0:  "train" (store few-shot examples for) the LLM expert on
                      sample_size examples from the train split first."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zero_shot = sample_size == 0

    train_df, test_df = load_train_test(sample_size=None if zero_shot else sample_size)
    print(f"test: {len(test_df)} rows | label counts: {test_df['label'].value_counts().to_dict()}")

    llm_model = build_llm_model(sample_size)

    if zero_shot:
        # No fit() and train_df=None: nothing from the train split reaches the LLM,
        # neither as few-shot examples nor in the generated role/context prompts.
        print(f"\nZero-shot mode ({LLM_MODEL}): no training examples.")
        train_df = None
    else:
        print(f"few-shot pool: {len(train_df)} rows | label counts: {train_df['label'].value_counts().to_dict()}")
        print(f"\nStoring {len(train_df)} examples as the few-shot pool...")
        llm_model.fit(train_df)

    print("\nEvaluating on test split...")
    result = llm_model.predict(train_df=train_df, test_df=test_df)

    metrics = result.metadata.get("metrics", {}) if result.metadata else {}
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    print("\nSample predictions:")
    for text, pred, true in list(zip(test_df[TEXT_COLUMN], result.predictions, test_df["label"].tolist()))[:5]:
        print(f"  pred={pred}  true={true}  text={text[:80]}...")

    return result


def test_textclassify_llm():
    """Pytest entry point for the integration test. Uses SAMPLE_SIZE
    (default 0 = zero-shot; set TEXTCLASSIFY_SAMPLE_SIZE for few-shot)."""
    result = run_llm_only(sample_size=SAMPLE_SIZE)
    assert result is not None
    assert result.predictions


if __name__ == "__main__":
    run_llm_only(sample_size=SAMPLE_SIZE)
