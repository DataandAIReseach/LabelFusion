"""
Sample a small batch of articles, run an LLM classifier (OpenAI gpt-4o-mini)
on all of them, then evaluate the predictions on a 20% held-out slice of that
sample.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from textclassify import OpenAIClassifier
from textclassify.core.types import ModelConfig, ModelType

ARTICLES_DIR = REPO_ROOT / "data" / "articles"
CSV_PATH = ARTICLES_DIR / "articles_all_commodities_baseline_seed7_20260722_190748.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
TEXT_COLUMN = "text"
RANDOM_STATE = 7

SAMPLE_SIZE = 200  # single batch (matches default batch_size) -- quick check that the batching fix works
EVAL_FRAC = 0.2  # fraction of the sample held out for evaluation

LLM_MODEL = "gpt-4o-mini"


def load_sample_data():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df[TEXT_COLUMN] = df["headline"].fillna("") + " " + df["body"].fillna("")
    print(f"Full dataset shape: {df.shape}")

    sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} rows")

    eval_df, _rest = train_test_split(
        sample_df, train_size=EVAL_FRAC, random_state=RANDOM_STATE
    )
    print(f"Eval slice: {len(eval_df)} rows ({EVAL_FRAC:.0%} of the sample)")

    return sample_df, eval_df.index


def build_llm_model() -> OpenAIClassifier:
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
        label_columns=ALL_COMMODITIES,
        multi_label=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name="train_test_llm",
        cache_dir=str(OUTPUT_DIR / "llm_cache"),
    )


def main():
    sample_df, eval_index = load_sample_data()

    llm = build_llm_model()

    print(f"\n=== Predicting on all {len(sample_df)} sampled articles ===")
    result = llm.predict(test_df=sample_df)

    print(f"\n=== Evaluating on the {EVAL_FRAC:.0%} held-out slice ({len(eval_index)} rows) ===")
    eval_positions = [sample_df.index.get_loc(i) for i in eval_index]

    y_true = sample_df.loc[eval_index, ALL_COMMODITIES].values
    y_pred = [[1 if c in result.predictions[pos] else 0 for c in ALL_COMMODITIES] for pos in eval_positions]
    y_pred = pd.DataFrame(y_pred, columns=ALL_COMMODITIES).values

    for i, c in enumerate(ALL_COMMODITIES):
        print(f"=== {c.upper()} ===")
        print(classification_report(y_true[:, i], y_pred[:, i]))

    exact_match = (y_true == y_pred).all(axis=1).mean()
    mean_accuracy = (y_true == y_pred).mean()
    print(f"Exact match (all {len(ALL_COMMODITIES)} correct): {exact_match:.3f}")
    print(f"Mean per-label accuracy: {mean_accuracy:.3f}")


if __name__ == "__main__":
    main()
