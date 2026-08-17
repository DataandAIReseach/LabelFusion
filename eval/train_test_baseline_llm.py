"""
Sample 5000 articles, split 80/20 into train_df (4000 rows) / rest_df (1000
rows), and predict on rest_df with the LLM (OpenAI gpt-5-nano). The few-shot
pool used for prompting is a separate, tiny 5-row sample drawn from train_df
(disjoint from rest_df, so no row is both a few-shot example and something
being predicted/evaluated).

fit(few_shot_pool) just stores it (no LLM calls) -- predict() then uses it
automatically as the few-shot example source. FIXED_EXAMPLES controls
whether the same examples get reused for every prompt (True) or resampled
per test row (False).

The 80/20 split is reproducible: RANDOM_STATE is used for both the initial
sample() and the train_test_split() below, and the CSV always loads in the
same row order.

Note: this predicts on all 1000 rows of rest_df -- ~1000 real API calls,
a real cost/time commitment, not a cheap smoke test.
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
RANDOM_STATE = 7  # fixed seed -> reproducible sampling across runs

SAMPLE_SIZE = 5000  # total rows sampled from the full dataset
TRAIN_FRAC = 0.8  # fraction of the sample that becomes train_df -- this is what
                   # actually gets predicted on and evaluated
FEW_SHOT_POOL_SIZE = 5  # few-shot examples, drawn from the leftover 20% (not train_df)

FEW_SHOT_MODE = "few_shot"  # named mode (5 examples) -- or pass an int for an exact count
FIXED_EXAMPLES = False  # True: reuse the same few-shot examples every prompt
                         # False: resample fresh examples per test row

LLM_MODEL = "gpt-5-nano"


def load_sample_data():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df[TEXT_COLUMN] = df["headline"].fillna("") + " " + df["body"].fillna("")
    print(f"Full dataset shape: {df.shape}")

    sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    train_df, rest_df = train_test_split(
        sample_df, train_size=TRAIN_FRAC, random_state=RANDOM_STATE
    )

    train_df = train_df.reset_index(drop=True)
    rest_df = rest_df.reset_index(drop=True)

    print(f"Sampled {len(sample_df)} rows total")
    print(f"train_df (80%): {len(train_df)} rows")
    print(f"rest_df (20%): {len(rest_df)} rows")

    return train_df, rest_df


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
        few_shot_mode=FEW_SHOT_MODE,
        fixed_examples=FIXED_EXAMPLES,
        output_dir=str(OUTPUT_DIR),
        experiment_name="train_test_llm",
        cache_dir=str(OUTPUT_DIR / "llm_cache"),
    )


def main():
    train_df, rest_df = load_sample_data()

    llm = build_llm_model()

    # --- Original approach: predict the 80% (train_df), using a 5-row few-shot
    # pool drawn from the 20% (rest_df). Commented out -- see below for the new
    # direction: predicting the 20% (rest_df), using a pool drawn from train_df.
    # few_shot_pool = rest_df.sample(n=FEW_SHOT_POOL_SIZE, random_state=RANDOM_STATE)
    # print(f"\n=== Storing {len(few_shot_pool)}-row few-shot pool via fit() ===")
    # llm.fit(few_shot_pool)
    # print(f"\n=== Predicting on {len(train_df)} articles (using stored few-shot pool) ===")
    # result = llm.predict(test_df=train_df)
    # eval_df = train_df

    # --- New approach: predict the 20% (rest_df), using a 5-row few-shot pool
    # drawn from the 80% (train_df) -- the set already predicted above.
    few_shot_pool = train_df.sample(n=FEW_SHOT_POOL_SIZE, random_state=RANDOM_STATE)
    print(f"\n=== Storing {len(few_shot_pool)}-row few-shot pool (from train_df) via fit() ===")
    llm.fit(few_shot_pool)

    print(f"\n=== Predicting on {len(rest_df)} articles (the other 20%) ===")
    result = llm.predict(test_df=rest_df)
    eval_df = rest_df

    print("\n=== Evaluating predictions ===")
    y_true = eval_df[ALL_COMMODITIES].values
    y_pred = [[1 if c in pred else 0 for c in ALL_COMMODITIES] for pred in result.predictions]
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
