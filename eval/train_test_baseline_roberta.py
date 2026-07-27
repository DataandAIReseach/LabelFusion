"""
Standalone ML-only (RoBERTa, no LLM/fusion) multi-label baseline: trained on
baseline_train.csv, validated on baseline_val.csv, evaluated on baseline_test.csv
(data/articles/baseline_split/, produced by create_baseline_split.py).

RoBERTa setup adapted from create_ml_model() in old/tests/eval_reuters.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from sklearn.metrics import classification_report

from textclassify.ml.roberta_classifier import RoBERTaClassifier
from textclassify.core.types import ModelConfig, ModelType

BASELINE_SPLIT_DIR = REPO_ROOT / "data" / "articles" / "baseline_split"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
TEXT_COLUMN = "text"

ROBERTA_MODEL = "roberta-base"
MAX_LENGTH = 256
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
BATCH_SIZE = 32


def load_splits():
    train_df = pd.read_csv(BASELINE_SPLIT_DIR / "baseline_train.csv")
    val_df = pd.read_csv(BASELINE_SPLIT_DIR / "baseline_val.csv")
    test_df = pd.read_csv(BASELINE_SPLIT_DIR / "baseline_test.csv")

    for df in (train_df, val_df, test_df):
        df[TEXT_COLUMN] = df["headline"].fillna("") + " " + df["body"].fillna("")

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_ml_model(experiment_name: str = "baseline_roberta") -> RoBERTaClassifier:
    ml_config = ModelConfig(
        model_name=ROBERTA_MODEL,
        model_type=ModelType.TRADITIONAL_ML,
        parameters={
            "model_name": ROBERTA_MODEL,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        },
    )
    return RoBERTaClassifier(
        config=ml_config,
        text_column=TEXT_COLUMN,
        label_columns=ALL_COMMODITIES,
        multi_label=True,
        auto_save_results=True,
        output_dir=str(OUTPUT_DIR),
        experiment_name=experiment_name,
    )


def evaluate(clf, split_df, split_name):
    result = clf.predict(split_df)

    y_true = split_df[ALL_COMMODITIES].values
    y_pred = [[1 if c in pred else 0 for c in ALL_COMMODITIES] for pred in result.predictions]
    y_pred = pd.DataFrame(y_pred, columns=ALL_COMMODITIES).values

    print(f"\n{'='*20} {split_name} {'='*20}")
    for i, c in enumerate(ALL_COMMODITIES):
        print(f"=== {c.upper()} ===")
        print(classification_report(y_true[:, i], y_pred[:, i]))

    exact_match = (y_true == y_pred).all(axis=1).mean()
    mean_accuracy = (y_true == y_pred).mean()
    print(f"Exact match (all {len(ALL_COMMODITIES)} correct): {exact_match:.3f}")
    print(f"Mean per-label accuracy: {mean_accuracy:.3f}")

    return result


def main():
    train_df, val_df, test_df = load_splits()

    ml_model = build_ml_model()

    print("\n=== Training RoBERTa on train set (validated on val set) ===")
    ml_model.fit(train_df, val_df)

    print("\n=== Evaluating on test set ===")
    evaluate(ml_model, test_df, "TEST")


if __name__ == "__main__":
    main()
