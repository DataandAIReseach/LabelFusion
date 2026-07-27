"""
Baseline XGBoost multi-label classifier for the commodity articles dataset,
plus Optuna hyperparameter optimization on the val split.

Fuses the data-prep/baseline-training logic (cell 3.1 in
eval_silver_gold_oil_gas.ipynb) with the Optuna search (cell 3.2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from convert_json_to_csvs import build_dataframe_from_latest_json

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
N_TRIALS = 40


def load_and_split():
    df, _ = build_dataframe_from_latest_json()

    # Combine headline + body as input text
    df["text"] = df["headline"].fillna("") + " " + df["body"].fillna("")

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    y = df[ALL_COMMODITIES].values

    # 70/10/20 train/val/test split: split off the 70% train first, then split
    # the remaining 30% into val (10% overall) / test (20% overall).
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=1/3, random_state=42)

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(clf, X_split, y_split, split_name):
    print(f"\n{'='*20} {split_name} {'='*20}")
    y_pred = clf.predict(X_split)

    for i, c in enumerate(ALL_COMMODITIES):
        print(f"=== {c.upper()} ===")
        print(classification_report(y_split[:, i], y_pred[:, i]))

    print(f"Exact match (all {len(ALL_COMMODITIES)} correct): {np.all(y_split == y_pred, axis=1).mean():.3f}")
    print(f"Mean per-label accuracy: {(y_split == y_pred).mean():.3f}")


def train_baseline(X_train, y_train):
    # Multi-label XGBoost
    clf = MultiOutputClassifier(XGBClassifier(
        n_estimators   = 200,
        max_depth      = 6,
        learning_rate  = 0.1,
        eval_metric    = "logloss",
        random_state   = 42,
        tree_method    = "hist",
    ))
    clf.fit(X_train, y_train)
    return clf


def optimize(X_train, y_train, X_val, y_val, n_trials=N_TRIALS):
    # Optimize XGBoost hyperparameters against the val split.
    # Objective: mean per-label accuracy on val, matching the metric printed above.
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = MultiOutputClassifier(XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        ))
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return (y_val == y_val_pred).mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost_commodity_multilabel")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest val mean per-label accuracy: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    print("\n=== Baseline XGBoost (fixed hyperparameters) ===")
    clf = train_baseline(X_train, y_train)
    evaluate(clf, X_val, y_val, "VAL")
    evaluate(clf, X_test, y_test, "TEST")

    print("\n=== Optuna hyperparameter optimization ===")
    optimize(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
