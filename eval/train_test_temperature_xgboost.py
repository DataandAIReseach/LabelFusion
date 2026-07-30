"""
Compares XGBoost multi-label classification performance across the
temperature-sweep datasets produced by article_creation.ipynb's
"3.2 All other datasets" cell -- articles_all_commodities_baseline_seed7_
temp{T}_n{N}.json files in data/articles/jsons/ -- to see how the LLM
generation temperature used to write each dataset affects how easy it
is to classify.

For each temperature dataset: 70/10/20 train/val/test split, an Optuna
hyperparameter search against val (same search space as
train_test_baseline_xgboost.py), then a final fit with the best params
evaluated on val and test.
"""

import glob
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from eval.convert_jsons_to_csvs import _flatten_articles_json

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
JSONS_DIR = REPO_ROOT / "data" / "articles" / "jsons"
TEMP_FILE_PATTERN = "articles_all_commodities_baseline_seed7_temp*_n*.json"
LOGS_DIR = REPO_ROOT / "logs"
N_TRIALS = 40

logger = logging.getLogger(__name__)


def _setup_logging(timestamp: str) -> Path:
    """Log to both the console and a timestamped file in ./logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"train_test_temperature_xgboost_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Optuna's own per-trial logging would otherwise flood 10 runs x
    # N_TRIALS trials worth of output; keep only warnings/errors from it.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return log_file


def find_temperature_datasets():
    """Locate every temperature-sweep JSON and parse its temperature/sample
    count back out of the filename (..._temp{T}_n{N}.json, e.g. temp12 -> 1.2)."""
    files = sorted(glob.glob(str(JSONS_DIR / TEMP_FILE_PATTERN)))
    datasets = []
    for path in files:
        m = re.search(r"_temp(\d+)_n(\d+)\.json$", path)
        if not m:
            continue
        temp_str, n_str = m.groups()
        temperature = float(f"{temp_str[0]}.{temp_str[1:]}")
        datasets.append((temperature, int(n_str), path))
    datasets.sort(key=lambda d: d[0])
    return datasets


def load_and_split(json_path):
    df = _flatten_articles_json(json_path)
    df["text"] = df["headline"].fillna("") + " " + df["body"].fillna("")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["text"])
    y = df[ALL_COMMODITIES].values

    # 70/10/20 train/val/test: split off the 70% train first, then split
    # the remaining 30% into val (10% overall) / test (20% overall).
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=1/3, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(X_train, y_train, params=None):
    clf = MultiOutputClassifier(XGBClassifier(
        **(params or dict(n_estimators=200, max_depth=6, learning_rate=0.1)),
        eval_metric  = "logloss",
        random_state = 42,
        tree_method  = "hist",
    ))
    clf.fit(X_train, y_train)
    return clf


def optimize(X_train, y_train, X_val, y_val, n_trials=N_TRIALS):
    """Optuna search against val mean per-label accuracy (same space as
    train_test_baseline_xgboost.py's optimize())."""
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
            **params, eval_metric="logloss", random_state=42, tree_method="hist",
        ))
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return (y_val == y_val_pred).mean()

    study = optuna.create_study(direction="maximize", study_name="xgboost_commodity_multilabel")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Best val mean per-label accuracy: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def evaluate(clf, X_split, y_split, temperature, split_name):
    y_pred = clf.predict(X_split)

    logger.info(f"\n{'='*20} TEMPERATURE={temperature} {split_name} {'='*20}")
    for i, c in enumerate(ALL_COMMODITIES):
        logger.info(f"=== {c.upper()} ===")
        logger.info(classification_report(y_split[:, i], y_pred[:, i]))

    exact_match = np.all(y_split == y_pred, axis=1).mean()
    mean_acc    = (y_split == y_pred).mean()
    logger.info(f"[temp={temperature}][{split_name}]  exact match: {exact_match:.3f}   "
                f"mean per-label accuracy: {mean_acc:.3f}")

    return exact_match, mean_acc


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _setup_logging(timestamp)
    logger.info(f"Logging to: {log_file}")

    datasets = find_temperature_datasets()
    if not datasets:
        raise FileNotFoundError(
            f"No temperature-sweep files found matching {TEMP_FILE_PATTERN} in {JSONS_DIR}")

    logger.info(f"Found {len(datasets)} temperature datasets: {[t for t, n, p in datasets]}")

    results = []
    for temperature, n_samples, json_path in datasets:
        logger.info(f"\n{'#'*60}\n# TEMPERATURE = {temperature}  ({n_samples} samples)  {Path(json_path).name}\n{'#'*60}")
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(json_path)
        logger.info(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

        logger.info("--- Baseline (fixed hyperparameters) ---")
        baseline_clf = train_baseline(X_train, y_train)
        evaluate(baseline_clf, X_val, y_val, temperature, "VAL (baseline)")
        base_test_exact, base_test_acc = evaluate(baseline_clf, X_test, y_test, temperature, "TEST (baseline)")

        logger.info("--- Optuna search ---")
        study = optimize(X_train, y_train, X_val, y_val)
        tuned_clf = train_baseline(X_train, y_train, params=study.best_params)
        tuned_val_exact, tuned_val_acc = evaluate(tuned_clf, X_val, y_val, temperature, "VAL (tuned)")
        tuned_test_exact, tuned_test_acc = evaluate(tuned_clf, X_test, y_test, temperature, "TEST (tuned)")

        results.append({
            "temperature":        temperature,
            "n_samples":          n_samples,
            "dataset_file":       Path(json_path).name,
            "baseline_test_exact_match": base_test_exact,
            "baseline_test_mean_accuracy": base_test_acc,
            "tuned_val_exact_match":     tuned_val_exact,
            "tuned_val_mean_accuracy":   tuned_val_acc,
            "tuned_test_exact_match":    tuned_test_exact,
            "tuned_test_mean_accuracy":  tuned_test_acc,
            "best_params":        study.best_params,
        })

    summary = pd.DataFrame(results).sort_values("temperature")
    logger.info(f"\n{'='*20} SUMMARY ACROSS TEMPERATURES {'='*20}")
    logger.info("\n" + summary.drop(columns=["best_params"]).to_string(index=False))

    results_file = LOGS_DIR / f"train_test_temperature_xgboost_{timestamp}_results.csv"
    summary.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()