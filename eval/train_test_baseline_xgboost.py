"""
Baseline XGBoost multi-label classifier for the commodity articles dataset,
plus Optuna hyperparameter optimization on the val split.

Fuses the data-prep/baseline-training logic (cell 3.1 in
eval_silver_gold_oil_gas.ipynb) with the Optuna search (cell 3.2).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from xgboost import XGBClassifier

from eval.convert_jsons_to_csvs import build_dataframe_from_latest_json, _flatten_articles_json

ALL_COMMODITIES = ["gold", "silver", "oil", "gas"]
N_TRIALS = 40
LOGS_DIR = REPO_ROOT / "logs"

# Optional: set to a specific articles JSON path to use that file instead of
# the most recently modified one in data/articles/jsons. Relative paths are
# resolved against the repo root, not the current working directory, so this
# works no matter where you launch the script from.
JSON_PATH = "./data/articles/jsons/articles_all_commodities_baseline_seed7_temp17_n200_confounded.json"

logger = logging.getLogger(__name__)


def _setup_logging(timestamp: str) -> Path:
    """Log to both the console and a timestamped file in ./logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"train_test_baseline_xgboost_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return log_file


def load_and_split(json_path=None):
    """json_path is optional -- if given, load that specific articles JSON
    directly (resolved against REPO_ROOT if relative); otherwise fall back to
    the most recently modified JSON in data/articles/jsons
    (build_dataframe_from_latest_json's default)."""
    if json_path is not None:
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = REPO_ROOT / json_path
        logger.info(f"Loading: {json_path}")
        df = _flatten_articles_json(json_path)
    else:
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

    logger.info(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def _combo_labels(y):
    """Collapse a (n, len(ALL_COMMODITIES)) binary label matrix into single
    commodity-combination class labels, e.g. row [1,1,0,0] -> 'gold+silver'.
    Needed because the underlying problem is multi-label (rows can have
    several active commodities at once) -- this turns it into a genuine
    multiclass label so a standard confusion matrix applies."""
    return np.array([
        "+".join(c for c, v in zip(ALL_COMMODITIES, row) if v == 1) or "none"
        for row in y
    ])


def plot_confusion_matrix(y_split, y_pred, split_name):
    """Multiclass confusion matrix over commodity-combination labels (e.g.
    'gold', 'gold+silver', 'oil+gas+silver', ...) -- logged and saved as a
    PNG to logs/."""
    y_true_combo = _combo_labels(y_split)
    y_pred_combo = _combo_labels(y_pred)
    combo_labels = sorted(set(y_true_combo) | set(y_pred_combo))

    cm = confusion_matrix(y_true_combo, y_pred_combo, labels=combo_labels)
    logger.info(f"\nMulticlass confusion matrix over commodity combinations ({split_name}):")
    logger.info("\n" + pd.DataFrame(cm, index=combo_labels, columns=combo_labels).to_string())

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / f"confusion_matrix_{split_name.lower()}.png"

    fig, ax = plt.subplots(figsize=(max(6, len(combo_labels) * 0.6), max(5, len(combo_labels) * 0.6)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=combo_labels).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax.set_title(f"Commodity-combination confusion matrix — {split_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot: {out_path}")


def evaluate(clf, X_split, y_split, split_name):
    logger.info(f"\n{'='*20} {split_name} {'='*20}")
    y_pred = clf.predict(X_split)

    for i, c in enumerate(ALL_COMMODITIES):
        logger.info(f"=== {c.upper()} ===")
        logger.info(classification_report(y_split[:, i], y_pred[:, i]))

    exact_match = np.all(y_split == y_pred, axis=1).mean()
    mean_acc    = (y_split == y_pred).mean()
    f1_micro    = f1_score(y_split, y_pred, average='micro', zero_division=0)
    f1_macro    = f1_score(y_split, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_split, y_pred, average='weighted', zero_division=0)
    f1_samples  = f1_score(y_split, y_pred, average='samples', zero_division=0)

    logger.info(f"Exact match (all {len(ALL_COMMODITIES)} correct): {exact_match:.3f}")
    logger.info(f"Mean per-label accuracy: {mean_acc:.3f}")
    logger.info(f"F1 (micro):    {f1_micro:.3f}")
    logger.info(f"F1 (macro):    {f1_macro:.3f}")
    logger.info(f"F1 (weighted): {f1_weighted:.3f}")
    logger.info(f"F1 (samples):  {f1_samples:.3f}")

    plot_confusion_matrix(y_split, y_pred, split_name)

    return {
        "split": split_name,
        "exact_match": exact_match,
        "mean_accuracy": mean_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_samples": f1_samples,
    }


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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"\nBest val mean per-label accuracy: {study.best_value:.4f}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    return study


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _setup_logging(timestamp)
    logger.info(f"Logging to: {log_file}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(JSON_PATH)

    logger.info("\n=== Baseline XGBoost (fixed hyperparameters) ===")
    clf = train_baseline(X_train, y_train)
    val_results  = evaluate(clf, X_val, y_val, "VAL")
    test_results = evaluate(clf, X_test, y_test, "TEST")

    logger.info("\n=== Optuna hyperparameter optimization ===")
    optimize(X_train, y_train, X_val, y_val)

    results_file = LOGS_DIR / f"train_test_baseline_xgboost_{timestamp}_results.csv"
    pd.DataFrame([val_results, test_results]).to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
