"""
Baseline XGBoost classifier for the stance of the synthetic inflation /
unemployment articles (dovish / hawkish / neutral), plus Optuna
hyperparameter optimization.

Data: the articles JSON written by dataset_scripts/data_gen/article_creation.ipynb
(dataset_scripts/data/articles/jsons/*.json). Each article carries its stance in
"label" ("dovish" | "hawkish" | "neutral") and "label_id" (0 dovish, 1 hawkish,
2 neutral -- the lab-manual coding of the FOMC data).

Input text is the article body only (the headline is empty for these texts).

Splits (the datasets are small, e.g. 200 articles):
  - 80 % dev / 20 % test, stratified by label.
  - The fixed-parameter baseline and the Optuna search are scored with
    stratified 5-fold cross-validation on the dev part (macro-F1), so tuning
    never sees the test articles and is not decided on a 20-article val split.
  - The best parameters are refit on the whole dev part and scored on test.
TF-IDF sits inside the pipeline, so it is fitted on the training folds only.

Outputs in ./logs/: the log, the confusion matrices (PNG) and a results CSV.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

CLASSES = ["dovish", "hawkish", "neutral"]   # index = label_id (0 dovish, 1 hawkish, 2 neutral)
N_TRIALS = 30
CV_FOLDS = 5
TEST_FRAC = 0.2
RANDOM_STATE = 42
LOGS_DIR = REPO_ROOT / "logs"
JSONS_DIR = REPO_ROOT / "dataset_scripts" / "data" / "articles" / "jsons"

# Optional: set to a specific articles JSON path to use that file instead of
# the most recently modified one in JSONS_DIR. Relative paths are resolved
# against the repo root, not the current working directory.
JSON_PATH = None

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


def _latest_json() -> Path:
    files = sorted(JSONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No articles JSON found in {JSONS_DIR} -- run article_creation.ipynb first")
    return files[-1]


def load_articles(json_path=None) -> tuple[pd.DataFrame, Path]:
    """One row per article: text, label (0/1/2), and a few metadata columns for the log."""
    if json_path is None:
        json_path = _latest_json()
    else:
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = REPO_ROOT / json_path

    with open(json_path, encoding="utf-8") as f:
        articles = json.load(f)

    rows = []
    for art in articles:
        meta = art["metadata"]
        label = art.get("label_id", meta.get("label"))
        if label is None:
            raise ValueError(f"{json_path}: article {meta.get('obs_id')} has no label -- "
                             "is this an articles file of the inflation/unemployment generator?")
        rows.append({
            "obs_id":    meta["obs_id"],
            "date":      meta["article_date"],
            "topic":     meta.get("commodity", ""),
            "IN":        meta.get("IN"),
            "text":      " ".join(s for s in (art.get("headline", ""), art.get("body", "")) if s).strip(),
            "label":     int(label),
        })
    df = pd.DataFrame(rows)

    empty = (df["text"] == "").sum()
    if empty:
        logger.info(f"Dropping {empty} articles with an empty text")
        df = df[df["text"] != ""].reset_index(drop=True)

    return df, json_path


def make_pipeline(**xgb_params) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("xgb", XGBClassifier(
            objective    = "multi:softprob",
            num_class    = len(CLASSES),
            eval_metric  = "mlogloss",
            random_state = RANDOM_STATE,
            tree_method  = "hist",
            **xgb_params,
        )),
    ])


BASELINE_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}


def _cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_score(params, X_dev, y_dev) -> float:
    """Mean macro-F1 over stratified CV folds of the dev split."""
    return cross_val_score(make_pipeline(**params), X_dev, y_dev, cv=_cv(),
                           scoring="f1_macro").mean()


def plot_confusion_matrix(y_true, y_pred, stage, timestamp, n_obs):
    """3x3 confusion matrix on the test split, logged and saved as a PNG in logs/."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    logger.info(f"\nConfusion matrix ({stage}, TEST) -- rows = true, columns = predicted:")
    logger.info("\n" + pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / f"confusion_matrix_{stage}_test_{timestamp}_n{n_obs}.png"
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Stance confusion matrix -- {stage}, TEST\n{timestamp}  (n={n_obs})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot: {out_path}")


def evaluate(model, X_test, y_test, stage, timestamp, n_obs, cv_f1=None) -> dict:
    """Scores on the held-out test split. stage: "majority", "baseline" or "best"."""
    logger.info(f"\n{'=' * 20} {stage.upper()} TEST {'=' * 20}")
    y_pred = model.predict(X_test)

    logger.info(classification_report(y_test, y_pred, labels=range(len(CLASSES)),
                                      target_names=CLASSES, zero_division=0))
    result = {
        "stage":       stage,
        "split":       "TEST",
        "cv_f1_macro": cv_f1,
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1_macro":    f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    logger.info(f"Accuracy:      {result['accuracy']:.3f}")
    logger.info(f"F1 (macro):    {result['f1_macro']:.3f}")
    logger.info(f"F1 (weighted): {result['f1_weighted']:.3f}")

    if stage != "majority":
        plot_confusion_matrix(y_test, y_pred, stage, timestamp, n_obs)
    return result


class MajorityClass:
    """Predicts the most frequent dev label for every article -- the floor any model has to beat."""

    def fit(self, X, y):
        self.label_ = int(pd.Series(y).mode()[0])
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)


def optimize(X_dev, y_dev, n_trials=N_TRIALS):
    """Optuna over the XGBoost hyperparameters, objective = CV macro-F1 on the dev split."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return cv_score(params, X_dev, y_dev)

    study = optuna.create_study(direction="maximize", study_name="xgboost_stance",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.enqueue_trial(BASELINE_PARAMS)   # the baseline configuration is always a candidate
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"\nBest CV macro-F1 on dev: {study.best_value:.4f}")
    logger.info("Best params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
    return study


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _setup_logging(timestamp)
    logger.info(f"Logging to: {log_file}")

    df, json_path = load_articles(JSON_PATH)
    n_obs = len(df)
    logger.info(f"Dataset JSON: {json_path}")
    logger.info(f"Articles: {n_obs}  |  labels: "
                + ", ".join(f"{c} {int((df['label'] == i).sum())}" for i, c in enumerate(CLASSES)))

    X_dev, X_test, y_dev, y_test = train_test_split(
        df["text"], df["label"].to_numpy(), test_size=TEST_FRAC,
        random_state=RANDOM_STATE, stratify=df["label"])
    logger.info(f"Dev: {len(X_dev)} ({CV_FOLDS}-fold CV for tuning)  Test: {len(X_test)}")

    results = []

    logger.info("\n=== Majority class (reference) ===")
    results.append(evaluate(MajorityClass().fit(X_dev, y_dev), X_test, y_test,
                            "majority", timestamp, n_obs))

    logger.info("\n=== Baseline XGBoost (fixed hyperparameters) ===")
    baseline_cv = cv_score(BASELINE_PARAMS, X_dev, y_dev)
    logger.info(f"CV macro-F1 on dev: {baseline_cv:.4f}")
    baseline = make_pipeline(**BASELINE_PARAMS).fit(X_dev, y_dev)
    results.append(evaluate(baseline, X_test, y_test, "baseline", timestamp, n_obs, baseline_cv))

    logger.info("\n=== Optuna hyperparameter optimization ===")
    study = optimize(X_dev, y_dev)

    logger.info("\n=== Best XGBoost (best Optuna params, refit on all dev articles) ===")
    best = make_pipeline(**study.best_params).fit(X_dev, y_dev)
    results.append(evaluate(best, X_test, y_test, "best", timestamp, n_obs, study.best_value))

    results_file = LOGS_DIR / f"train_test_baseline_xgboost_{timestamp}_n{n_obs}_results.csv"
    pd.DataFrame(results).assign(dataset=json_path.name).to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
