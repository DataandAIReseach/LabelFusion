"""RoBERTa-large alone (no LLM, no fusion) on every train/test pair; metrics averaged over seeds.

Every data/test_data/<dataset>-<seed>.xlsx (e.g. lab-manual-mm-test-5768) is paired with its
data/training_data/<dataset with -train->-<seed>.xlsx. Per pair:

  1. The train file is split 80/20 (stratified) into train / validation.
  2. Optuna (--trials, default 5) tunes the number of epochs (1-20) and the learning rate
     (5e-6..3e-5, log) of textclassify's RoBERTaLargeClassifier: every trial fine-tunes a fresh
     roberta-large on the train split and is scored by macro-F1 on the validation split. The
     default configuration (3 epochs, lr 1e-5) is always the first trial. Everything else stays
     at its default (batch 16, max_length 128, weight decay 0.01). The learning-rate range sits
     lower than for roberta-base because roberta-large diverges more easily; 0 epochs is left
     out because without fine-tuning the classification head is untrained.
  3. RoBERTa-large is trained with the best parameters and predicts the test file. The test
     file is never used for tuning.

The pairs of one dataset differ only in the seed of the resampling, so per dataset accuracy and
macro-F1 are averaged over the seeds, with the standard deviation (sample std, ddof=1).

Writes to ./outputs/roberta_large/:
    <test file name>.csv / .json   per pair: sentence, label, true, roberta_pred
    summary_seeds.csv              one row per pair (dataset, seed, metrics, chosen epochs + lr)
    summary.csv                    one row per dataset: <metric>_mean / <metric>_std over its seeds
    <dataset>_summary.json         per dataset: run settings, every seed's run, mean and std per metric
All summaries are rewritten after every pair, so a stopped run keeps its progress.

Fine-tuned models are never kept (~1.4 GB each): they are written to LABELFUSION_MODEL_CACHE
(default ~/.cache/labelfusion) by RoBERTaClassifier.fit() and deleted right after.

    python testing/predict_all_files_roberta_large.py                                   # every dataset, every seed
    python testing/predict_all_files_roberta_large.py --datasets mm-test pc-split-test  # datasets containing these
    python testing/predict_all_files_roberta_large.py --datasets pc-test --limit-train 40 --trials 0 --epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

from predict_all_files_labelFusion import (  # noqa: E402
    LABEL_COLUMNS, LABEL_MAP, MODEL_CACHE, OUT_DIR, RANDOM_STATE, TEST_DIR, TEXT_COLUMN,
    delete_saved_model, load, paired_train, scores, split_pair,
)

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402

RESULT_DIR = OUT_DIR / "roberta_large"
SEED_RE = re.compile(r"^(?P<dataset>.+)-(?P<seed>\d+)$")
METRICS = ["roberta_accuracy", "roberta_f1_macro"]


def default_params(args) -> dict:
    return {"learning_rate": 1e-5, "num_epochs": args.epochs, "batch_size": 16, "weight_decay": 0.01, "max_length": 128}


def fit_predict(params: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, pred_df: pd.DataFrame) -> list[str]:
    """Fine-tune a fresh roberta-large on train_df (validating on val_df), predict pred_df."""
    config = ModelConfig(model_name="roberta-large", model_type=ModelType.TRADITIONAL_ML, parameters=dict(params))
    model = RoBERTaLargeClassifier(
        config=config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        auto_save_results=False,
        cache_dir=str(MODEL_CACHE),
    )
    model.fit(train_df, val_df)
    delete_saved_model(train_df)
    preds = model.predict_without_saving(pred_df).predictions
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def tune(train_df: pd.DataFrame, val_df: pd.DataFrame, name: str, args) -> dict:
    """Optuna over num_epochs and learning_rate; every trial is scored by macro-F1 on val_df."""
    val_true = val_df["label"].map(LABEL_MAP).tolist()
    print(f"### tuning epochs + learning rate on {name}: train {len(train_df)} / val {len(val_df)}")

    def objective(trial: optuna.Trial) -> float:
        params = default_params(args)
        params["num_epochs"] = trial.suggest_int("num_epochs", 1, 20)
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
        return f1_score(val_true, fit_predict(params, train_df, val_df, val_df), average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    d = default_params(args)
    # The defaults are always the first candidate, clamped into the tuned epoch range.
    study.enqueue_trial({"num_epochs": min(max(d["num_epochs"], 1), 20), "learning_rate": d["learning_rate"]})
    study.optimize(objective, n_trials=args.trials)

    params = default_params(args)
    params.update(study.best_trial.params)
    print(f"### best macro-F1 {study.best_value:.3f} after {len(study.trials)} trials: "
          f"num_epochs={params['num_epochs']} learning_rate={params['learning_rate']:.2e}")
    return params


def run_pair(train_path: Path, test_path: Path, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    params = tune(train_df, val_df, name, args) if args.trials > 0 else default_params(args)
    preds = fit_predict(params, train_df, val_df, test_df)

    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "roberta_pred": preds,
    })
    out.to_csv(RESULT_DIR / f"{name}.csv", index=False)
    out.to_json(RESULT_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)

    m = SEED_RE.match(name)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "rows": len(out),
           "seconds": round(time.time() - t0)}
    row.update({f"roberta_{k}": v for k, v in scores(out["true"].tolist(), preds).items()})
    row["num_epochs"], row["learning_rate"] = params["num_epochs"], params["learning_rate"]
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return row


def aggregate(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: mean and sample std (ddof=1) of every metric over its seeds."""
    grouped = seed_rows.groupby("dataset")
    summary = pd.DataFrame({
        "n_seeds": grouped.size(),
        "seeds": grouped["seed"].apply(lambda s: " ".join(sorted(s, key=int))),
    })
    for metric in METRICS:
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_std"] = grouped[metric].std(ddof=1)
    return summary.reset_index()


def write_dataset_json(dataset: str, seed_rows: pd.DataFrame, args) -> Path:
    """<dataset>_summary.json: the run settings, every seed's run, and mean / std per metric."""
    def clean(value):
        if isinstance(value, float) and math.isnan(value):  # std of a single seed
            return None
        return value.item() if hasattr(value, "item") else value

    runs = seed_rows[seed_rows["dataset"] == dataset]
    summary = aggregate(runs).iloc[0]
    payload = {
        "dataset": dataset,
        "model": "roberta-large",
        "seeds": summary["seeds"].split(),
        "n_seeds": int(summary["n_seeds"]),
        "settings": {"trials": args.trials, "limit_train": args.limit_train, "default_epochs": args.epochs},
        "mean": {metric: clean(summary[f"{metric}_mean"]) for metric in METRICS},
        "std": {metric: clean(summary[f"{metric}_std"]) for metric in METRICS},
        "runs": [{k: clean(v) for k, v in run.items()} for run in runs.to_dict(orient="records")],
    }
    path = RESULT_DIR / f"{dataset}_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only datasets (test file name without seed) containing one of these")
    parser.add_argument("--epochs", type=int, default=3, help="epochs of the default configuration")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Optuna trials per pair tuning epochs + learning rate (0 = no tuning, defaults)")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    for test_path in sorted(TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        m = SEED_RE.match(test_path.stem)
        if not m or (args.datasets and not any(d in m["dataset"] for d in args.datasets)):
            continue
        train_path = paired_train(test_path)
        if not train_path.exists():
            print(f"skip {test_path.stem}: no train file {train_path.name}")
            continue
        pairs.append((train_path, test_path))
    if not pairs:
        raise SystemExit("no matching train/test pairs found")
    print(f"{len(pairs)} pairs: {', '.join(p.stem for _, p in pairs)}")

    rows = []
    for train_path, test_path in pairs:
        row = run_pair(train_path, test_path, args)
        rows.append(row)
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(RESULT_DIR / "summary_seeds.csv", index=False)
        aggregate(seed_rows).to_csv(RESULT_DIR / "summary.csv", index=False)
        write_dataset_json(row["dataset"], seed_rows, args)

    summary = aggregate(pd.DataFrame(rows))
    print("\n=== mean +- std over seeds ===")
    for _, r in summary.iterrows():
        cells = "  ".join(f"{metric} {r[f'{metric}_mean']:.3f}+-{r[f'{metric}_std']:.3f}" for metric in METRICS)
        print(f"{r['dataset']} (n={r['n_seeds']}): {cells}")
    print(f"\nSaved: {RESULT_DIR / 'summary.csv'}, {RESULT_DIR / 'summary_seeds.csv'} and "
          f"one <dataset>_summary.json per dataset in {RESULT_DIR}")


if __name__ == "__main__":
    main()
