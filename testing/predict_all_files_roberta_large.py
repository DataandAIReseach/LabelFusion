"""RoBERTa-large alone (no LLM, no fusion) on every train/test pair; metrics averaged over seeds.

Every data/test_data/<dataset>-<seed>.xlsx (e.g. lab-manual-mm-test-5768) is paired with its
data/training_data/<dataset with -train->-<seed>.xlsx. Per pair:

  1. The train file is split 80/20 (stratified) into train / validation.
  2. Optuna tunes the learning rate (1e-7..1e-4, log), the batch size (4, 8, 16, 32) and the
     number of epochs (1-20) of textclassify's RoBERTaLargeClassifier: every trial fine-tunes a
     fresh roberta-large on the train split and is scored by weighted F1 on the validation split.
     The first 16 trials are always the full grid of learning rate {1e-4, 1e-5, 1e-6, 1e-7} x
     batch size {32, 16, 8, 4} at the default epoch count (--epochs, 3), so the search covers at
     least that grid search; --trials above 16 lets Optuna's sampler explore further (including
     the epochs). Everything else stays at its default (max_length 128, weight decay 0.01).
     0 epochs is left out because without fine-tuning the classification head is untrained.
     The splits give 64/16/20 train/validation/test, e.g. 1522 / 381 / 476 rows for combine.
  3. RoBERTa-large is trained with the best parameters and predicts the test file. The test
     file is never used for tuning.

The pairs of one dataset differ only in the seed of the resampling, so per dataset accuracy and
weighted F1 (and macro-F1) are averaged over the seeds, with the standard deviation (sample std, ddof=1).

Writes to ./outputs/roberta_large/:
    <test file name>.csv / .json   per pair: sentence, label, true, roberta_pred
    summary_seeds.csv              one row per pair (dataset, seed, metrics, chosen epochs + lr + batch size,
                                   val_f1_weighted = the best trial's validation score)
    trials/<test file name>_trials.csv   every Optuna trial of the pair: parameters + validation weighted F1
    summary.csv                    one row per dataset: <metric>_mean / <metric>_std over its seeds
    <dataset>_summary.json         per dataset: run settings, every seed's run, mean and std per metric
All summaries are rewritten after every pair, so a stopped run keeps its progress.

Fine-tuned models are never kept (~1.4 GB each): they are written to LABELFUSION_MODEL_CACHE
(default ~/.cache/labelfusion) by RoBERTaClassifier.fit() and deleted right after.

    python testing/predict_all_files_roberta_large.py                                   # every dataset, every seed: 16-point grid + 8 sampler trials
    python testing/predict_all_files_roberta_large.py --trials 16                       # grid only, no refinement
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
METRICS = ["roberta_f1_weighted", "roberta_f1_macro", "roberta_accuracy"]
GRID_LEARNING_RATES = [1e-4, 1e-5, 1e-6, 1e-7]
GRID_BATCH_SIZES = [32, 16, 8, 4]


def default_params(args) -> dict:
    return {"learning_rate": 1e-5, "num_epochs": args.epochs, "batch_size": 16, "weight_decay": 0.01, "max_length": 128,
            "class_weights": True}


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


def save_trials(study: optuna.Study, name: str) -> None:
    """All trials of one pair's study (parameters + validation weighted F1) -> trials/<name>_trials.csv.
    `source` tells the 16 enqueued grid points apart from the trials Optuna proposed itself."""
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    df = df.rename(columns={"value": "val_f1_weighted"})
    df.insert(1, "source", ["grid" if n < len(GRID_LEARNING_RATES) * len(GRID_BATCH_SIZES) else "optuna" for n in df["number"]])
    path = RESULT_DIR / "trials" / f"{name}_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def tune(train_df: pd.DataFrame, val_df: pd.DataFrame, name: str, args) -> tuple[dict, float]:
    """Optuna over learning_rate, batch_size and num_epochs; every trial is scored by weighted F1 on
    val_df. The 16 points of the learning-rate x batch-size grid always run first."""
    val_true = val_df["label"].map(LABEL_MAP).tolist()
    print(f"### tuning learning rate + batch size + epochs on {name}: train {len(train_df)} / val {len(val_df)}")

    def objective(trial: optuna.Trial) -> float:
        params = default_params(args)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True)
        params["batch_size"] = trial.suggest_categorical("batch_size", GRID_BATCH_SIZES)
        params["num_epochs"] = trial.suggest_int("num_epochs", 1, 20)
        return f1_score(val_true, fit_predict(params, train_df, val_df, val_df), average="weighted")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    epochs = min(max(args.epochs, 1), 20)
    for lr in GRID_LEARNING_RATES:
        for batch_size in GRID_BATCH_SIZES:
            study.enqueue_trial({"learning_rate": lr, "batch_size": batch_size, "num_epochs": epochs})
    study.optimize(objective, n_trials=max(args.trials, len(GRID_LEARNING_RATES) * len(GRID_BATCH_SIZES)))

    save_trials(study, name)
    params = default_params(args)
    params.update(study.best_trial.params)
    print(f"### best weighted F1 {study.best_value:.3f} after {len(study.trials)} trials: learning_rate="
          f"{params['learning_rate']:.2e} batch_size={params['batch_size']} num_epochs={params['num_epochs']}")
    return params, study.best_value


def run_pair(train_path: Path, test_path: Path, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    params, val_f1 = tune(train_df, val_df, name, args) if args.trials > 0 else (default_params(args), None)
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
    row["batch_size"] = params["batch_size"]
    row["val_f1_weighted"] = val_f1  # best trial on the validation split; compare with roberta_f1_weighted on test
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
    parser.add_argument("--trials", type=int, default=24,
                        help="Optuna trials per pair: the 16 grid points first, the rest is free refinement "
                             "(0 = no tuning, defaults)")
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
