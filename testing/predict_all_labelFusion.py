"""LabelFusion, two stages: roberta-large fine-tuned alone, then a fusion MLP on its frozen embeddings + LLM labels.

Every data/test_data/<dataset>-<seed>.xlsx (e.g. lab-manual-mm-test-5768) is paired with its
data/training_data/<dataset with -train->-<seed>.xlsx. Per pair:

  Splits   test = the test file (20%); the train file is split 80/20 (stratified) into train and
           validation, i.e. 64 / 16 / 20 of the data (e.g. 1522 / 381 / 476 rows for combine).

  Stage 1  roberta-large alone, on the train split. Optuna (--trials, default 24) searches the
           learning rate (1e-7..1e-4, log), the batch size (4, 8, 16, 32) and the epochs (1-20);
           the first 16 trials are always the full grid learning rate {1e-4, 1e-5, 1e-6, 1e-7} x
           batch size {32, 16, 8, 4} (--epochs, 10), the rest is free refinement around it. Every
           trial fine-tunes a fresh roberta-large and is scored by weighted F1 on the validation
           split. The best parameters train the final roberta-large on the train split. As in
           predict_all_files_roberta_large.py, training uses class weights, learning-rate warmup and
           best-epoch selection on the validation split (epochs = maximum), and the final training
           is repeated with another seed (--retries, default 2) if it collapses to one class.

  Stage 2  roberta-large is frozen; only its [CLS] embedding (1024) is used. textclassify's fusion
           MLP learns from that embedding concatenated with the LLM's label for the sentence,
           with a much higher learning rate than stage 1 (1e-4..1e-2). It is trained on the
           VALIDATION split, which roberta-large never trained on: on the sentences roberta-large
           was fine-tuned on, its embeddings are too good, and an MLP trained there would learn to
           trust them over the LLM label. Optuna (--mlp-trials, default 30) tunes the MLP's learning
           rate, epochs (1-50), hidden layers and batch size; as roberta-large is frozen, its
           embeddings are computed once, so these trials are cheap. Each trial trains on one half
           of the validation split and is scored (weighted F1) on the other half; the best parameters
           then train the MLP on the whole validation split, and it predicts the test file.
           FusionEnsemble holds back 10% of whatever the MLP is trained on to monitor its loss.

The LLM is NOT called again: its labels come from the files written by predict_all_files_llm.py
(./outputs/predictions/...): train file rows <- GPT-5 nano zero-shot (train_gpt-5-nano_zero_shot),
test file rows <- GPT-5 nano 5-shot (test_gpt-5-nano_5_shot). It only returns a hard class choice, so
its fusion input is a one-hot vector (dovish / hawkish / neutral), not real logits.
The test file is never used for tuning or training.

Scored on the test file, per pair: the LLM alone, roberta-large alone (stage 1) and the fusion, by weighted F1
(the paper's metric; macro-F1 and accuracy are reported too, and tuning optimises weighted F1). The
pairs of one dataset differ only in the seed of the resampling, so per dataset the weighted F1 (and macro-F1,
accuracy) of all three are averaged over the seeds, with the standard deviation (sample std, ddof=1).

Writes to ./outputs/labelfusion_roberta_large/:
    <test file name>.csv / .json   per pair: sentence, label, true, llm_pred, roberta_pred, fusion_pred
    summary_seeds.csv              one row per pair (dataset, seed, metrics, chosen parameters of both
                                   stages, roberta_val_f1 / mlp_val_f1 = the best trials' scores)
    summary.csv                    one row per dataset: <metric>_mean / <metric>_std over its seeds
    <dataset>_summary.json         per dataset: run settings, every seed's run, mean and std per metric
    trials/<test file name>_trials.csv      every stage-1 Optuna trial: parameters + validation weighted F1
    trials/<test file name>_mlp_trials.csv  every stage-2 Optuna trial
All summaries are rewritten after every pair, so a stopped run keeps its progress.

    python testing/predict_all_labelFusion.py                                   # every dataset, every seed
    python testing/predict_all_labelFusion.py --datasets mm-test pc-split-test  # datasets containing these
    python testing/predict_all_labelFusion.py --trials 16                       # stage-1 grid only, no refinement
    python testing/predict_all_labelFusion.py --datasets pc-test --limit-train 40 --trials 0 --mlp-trials 0 --epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

import predict_all_files_roberta_large as stage1  # noqa: E402
from predict_all_files_labelFusion import (  # noqa: E402
    HIDDEN_DIMS, LABEL_COLUMNS, LABEL_MAP, LLM_TEST_DIR, LLM_TRAIN_DIR, MODEL_CACHE, OUT_DIR, RANDOM_STATE,
    TEST_DIR, TEXT_COLUMN, EmbeddingMemo, PrecomputedLLM, build_fusion, delete_saved_model, llm_predictions,
    load, paired_train, scores, split_pair,
)

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402

RESULT_DIR = OUT_DIR / "labelfusion_roberta_large"
stage1.RESULT_DIR = RESULT_DIR  # stage 1's trials/<pair>_trials.csv go into this script's folder
SEED_RE = re.compile(r"^(?P<dataset>.+)-(?P<seed>\d+)$")
METRICS = [f"{who}_{m}" for who in ("llm", "roberta", "fusion") for m in ("f1_weighted", "f1_macro", "accuracy")]


def train_roberta(params: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, retries: int = 0):
    """Fine-tune roberta-large on train_df (monitored on val_df; a run collapsing to one class is retried
    with another seed, see stage 1's train_model), then freeze it and memoise its embeddings."""
    model = stage1.train_model(params, train_df, val_df, retries)
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False
    return model, EmbeddingMemo(model)


def default_mlp_params() -> dict:
    return {"fusion_hidden_dims": [64, 32], "fusion_lr": 1e-3, "num_epochs": 20, "batch_size": 16}


def mlp_fit_predict(roberta, llm, mlp, rob_train_df, mlp_train_df, pred_df, pred_llm, name, tag, verbose) -> list[str]:
    """Train the fusion MLP on mlp_train_df (frozen roberta embeddings + LLM labels), predict pred_df."""
    def run(fn, *a, **kw):
        if verbose:
            return fn(*a, **kw)
        with contextlib.redirect_stdout(io.StringIO()):  # the library's progress prints
            return fn(*a, **kw)

    fusion = build_fusion(roberta, llm, mlp, name, tag, save=False)
    run(fusion.fit, rob_train_df, mlp_train_df)  # roberta counts as trained -> only the MLP is fitted
    preds = run(fusion.predict, pred_df, train_df=rob_train_df, test_llm_predictions=pred_llm).predictions
    del fusion
    gc.collect()
    return preds


def tune_mlp(roberta, llm, rob_train_df, val_df, name, args) -> tuple[dict, float]:
    """Optuna over the MLP's learning rate, epochs, hidden layers and batch size: fit on one half of
    val_df, weighted F1 on the other half."""
    try:
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE, stratify=val_df["label"])
    except ValueError:  # a class too small to stratify
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE)
    fit_df, eval_df = fit_df.reset_index(drop=True), eval_df.reset_index(drop=True)
    eval_llm, eval_true = llm.lookup(eval_df), eval_df["label"].map(LABEL_MAP).tolist()
    print(f"### tuning the fusion MLP on {name}: fit {len(fit_df)} / eval {len(eval_df)}")

    def apply(params: dict) -> dict:
        return {"fusion_lr": params["fusion_lr"], "num_epochs": params["epochs"], "batch_size": params["batch_size"],
                "fusion_hidden_dims": HIDDEN_DIMS[params["hidden_dims"]]}

    def objective(trial: optuna.Trial) -> float:
        mlp = apply({
            "fusion_lr": trial.suggest_float("fusion_lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 1, 50),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "hidden_dims": trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMS)),
        })
        preds = mlp_fit_predict(roberta, llm, mlp, rob_train_df, fit_df, eval_df, eval_llm, name,
                                f"mlp{trial.number}", verbose=False)
        return f1_score(eval_true, preds, average="weighted")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    d = default_mlp_params()
    study.enqueue_trial({"fusion_lr": d["fusion_lr"], "epochs": d["num_epochs"], "batch_size": d["batch_size"],
                         "hidden_dims": "64-32"})
    study.optimize(objective, n_trials=args.mlp_trials)

    path = RESULT_DIR / "trials" / f"{name}_mlp_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(attrs=("number", "value", "params", "state", "duration")).rename(
        columns={"value": "val_f1_weighted"}).to_csv(path, index=False)
    mlp = apply(study.best_trial.params)
    print(f"### best MLP weighted F1 {study.best_value:.3f} after {len(study.trials)} trials: fusion_lr={mlp['fusion_lr']:.2e} "
          f"epochs={mlp['num_epochs']} batch_size={mlp['batch_size']} hidden={mlp['fusion_hidden_dims']}")
    return mlp, study.best_value


def run_pair(train_path: Path, test_path: Path, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    llm_test = llm_predictions(LLM_TEST_DIR / f"{test_path.stem}.csv")
    llm_test_preds = [llm_test[t] for t in test_df[TEXT_COLUMN]]

    # Stage 1: roberta-large alone.
    if args.trials > 0:
        rob, roberta_val_f1 = stage1.tune(train_df, val_df, name, args)
    else:
        rob, roberta_val_f1 = stage1.default_params(args), None
    roberta, memo = train_roberta(rob, train_df, val_df, retries=args.retries)
    roberta_preds = roberta.predict_without_saving(test_df).predictions

    # Stage 2: frozen roberta-large embeddings + LLM labels -> fusion MLP, trained on the validation split.
    if args.mlp_trials > 0:
        mlp, mlp_val_f1 = tune_mlp(roberta, llm, train_df, val_df, name, args)
    else:
        mlp, mlp_val_f1 = default_mlp_params(), None
    fusion_preds = mlp_fit_predict(roberta, llm, mlp, train_df, val_df, test_df, llm_test_preds, name, "final", verbose=True)
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "llm_pred": llm_test_preds,
        "roberta_pred": roberta_preds,
        "fusion_pred": fusion_preds,
    })
    shutil.rmtree(RESULT_DIR / "cache", ignore_errors=True)  # FusionEnsemble's per-run prediction caches
    out.to_csv(RESULT_DIR / f"{name}.csv", index=False)
    out.to_json(RESULT_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)

    m = SEED_RE.match(name)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "rows": len(out),
           "seconds": round(time.time() - t0)}
    for who in ("llm", "roberta", "fusion"):
        row.update({f"{who}_{k}": v for k, v in scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row.update({"roberta_lr": rob["learning_rate"], "roberta_batch_size": rob["batch_size"],
                "roberta_epochs": rob["num_epochs"], "roberta_val_f1": roberta_val_f1,
                "mlp_lr": mlp["fusion_lr"], "mlp_epochs": mlp["num_epochs"], "mlp_batch_size": mlp["batch_size"],
                "mlp_hidden": "-".join(map(str, mlp["fusion_hidden_dims"])), "mlp_val_f1": mlp_val_f1})
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
        "model": "labelfusion: roberta-large fine-tuned alone, then fusion MLP on frozen embeddings + gpt-5-nano labels",
        "seeds": summary["seeds"].split(),
        "n_seeds": int(summary["n_seeds"]),
        "settings": {"trials": args.trials, "mlp_trials": args.mlp_trials, "limit_train": args.limit_train,
                     "default_roberta_epochs": args.epochs, "retries": args.retries},
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
    parser.add_argument("--epochs", type=int, default=10,
                        help="roberta-large (maximum) epochs of the default configuration and of the 16 grid points")
    parser.add_argument("--retries", type=int, default=2,
                        help="repeat the final roberta-large training with another seed up to N times if it collapses to one class")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--trials", type=int, default=24,
                        help="stage-1 Optuna trials per pair: the 16 grid points first, the rest is free refinement "
                             "(0 = defaults)")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="stage-2 Optuna trials per pair tuning the fusion MLP (0 = defaults)")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RESULT_DIR)  # FusionEnsemble writes its prediction caches to ./cache

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
    print("\n=== weighted F1 mean +- std over seeds ===")
    for _, r in summary.iterrows():
        cells = "  ".join(f"{who} {r[f'{who}_f1_weighted_mean']:.3f}+-{r[f'{who}_f1_weighted_std']:.3f}"
                          for who in ("llm", "roberta", "fusion"))
        print(f"{r['dataset']} (n={r['n_seeds']}): {cells}")
    print(f"\nSaved: {RESULT_DIR / 'summary.csv'}, {RESULT_DIR / 'summary_seeds.csv'} and "
          f"one <dataset>_summary.json per dataset in {RESULT_DIR}")


if __name__ == "__main__":
    main()
