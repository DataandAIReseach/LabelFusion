"""LabelFusion: roberta-large trained end-to-end with the LLM labels on every train/test pair; F1 averaged over seeds.

Every data/test_data/<dataset>-<seed>.xlsx (e.g. lab-manual-mm-test-5768) is paired with its
data/training_data/<dataset with -train->-<seed>.xlsx. roberta-large is NOT fine-tuned on its own
first: starting from the pretrained model, it is trained jointly with textclassify's fusion MLP
(FusionEnsemble, joint_training=True). The MLP's input is the roberta-large [CLS] embedding (1024)
concatenated with the LLM's label for the sentence, and the fusion loss is backpropagated through
the MLP and roberta-large together (two parameter groups: roberta_lr for roberta-large,
fusion_lr for the MLP).

  - The LLM expert is NOT called again: its labels come from the files written by
    predict_all_files_llm.py (./outputs/predictions/...):
      train file rows  <- GPT-5 nano, zero-shot   (train_gpt-5-nano_zero_shot)
      test file rows   <- GPT-5 nano, 5-shot      (test_gpt-5-nano_5_shot)
    The LLM only returns a hard class choice, so its fusion input is a one-hot class vector
    (dovish / hawkish / neutral), not real logits.

Per pair:
  1. The train file is split 80/20 (stratified) into train / validation.
  2. Optuna (--trials, default 5) tunes the number of joint epochs (0-20), the roberta-large
     learning rate (5e-6..3e-5, log) and the fusion MLP learning rate (1e-4..1e-2, log): every
     trial trains a fresh pretrained roberta-large + MLP on the train split and is scored by
     macro-F1 on the validation split. The default configuration (3 epochs, roberta lr 1e-5,
     MLP lr 1e-3) is always the first trial; everything else stays at its default (MLP hidden
     layers 64-32, batch 16, max_length 128). 0 epochs means nothing is trained.
  3. A fresh roberta-large + MLP is trained with the best parameters on the whole train file
     and predicts the test file. The test file is never used for tuning.
  FusionEnsemble holds back 10% of whatever it is trained on to monitor the loss, so the model
  effectively learns from 90% of that data.

There is no "RoBERTa alone" score here: roberta-large's own classification head is never
trained. For roberta-large on its own, see predict_all_files_roberta_large.py.

The pairs of one dataset differ only in the seed of the resampling, so per dataset the macro-F1
(and accuracy) of the LLM and of the fusion are averaged over the seeds, with the standard
deviation (sample std, ddof=1, i.e. the usual "mean +- std over N runs").

Writes to ./outputs/labelfusion_roberta_large/:
    <test file name>.csv / .json   per pair: sentence, label, true, llm_pred, fusion_pred
    summary_seeds.csv              one row per pair (dataset, seed, metrics, chosen epochs + lrs)
    summary.csv                    one row per dataset: <metric>_mean / <metric>_std over its seeds
    <dataset>_summary.json         per dataset: run settings, every seed's run, mean and std per metric
All summaries are rewritten after every pair, so a stopped run keeps its progress.

    python testing/predict_all_labelFusion.py                                   # every dataset, every seed
    python testing/predict_all_labelFusion.py --datasets mm-test pc-split-test  # datasets containing these
    python testing/predict_all_labelFusion.py --datasets pc-test --limit-train 40 --trials 0 --epochs 1  # smoke test
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

from predict_all_files_labelFusion import (  # noqa: E402
    LABEL_COLUMNS, LABEL_MAP, LLM_TEST_DIR, LLM_TRAIN_DIR, OUT_DIR, RANDOM_STATE, TEST_DIR, TEXT_COLUMN,
    EmbeddingMemo, PrecomputedLLM, build_fusion, llm_predictions, load, paired_train, scores, split_pair,
)

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402

RESULT_DIR = OUT_DIR / "labelfusion_roberta_large"
SEED_RE = re.compile(r"^(?P<dataset>.+)-(?P<seed>\d+)$")
METRICS = [f"{who}_{m}" for who in ("llm", "fusion") for m in ("f1_macro", "accuracy")]


def load_pretrained_roberta_large() -> tuple[RoBERTaLargeClassifier, EmbeddingMemo]:
    """A fresh pretrained roberta-large (no fine-tuning yet) for one joint training run."""
    config = ModelConfig(model_name="roberta-large", model_type=ModelType.TRADITIONAL_ML,
                         parameters={"max_length": 128, "batch_size": 16})
    model = RoBERTaLargeClassifier(
        config=config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        auto_save_results=False,
    )
    model.load_pretrained()  # joint training unfreezes it again inside FusionEnsemble
    return model, EmbeddingMemo(model)


def default_params(args) -> dict:
    return {"num_epochs": args.epochs, "ml_lr": 1e-5, "fusion_lr": 1e-3,
            "fusion_hidden_dims": [64, 32], "batch_size": 16}


def fit_predict(fus, train_df, pred_df, pred_llm, llm, name, tag, verbose) -> list[str]:
    """Train a fresh roberta-large jointly with the fusion MLP on train_df, predict pred_df."""
    def run(fn, *a, **kw):
        if verbose:
            return fn(*a, **kw)
        with contextlib.redirect_stdout(io.StringIO()):  # the library's progress prints
            return fn(*a, **kw)

    roberta, memo = load_pretrained_roberta_large()
    fusion = build_fusion(roberta, llm, {**fus, "joint_training": True}, name, tag, save=False)
    # FusionEnsemble trains on its val_df argument; roberta-large counts as "trained" (no separate
    # stage), so train_df is only used to look up LLM labels -- pass the same rows to both.
    run(fusion.fit, train_df, train_df)
    # Embeddings memoised before/while training come from older weights: predict with the trained ones.
    memo.cache.clear()
    preds = run(fusion.predict, pred_df, train_df=train_df, test_llm_predictions=pred_llm).predictions
    del fusion, roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def tune_params(train_df, val_df, llm, name, args) -> dict:
    """Optuna over the joint epochs and both learning rates: train on train_df, macro-F1 on val_df."""
    val_llm, val_true = llm.lookup(val_df), val_df["label"].map(LABEL_MAP).tolist()
    print(f"### tuning epochs + learning rates on {name}: train {len(train_df)} / val {len(val_df)}")

    def apply(params: dict) -> dict:
        fus = default_params(args)
        fus["num_epochs"], fus["ml_lr"], fus["fusion_lr"] = params["epochs"], params["roberta_lr"], params["fusion_lr"]
        return fus

    def objective(trial: optuna.Trial) -> float:
        fus = apply({
            "epochs": trial.suggest_int("epochs", 0, 20),
            "roberta_lr": trial.suggest_float("roberta_lr", 5e-6, 3e-5, log=True),
            "fusion_lr": trial.suggest_float("fusion_lr", 1e-4, 1e-2, log=True),
        })
        preds = fit_predict(fus, train_df, val_df, val_llm, llm, name, f"trial{trial.number}", verbose=False)
        return f1_score(val_true, preds, average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    d = default_params(args)
    # The defaults are always the first candidate, clamped into the tuned epoch range.
    study.enqueue_trial({"epochs": min(max(d["num_epochs"], 0), 20), "roberta_lr": d["ml_lr"], "fusion_lr": d["fusion_lr"]})
    study.optimize(objective, n_trials=args.trials)

    fus = apply(study.best_trial.params)
    print(f"### best macro-F1 {study.best_value:.3f} after {len(study.trials)} trials: epochs={fus['num_epochs']} "
          f"roberta_lr={fus['ml_lr']:.2e} fusion_lr={fus['fusion_lr']:.2e}")
    return fus


def run_pair(train_path: Path, test_path: Path, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    llm_test = llm_predictions(LLM_TEST_DIR / f"{test_path.stem}.csv")
    llm_test_preds = [llm_test[t] for t in test_df[TEXT_COLUMN]]

    fus = tune_params(train_df, val_df, llm, name, args) if args.trials > 0 else default_params(args)
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    fusion_preds = fit_predict(fus, full_train_df, test_df, llm_test_preds, llm, name, "final", verbose=True)

    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "llm_pred": llm_test_preds,
        "fusion_pred": fusion_preds,
    })
    shutil.rmtree(RESULT_DIR / "cache", ignore_errors=True)  # FusionEnsemble's per-run prediction caches
    out.to_csv(RESULT_DIR / f"{name}.csv", index=False)
    out.to_json(RESULT_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)

    m = SEED_RE.match(name)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "rows": len(out),
           "seconds": round(time.time() - t0)}
    for who in ("llm", "fusion"):
        row.update({f"{who}_{k}": v for k, v in scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row["epochs"], row["roberta_lr"], row["fusion_lr"] = fus["num_epochs"], fus["ml_lr"], fus["fusion_lr"]
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
        "model": "labelfusion: roberta-large trained jointly with gpt-5-nano labels",
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
    parser.add_argument("--epochs", type=int, default=3, help="joint training epochs of the default configuration")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Optuna trials per pair tuning epochs + both learning rates (0 = defaults)")
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
    print("\n=== macro-F1 mean +- std over seeds ===")
    for _, r in summary.iterrows():
        cells = "  ".join(f"{who} {r[f'{who}_f1_macro_mean']:.3f}+-{r[f'{who}_f1_macro_std']:.3f}"
                          for who in ("llm", "fusion"))
        print(f"{r['dataset']} (n={r['n_seeds']}): {cells}")
    print(f"\nSaved: {RESULT_DIR / 'summary.csv'}, {RESULT_DIR / 'summary_seeds.csv'} and "
          f"one <dataset>_summary.json per dataset in {RESULT_DIR}")


if __name__ == "__main__":
    main()
