"""Fusion on the LOGITS of roberta-large instead of its embedding: a small stacking model on roberta-large + the LLM label.

predict_low_data_regimes.py fuses roberta-large's 1024-dim [CLS] embedding with the LLM's label (3 numbers) in an MLP. The LLM label
is then under 0.5 % of the input and the fusion ends up below roberta-large alone, although the two models make different errors
(roberta_llm_label_compare.py measures an oracle accuracy about 15 points above the better one of the two). This script tests the
opposite design: the fusion model sees only 6 numbers per sentence, so roberta-large and the LLM have the same weight and the model cannot
over-fit on embedding noise.

For every test file and every regime p (the same runs, regimes and LLM caches as predict_low_data_regimes.py) it
  1. splits the regime 80/20 (train / validation) and fine-tunes roberta-large ALONE on the train split, exactly like the column
     "roberta" there (Optuna over lr x batch size x epochs, class weights, warmup, best epoch, retries),
  2. takes roberta-large's log-probabilities (= logits up to a constant per sentence) of the validation split and of the full test set,
  3. fits a logistic regression on the VALIDATION split (roberta-large was not trained on it, so its logits there are honest; on its own
     train sentences they are over-confident), class weights balanced, C chosen by 5-fold cross-validation inside the validation split,
  4. applies it to the test set.
Columns of the predictions:
  llm          the LLM labels of the regime on the full test set (few-shot, 2 examples per 10 %): the reference. Not trained.
  roberta      roberta-large fine-tuned alone on the train split.
  fusion_logit logistic regression on [roberta-large log-probabilities (3), LLM label as one-hot (3)]. On the validation split the LLM label
               is the zero-shot one, on the test set the regime's few-shot one (as in all fusion runs).
  logit_only   the same without the LLM label: a recalibration of roberta-large on the validation split. fusion_logit - logit_only is
               what the LLM label adds; logit_only - roberta is what the recalibration alone does.
The validation split is small (38 to 229 sentences for 10 to 60 %), but 6 inputs are few enough for that. roberta-large chose its best
epoch on this split, so its logits there are slightly optimistic (not trained on, but selected on).

Resumable like predict_low_data_regimes.py (checkpoints/: every Optuna trial that was done is read from disk, finished runs are
skipped; only the final fine-tuning of an interrupted run is repeated). Output: ./outputs/predict_low_data_regimes_logit/
    <test file>_<p>_logit_<date_time>_<job id>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred, fusion_logit_pred, logit_only_pred
    summary_seeds_logit_<date_time>_<job id>.csv             one row per run (all metrics, roberta parameters, C of the logistic regressions)
    summary_logit_<date_time>_<job id>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_logit_<date_time>_<job id>.csv, trials/...

    python testing/predict_low_data_regimes_logit.py --dry-run
    python testing/predict_low_data_regimes_logit.py --datasets pc-split-test --percents 10 70 --trials 0 --epochs 2   # smoke test
    python testing/predict_low_data_regimes_logit.py                                                                  # everything
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

import predict_low_data_regimes as plr  # noqa: E402  (its data, cache and checkpoint helpers; it also puts testing/old on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

base = plr.base
stage1 = plr.stage1
RESULT_DIR = TESTING_DIR.parent / "outputs" / "predict_low_data_regimes_logit"
WORK_DIR = RESULT_DIR / ".work" / plr.RUN_ID
SUFFIX = f"logit_{datetime.now():%Y%m%d_%H%M%S}_{plr.RUN_ID}"  # start time + job id: parallel jobs never share a file name
WHO = ("llm", "roberta", "fusion_logit", "logit_only")
METRICS = [f"{who}_{m}" for who in WHO for m in ("f1_weighted", "f1_macro", "accuracy")]
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)

# predict_low_data_regimes.py's helpers read these module globals when they are called: point them at this script's folders
plr.RESULT_DIR, plr.WORK_DIR, plr.SUFFIX, plr.CHECKPOINT_DIR = RESULT_DIR, WORK_DIR, SUFFIX, RESULT_DIR / "checkpoints"
plr.WHO, plr.METRICS = WHO, METRICS


def stack_features(probabilities: list[dict], llm_labels: list[str] | None = None) -> np.ndarray:
    """roberta-large's log-probabilities (3), plus the LLM label as one-hot (3) if llm_labels is given."""
    classes = base.LABEL_COLUMNS
    features = np.log(np.clip(np.array([[p.get(c, 0.0) for c in classes] for p in probabilities]), 1e-6, 1.0))
    if llm_labels is not None:
        features = np.hstack([features, np.array([[1.0 if label == c else 0.0 for c in classes] for label in llm_labels])])
    return features


def fit_predict_stack(x_val: np.ndarray, y_val: list[str], x_test: np.ndarray, fallback: list[str]) -> tuple[list[str], float | None]:
    """Logistic regression fitted on the validation split, applied to the test set; returns the labels and the chosen C."""
    try:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=base.RANDOM_STATE)
        model = LogisticRegressionCV(Cs=CS, cv=folds, scoring="f1_weighted", class_weight="balanced", max_iter=2000).fit(x_val, y_val)
        return model.predict(x_test).tolist(), float(model.C_[0])
    except ValueError:  # a class with fewer than 5 validation sentences: no cross-validation, a fixed regularisation
        try:
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(x_val, y_val)
            return model.predict(x_test).tolist(), 1.0
        except ValueError:  # a single class in the validation split: nothing to learn, keep roberta-large's labels
            return list(fallback), None


def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = plr.prepare(test_path, train_path, llm_path, percent)
    train_df, val_df = plr.split_regime(regime_df)
    print(f"\n=== {name}: regime {len(regime_df)} (train {len(train_df)} / val {len(val_df)}) / test {len(test_df)} ===")
    row.update({"training": "finetuned", "train_rows": len(train_df), "val_rows": len(val_df)})

    # roberta-large alone, as stage 1 of predict_low_data_regimes.py: Optuna over lr x batch size x epochs, then the final fine-tuning
    if args.trials > 0:
        params, val_f1 = stage1.tune(train_df, val_df, name, args)
    else:
        params, val_f1 = stage1.default_params(args), None
    roberta, memo = base.train_roberta(params, train_df, val_df, retries=args.retries)
    # base.train_roberta wraps the model in EmbeddingMemo, whose predict() returns only labels and embeddings: the class's own method
    # (called on the instance, bypassing the memo) also returns the class probabilities
    rob_val = plr.RoBERTaLargeClassifier.predict_without_saving(roberta, val_df)
    rob_test = plr.RoBERTaLargeClassifier.predict_without_saving(roberta, test_df)
    if rob_val.probabilities is None or rob_test.probabilities is None:
        raise RuntimeError("roberta-large returned no class probabilities")
    out["roberta_pred"] = rob_test.predictions
    row.update({"roberta_lr": params["learning_rate"], "roberta_batch_size": params["batch_size"],
                "roberta_epochs": params["num_epochs"], "roberta_val_f1": val_f1})
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # the logistic regressions on the validation split
    y_val = val_df["label"].map(base.LABEL_MAP).tolist()
    llm_val_labels = llm_train.lookup(val_df)  # the zero-shot labels of the validation sentences
    out["fusion_logit_pred"], row["fusion_logit_C"] = fit_predict_stack(
        stack_features(rob_val.probabilities, llm_val_labels), y_val, stack_features(rob_test.probabilities, llm_test_preds), rob_test.predictions)
    out["logit_only_pred"], row["logit_only_C"] = fit_predict_stack(
        stack_features(rob_val.probabilities), y_val, stack_features(rob_test.probabilities), rob_test.predictions)
    return plr.finish(row, out, t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found, 10 to 100 %%)")
    parser.add_argument("--llm-cache-dir", type=Path, default=plr.LLM_CACHE_DIR, help="the LLM caches (default: outputs/llm_cache)")
    parser.add_argument("--llm-model", default=plr.LLM_MODEL, help="the LLM whose cache folders are read (e.g. gpt-5-nano)")
    parser.add_argument("--trials", type=int, default=24,
                        help="roberta-large Optuna trials per run (the 16 grid points first, the rest is free); 0 = the default configuration")
    parser.add_argument("--epochs", type=int, default=10, help="(maximum) epochs of the default configuration and of the grid points")
    parser.add_argument("--retries", type=int, default=2, help="repeat a fine-tuning that collapsed to one class with another seed")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip runs that already have a result file in the output folder (default; --no-resume repeats them)")
    args = parser.parse_args()
    args.llm_cache_dir = args.llm_cache_dir.resolve()

    plr.LLM_CACHE_DIR, plr.LLM_MODEL = args.llm_cache_dir, args.llm_model
    if plr.find_cache(plr.train_cache_name()) is None:
        raise SystemExit(f"no zero-shot cache predict_all_{plr.train_cache_name()} with a cache_*.pkl in {plr.LLM_CACHE_DIR}")
    try:
        print(f"zero-shot LLM labels of the training sentences: {len(plr.train_labels())}")
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")

    todo = plr.jobs(args)
    if not todo:
        raise SystemExit("no runs found (are the regimes of dataset_scripts/make_low_data_regimes.py and the LLM caches there?)")
    if args.resume:
        done = {re.sub(r"_logit_\d{8}_\d{6}(_\w+)?$", "", p.stem) for p in RESULT_DIR.glob("lab-manual-*_logit_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs ({', '.join(sorted({str(j[3]) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j[0] for j in todo})} test files; models: llm, roberta, fusion_logit, logit_only "
          f"(roberta-large fine-tuned with {args.trials} Optuna trials per run, then logistic regressions)")
    if args.dry_run or not todo:
        for test_path, train_path, llm_path, percent in todo[:10]:
            print(f"  {test_path.stem}_{percent}: regime {train_path.name}, LLM cache {llm_path.name}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    os.chdir(WORK_DIR)  # the library writes its caches to ./cache: this run's own folder
    staging = WORK_DIR / "out"
    staging.mkdir()
    base.RESULT_DIR = stage1.RESULT_DIR = plr.files.FUSION_DIR = staging

    plr.install_checkpoint_wrappers()
    rows, failures = [], []
    for test_path, train_path, llm_path, percent in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name = f"{test_path.stem}_{percent}"
        plr.CKPT.start(name, {"regime": hashlib.md5(train_path.read_bytes()).hexdigest(), "test": test_path.name, "percent": percent,
                              "epochs": args.epochs, "llm_model": plr.LLM_MODEL, "kind": "logit", "v": 1})
        try:
            row, out = run_job(test_path, train_path, llm_path, percent, args)
        except Exception as e:  # one failing run must not stop the sweep
            traceback.print_exc()
            plr.CKPT.finish(delete=False)  # keep what was computed: the next start of this run continues from it
            failures.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(failures).to_csv(RESULT_DIR / f"failures_{SUFFIX}.csv", index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        out.to_csv(staging / f"{name}.csv", index=False)
        out.to_json(staging / f"{name}.json", orient="records", force_ascii=False, indent=2)
        plr.publish(staging)
        print(f"### {name}: {plr.CKPT.hits} results from the checkpoint, {plr.CKPT.misses} newly computed")
        plr.CKPT.finish(delete=True)  # the run is finished and published: its checkpoint is not needed any more
        rows.append(row)
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(RESULT_DIR / f"summary_seeds_{SUFFIX}.csv", index=False)
        plr.aggregate(seed_rows).to_csv(RESULT_DIR / f"summary_{SUFFIX}.csv", index=False)

    if rows:
        summary = plr.aggregate(pd.DataFrame(rows))
        print("\n=== weighted F1 mean +- std over seeds ===")
        for _, r in summary.iterrows():
            cells = "  ".join(f"{who} {r[f'{who}_f1_weighted_mean']:.3f}+-{r[f'{who}_f1_weighted_std']:.3f}"
                              for who in WHO if f"{who}_f1_weighted_mean" in r)
            print(f"{r['dataset']} {int(r['percent']):>3}% (n={r['n_seeds']}): {cells}")
    if failures:
        print(f"\n{len(failures)} runs failed: {RESULT_DIR / f'failures_{SUFFIX}.csv'}")
    print(f"\nSaved in {RESULT_DIR}: summary_{SUFFIX}.csv, summary_seeds_{SUFFIX}.csv and one <test file>_<p>_{SUFFIX}.csv/.json per run")


if __name__ == "__main__":
    main()
