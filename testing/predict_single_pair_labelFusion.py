"""LabelFusion on every train/test pair, grouped by dataset; metrics averaged over the seeds.

Every data/test_data/<dataset>-<seed>.xlsx (e.g. lab-manual-mm-test-5768) is paired with its
data/training_data/<dataset with -train->-<seed>.xlsx and run once with standard (non-tuned)
hyperparameters. The pairs of one dataset differ only in the seed of the resampling, so per
dataset the accuracy / macro-F1 of the LLM, RoBERTa and the fusion are averaged over the seeds,
with the standard deviation (sample std, ddof=1, i.e. the usual "mean +- std over N runs").

Reuses the building blocks (PrecomputedLLM, EmbeddingMemo, fit_predict, default_params, ...)
from predict_all_files_labelFusion.py so behaviour matches the --trials 0 case of that script.

With --trials > 0, Optuna tunes ONLY the two epoch counts (RoBERTa num_epochs, fusion MLP
num_epochs) -- every other hyperparameter stays at its default. This keeps the search space
small (2 dimensions instead of 9) so a handful of trials can actually cover it, which matters
on the small files here: a single held-out eval split of ~10% of the train file is noisy, and
tuning 9 hyperparameters against it (as predict_all_files_labelFusion.py's --trials does) risks
picking parameters that just got lucky on that split rather than epoch counts that generalize.
Tuning runs separately for every pair, on that pair's own train file: the validation half is
itself split in two, one half to fit the fusion MLP, the other to score each trial (macro-F1);
the test file is never touched by tuning.

Writes to ./outputs/fusion/single/:
    <test file name>.csv / .json   per pair: sentence, label, true, llm_pred, roberta_pred, fusion_pred
    summary_seeds.csv              one row per pair (dataset, seed, metrics, chosen epochs)
    summary.csv                    one row per dataset: <metric>_mean / <metric>_std over its seeds
Both summaries are rewritten after every pair, so a stopped run keeps its progress.

    python testing/predict_single_pair_labelFusion.py                                   # every dataset, every seed
    python testing/predict_single_pair_labelFusion.py --datasets mm-test pc-split-test  # datasets containing these
    python testing/predict_single_pair_labelFusion.py --datasets pc-test --limit-train 60 --epochs 1  # smoke test
    python testing/predict_single_pair_labelFusion.py --trials 8 --joint-training       # tune epochs, joint training
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

from predict_all_files_labelFusion import (  # noqa: E402
    FUSION_DIR, LLM_TEST_DIR, LLM_TRAIN_DIR, TEST_DIR, TEXT_COLUMN, LABEL_MAP, RANDOM_STATE,
    PrecomputedLLM, default_params, fit_predict, llm_predictions, load, paired_train, scores, split_pair,
)

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

OUT_DIR = FUSION_DIR / "single"
SEED_RE = re.compile(r"^(?P<dataset>.+)-(?P<seed>\d+)$")
METRICS = [f"{who}_{m}" for who in ("llm", "roberta", "fusion") for m in ("accuracy", "f1_macro")]


def tune_epochs(train_df, val_df, llm, name, args) -> tuple[dict, dict]:
    """Optuna study over roberta_epochs and fusion_epochs only; everything else is default_params().
    Mirrors predict_all_files_labelFusion.py's tune(): the val split is halved, fusion MLP fits on
    one half, every trial is scored (macro-F1) on the other. The test file is never used here."""
    try:
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE, stratify=val_df["label"])
    except ValueError:  # a class too small to stratify
        fit_df, eval_df = train_test_split(val_df, train_size=0.5, random_state=RANDOM_STATE)
    fit_df, eval_df = fit_df.reset_index(drop=True), eval_df.reset_index(drop=True)
    eval_llm, eval_true = llm.lookup(eval_df), eval_df["label"].map(LABEL_MAP).tolist()
    print(f"### epoch tuning on {name}: fit {len(fit_df)} / eval {len(eval_df)}")

    def objective(trial: optuna.Trial) -> float:
        rob, fus = default_params(args)
        rob["num_epochs"] = trial.suggest_int("roberta_epochs", 10, 100, step=10)
        fus["num_epochs"] = trial.suggest_int("fusion_epochs", 0, 20)
        fus["joint_training"] = args.joint_training
        preds, _ = fit_predict(rob, fus, train_df, fit_df, eval_df, eval_llm, llm, name,
                               f"trial{trial.number}", save=False, verbose=False)
        return f1_score(eval_true, preds, average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    d_rob, d_fus = default_params(args)
    # Clamp the enqueued starting point into the tuned ranges (roberta 10-100 step 10, fusion 0-20).
    study.enqueue_trial({
        "roberta_epochs": min(max(round(d_rob["num_epochs"] / 10) * 10, 10), 100),
        "fusion_epochs": min(max(d_fus["num_epochs"], 0), 20),
    })
    study.optimize(objective, n_trials=args.trials)

    rob, fus = default_params(args)
    rob["num_epochs"] = study.best_trial.params["roberta_epochs"]
    fus["num_epochs"] = study.best_trial.params["fusion_epochs"]
    print(f"### best macro-F1 {study.best_value:.3f} after {len(study.trials)} trials: "
          f"roberta_epochs={rob['num_epochs']} fusion_epochs={fus['num_epochs']}")
    return rob, fus


def run_pair(train_path: Path, test_path: Path, args) -> dict:
    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    llm_test = llm_predictions(LLM_TEST_DIR / f"{test_path.stem}.csv")
    llm_test_preds = [llm_test[t] for t in test_df[TEXT_COLUMN]]

    if args.trials > 0:
        rob, fus = tune_epochs(train_df, val_df, llm, name, args)
    else:
        rob, fus = default_params(args)
    fus["joint_training"] = args.joint_training
    fusion_preds, roberta_preds = fit_predict(
        rob, fus, train_df, val_df, test_df, llm_test_preds, llm, name, "single", save=True, verbose=True
    )

    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "llm_pred": llm_test_preds,
        "roberta_pred": roberta_preds,
        "fusion_pred": fusion_preds,
    })
    shutil.rmtree(FUSION_DIR / "cache" / "predictions", ignore_errors=True)
    out.to_csv(OUT_DIR / f"{name}.csv", index=False)
    out.to_json(OUT_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)

    m = SEED_RE.match(name)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "rows": len(out),
           "seconds": round(time.time() - t0)}
    for who in ("llm", "roberta", "fusion"):
        row.update({f"{who}_{k}": v for k, v in scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row["roberta_epochs"], row["fusion_epochs"] = rob["num_epochs"], fus["num_epochs"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only datasets (test file name without seed) containing one of these")
    parser.add_argument("--epochs", type=int, default=3, help="RoBERTa epochs of the default configuration")
    parser.add_argument("--fusion-epochs", type=int, default=30, help="fusion MLP epochs of the default configuration")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    parser.add_argument("--trials", type=int, default=0, help="Optuna trials tuning only the epoch counts (0 = defaults)")
    parser.add_argument("--joint-training", action="store_true",
                         help="fine-tune RoBERTa jointly with the fusion MLP (gradients flow through both, "
                              "two param groups at ml_lr/fusion_lr) instead of freezing RoBERTa after its "
                              "own separate training stage")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(FUSION_DIR)  # FusionEnsemble writes its prediction caches to ./cache

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
        rows.append(run_pair(train_path, test_path, args))
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(OUT_DIR / "summary_seeds.csv", index=False)
        aggregate(seed_rows).to_csv(OUT_DIR / "summary.csv", index=False)

    summary = aggregate(pd.DataFrame(rows))
    print("\n=== mean +- std over seeds ===")
    for _, r in summary.iterrows():
        cells = "  ".join(f"{metric} {r[f'{metric}_mean']:.3f}+-{r[f'{metric}_std']:.3f}" for metric in METRICS)
        print(f"{r['dataset']} (n={r['n_seeds']}): {cells}")
    print(f"\nSaved: {OUT_DIR / 'summary.csv'} and {OUT_DIR / 'summary_seeds.csv'}")


if __name__ == "__main__":
    main()
