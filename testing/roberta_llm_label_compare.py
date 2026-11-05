"""Do roberta-large and the LLM complement each other, or do they get the SAME sentences wrong?

For every test file and every training-data regime p = 10, 20, ..., 100 % (the same runs, regimes and LLM caches as
predict_low_data_regimes.py) roberta-large is FINE-TUNED ALONE on the regime, exactly like the column "roberta" of the full
program there (80/20 train/validation split of the regime, Optuna over learning rate x batch size x epochs, class weights, warmup,
best epoch, retries on a one-class collapse), and it labels the full test set. Its labels are stored next to the labels the LLM
gave the same sentences (read from outputs/llm_cache, no LLM is called; few-shot, 2 examples per 10 % of the training data).
Nothing else is trained: no fusion MLP, no time series.

Per run and sentence the outcome is one of
    both_correct       roberta and LLM are right
    only_roberta       only roberta is right   (the LLM could learn from roberta)
    only_llm           only the LLM is right   (roberta could learn from the LLM)
    both_wrong_same    both are wrong and say the SAME wrong class   (no combination of the two can fix this)
    both_wrong_diff    both are wrong, with different wrong classes

How to read the summary (per dataset and regime: mean / std over the seeds; and over everything per regime):
    agreement, kappa        how often / how much beyond chance the two say the same thing
    error_overlap           both_wrong / (sentences where at least one is wrong): 1 = they always fail together, 0 = never
    error_phi               correlation of the two error indicators: ~0 = independent errors, > 0 = errors on the same sentences
    both_wrong_excess       both_wrong minus what two INDEPENDENT classifiers with these accuracies would have in common
                            (> 0: they fail on the same sentences more often than chance, < 0: they complement each other)
    oracle_accuracy         accuracy of a perfect chooser between the two = the ceiling of any fusion of these two labels
    rescued_by_llm          share of roberta's errors the LLM gets right (rescued_by_roberta: the other way round)
A fusion can only help where the oracle accuracy is clearly above the better one of the two accuracies.

Resumable like predict_low_data_regimes.py (checkpoints/: every Optuna trial that was done is read from disk, finished runs are
skipped; only the final fine-tuning of an interrupted run is repeated). Output: ./outputs/roberta_llm_label_compare/
    <test file>_<p>_roberta_llm_compare_<date_time>_<job id>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred,
                                                                          llm_correct, roberta_correct, outcome, agree
    summary_seeds_roberta_llm_compare_<date_time>_<job id>.csv     one row per run (metrics, counts, overlap measures, parameters)
    summary_roberta_llm_compare_<date_time>_<job id>.csv           per dataset and regime: mean / std (ddof=1) over the seeds
    summary_by_percent_roberta_llm_compare_<date_time>_<job id>.csv  per regime, over all datasets and seeds
    summary_by_class_roberta_llm_compare_<date_time>_<job id>.csv  outcome counts per run and true class
    failures_roberta_llm_compare_<date_time>_<job id>.csv, trials/...

    python testing/roberta_llm_label_compare.py --dry-run
    python testing/roberta_llm_label_compare.py --datasets pc-split-test --percents 10 100 --trials 0 --epochs 2   # smoke test
    python testing/roberta_llm_label_compare.py                                                                  # everything
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
from sklearn.metrics import cohen_kappa_score  # noqa: E402

base = plr.base
stage1 = plr.stage1
RESULT_DIR = TESTING_DIR.parent / "outputs" / "roberta_llm_label_compare"
WORK_DIR = RESULT_DIR / ".work" / plr.RUN_ID
SUFFIX = f"roberta_llm_compare_{datetime.now():%Y%m%d_%H%M%S}_{plr.RUN_ID}"  # start time + job id: parallel jobs never share a file name
WHO = ("llm", "roberta")
METRICS = [f"{who}_{m}" for who in WHO for m in ("f1_weighted", "f1_macro", "accuracy")]
OUTCOMES = ("both_correct", "only_roberta", "only_llm", "both_wrong_same", "both_wrong_diff")
OVERLAP = ["agreement", "kappa", "error_overlap", "error_phi", "both_wrong_excess", "oracle_accuracy",
           "rescued_by_llm", "rescued_by_roberta"]
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)

# predict_low_data_regimes.py's helpers read these module globals when they are called: point them at this script's folders
plr.RESULT_DIR, plr.WORK_DIR, plr.SUFFIX, plr.CHECKPOINT_DIR = RESULT_DIR, WORK_DIR, SUFFIX, RESULT_DIR / "checkpoints"
plr.WHO, plr.METRICS = WHO, METRICS


def outcomes(true: pd.Series, llm: pd.Series, roberta: pd.Series) -> pd.Series:
    """The outcome of every sentence (see OUTCOMES)."""
    lc, rc = llm == true, roberta == true
    out = pd.Series("both_wrong_diff", index=true.index)
    out[~lc & ~rc & (llm == roberta)] = "both_wrong_same"
    out[lc & ~rc] = "only_llm"
    out[~lc & rc] = "only_roberta"
    out[lc & rc] = "both_correct"
    return out


def overlap_measures(out: pd.DataFrame) -> dict:
    """Counts and overlap measures of one run (llm_pred vs roberta_pred on the full test set)."""
    n = len(out)
    lc, rc = (out["llm_pred"] == out["true"]).to_numpy(), (out["roberta_pred"] == out["true"]).to_numpy()
    counts = out["outcome"].value_counts()
    row = {f"n_{o}": int(counts.get(o, 0)) for o in OUTCOMES}
    both_wrong = row["n_both_wrong_same"] + row["n_both_wrong_diff"]
    either_wrong = n - row["n_both_correct"]
    llm_errors, roberta_errors = int((~lc).sum()), int((~rc).sum())
    row["agreement"] = float((out["llm_pred"] == out["roberta_pred"]).mean())
    try:
        row["kappa"] = float(cohen_kappa_score(out["llm_pred"], out["roberta_pred"]))
    except ValueError:
        row["kappa"] = float("nan")
    row["error_overlap"] = both_wrong / either_wrong if either_wrong else float("nan")
    row["error_phi"] = float(np.corrcoef(~lc, ~rc)[0, 1]) if 0 < llm_errors < n and 0 < roberta_errors < n else float("nan")
    row["both_wrong_excess"] = both_wrong - llm_errors * roberta_errors / n  # expected common errors if independent
    row["oracle_accuracy"] = (n - both_wrong) / n
    row["rescued_by_llm"] = row["n_only_llm"] / roberta_errors if roberta_errors else float("nan")
    row["rescued_by_roberta"] = row["n_only_roberta"] / llm_errors if llm_errors else float("nan")
    return row


def class_rows(out: pd.DataFrame, row: dict) -> list[dict]:
    """Outcome counts per true class of one run."""
    records = []
    for cls, group in out.groupby("true"):
        counts = group["outcome"].value_counts()
        records.append({"dataset": row["dataset"], "seed": row["seed"], "percent": row["percent"], "true": cls, "n": len(group),
                        **{f"n_{o}": int(counts.get(o, 0)) for o in OUTCOMES},
                        "llm_recall": float((group["llm_pred"] == cls).mean()), "roberta_recall": float((group["roberta_pred"] == cls).mean())})
    return records


def aggregate(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """mean / std (ddof=1) over the seeds (and datasets, if they are not in `by`) of every metric and overlap measure."""
    columns = [c for c in METRICS + OVERLAP + [f"n_{o}" for o in OUTCOMES] if c in rows]
    grouped = rows.groupby(by)
    summary = pd.DataFrame({"n_runs": grouped.size()})
    for column in columns:
        summary[f"{column}_mean"] = grouped[column].mean()
        summary[f"{column}_std"] = grouped[column].std(ddof=1)
    return summary.reset_index()


def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    name, regime_df, test_df, _llm_train, _llm_test_preds, out, row = plr.prepare(test_path, train_path, llm_path, percent)
    train_df, val_df = plr.split_regime(regime_df)
    print(f"\n=== {name}: regime {len(regime_df)} (train {len(train_df)} / val {len(val_df)}) / test {len(test_df)} ===")
    row.update({"training": "finetuned", "train_rows": len(train_df), "val_rows": len(val_df)})

    # roberta-large alone, as stage 1 of predict_low_data_regimes.py: Optuna over lr x batch size x epochs, then the final fine-tuning
    if args.trials > 0:
        params, val_f1 = stage1.tune(train_df, val_df, name, args)
    else:
        params, val_f1 = stage1.default_params(args), None
    roberta, memo = base.train_roberta(params, train_df, val_df, retries=args.retries)
    out["roberta_pred"] = roberta.predict_without_saving(test_df).predictions
    row.update({"roberta_lr": params["learning_rate"], "roberta_batch_size": params["batch_size"],
                "roberta_epochs": params["num_epochs"], "roberta_val_f1": val_f1})
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out["llm_correct"] = out["llm_pred"] == out["true"]
    out["roberta_correct"] = out["roberta_pred"] == out["true"]
    out["agree"] = out["llm_pred"] == out["roberta_pred"]
    out["outcome"] = outcomes(out["true"], out["llm_pred"], out["roberta_pred"])
    row.update(overlap_measures(out))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()
           if k.startswith("n_") or k in ("agreement", "kappa", "error_overlap", "oracle_accuracy")})
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
        done = {re.sub(r"_roberta_llm_compare_\d{8}_\d{6}(_\w+)?$", "", p.stem)
                for p in RESULT_DIR.glob("lab-manual-*_roberta_llm_compare_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs ({', '.join(sorted({str(j[3]) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j[0] for j in todo})} test files; roberta-large fine-tuned alone with {args.trials} Optuna trials per run, "
          f"compared with the LLM labels")
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
    rows, per_class, failures = [], [], []
    for test_path, train_path, llm_path, percent in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name = f"{test_path.stem}_{percent}"
        plr.CKPT.start(name, {"regime": hashlib.md5(train_path.read_bytes()).hexdigest(), "test": test_path.name, "percent": percent,
                              "epochs": args.epochs, "llm_model": plr.LLM_MODEL, "kind": "roberta_llm_compare", "v": 1})
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
        per_class += class_rows(out, row)
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(RESULT_DIR / f"summary_seeds_{SUFFIX}.csv", index=False)
        aggregate(seed_rows, ["dataset", "percent"]).to_csv(RESULT_DIR / f"summary_{SUFFIX}.csv", index=False)
        aggregate(seed_rows, ["percent"]).to_csv(RESULT_DIR / f"summary_by_percent_{SUFFIX}.csv", index=False)
        pd.DataFrame(per_class).to_csv(RESULT_DIR / f"summary_by_class_{SUFFIX}.csv", index=False)

    if rows:
        summary = aggregate(pd.DataFrame(rows), ["percent"])
        print("\n=== over all datasets and seeds, per regime ===")
        for _, r in summary.iterrows():
            print(f"{int(r['percent']):>3}% (n={int(r['n_runs'])}): roberta acc {r['roberta_accuracy_mean']:.3f}  llm acc {r['llm_accuracy_mean']:.3f}  "
                  f"oracle {r['oracle_accuracy_mean']:.3f}  agreement {r['agreement_mean']:.3f}  error_overlap {r['error_overlap_mean']:.3f}  "
                  f"error_phi {r['error_phi_mean']:.3f}")
    if failures:
        print(f"\n{len(failures)} runs failed: {RESULT_DIR / f'failures_{SUFFIX}.csv'}")
    print(f"\nSaved in {RESULT_DIR}: summary_{SUFFIX}.csv, summary_by_percent_{SUFFIX}.csv, summary_by_class_{SUFFIX}.csv, "
          f"summary_seeds_{SUFFIX}.csv and one <test file>_<p>_{SUFFIX}.csv/.json per run")


if __name__ == "__main__":
    main()
