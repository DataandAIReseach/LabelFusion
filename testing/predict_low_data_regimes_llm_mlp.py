"""Check of the MLP training: the LLM's label + a downstream MLP, and NOTHING else is trained.

predict_low_data_regimes.py compares roberta-large, the LLM and the fusion models. If the fusion models are worse than the LLM
alone, the cause could be roberta-large / the time series, or the training of the fusion MLP itself. This script rules the
first out: there is no roberta-large, no TimesFM, no fine-tuning of anything. Per test file and regime p (the same runs, the
same regimes, the same LLM caches as predict_low_data_regimes.py) it makes the predictions with the LLM label and a fusion
MLP behind it, and the MLP is the ONLY thing that learns:

  llm            the LLM labels of the regime on the full test set (few-shot, 2 examples per 10 %): the reference. Not trained.
  llm_lookup     the best a function of the LLM label alone can do on the training data, without any training procedure: every
                 LLM label (dovish / hawkish / neutral) is mapped to the most frequent TRUE class among the regime's sentences
                 that got that label (zero-shot labels). Cannot be hurt by an optimisation problem.
  llm_mlp_all    the project's fusion MLP (textclassify's FusionEnsemble / FusionWrapper / FusionMLP, the same Optuna search over
                 learning rate, epochs, hidden layers and batch size as in the other scripts) with the LLM one-hot as its ONLY
                 information: roberta-large is replaced by a constant zero embedding. Trained on the WHOLE regime.
  llm_mlp_80     the same, trained on the train split (80 % of the regime): the data the end-to-end models learn from.
  llm_mlp_20     the same, trained on the validation split (20 % of the regime): the data the two-stage fusion MLPs learned from.

How to read it: a function of the LLM label alone cannot be much better than llm_lookup, and a correctly trained MLP should end up
close to llm_lookup (and close to llm). If llm_mlp_* is clearly WORSE than llm / llm_lookup, the MLP training (optimisation, search
space, small training sets) is the problem; if it matches them, the training is fine and the weak fusion results come from the
roberta-large / time series side. Comparing llm_mlp_80 / _20 / _all shows how much the amount of training data matters.
One difference to keep in mind: the MLP learns on the ZERO-SHOT labels of the training sentences, but is applied to the
regime's FEW-SHOT labels of the test sentences (as in all fusion runs); llm_lookup has the same mismatch.

The MLP is tuned per run and setting with Optuna (--mlp-trials, default 30; its validation is half of the training data, as in
predict_all_labelFusion.py). Like predict_low_data_regimes.py it reads the LLM labels from outputs/llm_cache (nothing is asked
again), is resumable (checkpoints/, finished runs are skipped) and writes to ./outputs/predict_low_data_regimes_llm_mlp/:
    <test file>_<p>_llm_mlp_<date_time>.csv / .json   per run: sentence, label, true, llm_pred, llm_lookup_pred, llm_mlp_all_pred, ...
    summary_seeds_llm_mlp_<date_time>.csv             one row per run (all metrics, MLP parameters)
    summary_llm_mlp_<date_time>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_llm_mlp_<date_time>.csv, trials/...

    python testing/predict_low_data_regimes_llm_mlp.py --dry-run
    python testing/predict_low_data_regimes_llm_mlp.py --datasets pc-split-test --percents 10 70
    python testing/predict_low_data_regimes_llm_mlp.py                                            # everything
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from textclassify.core.types import ClassificationResult, ClassificationType  # noqa: E402

base = plr.base
RESULT_DIR = TESTING_DIR.parent / "outputs" / "predict_low_data_regimes_llm_mlp"
WORK_DIR = RESULT_DIR / ".work" / plr.RUN_ID
SUFFIX = f"llm_mlp_{datetime.now():%Y%m%d_%H%M%S}"
WHO = ("llm", "llm_lookup", "llm_mlp_all", "llm_mlp_80", "llm_mlp_20")
METRICS = [f"{who}_{m}" for who in WHO for m in ("f1_weighted", "f1_macro", "accuracy")]
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)

# predict_low_data_regimes.py's helpers read these module globals when they are called: point them at this script's folders
plr.RESULT_DIR, plr.WORK_DIR, plr.SUFFIX, plr.CHECKPOINT_DIR = RESULT_DIR, WORK_DIR, SUFFIX, RESULT_DIR / "checkpoints"
plr.WHO, plr.METRICS = WHO, METRICS


class ZeroEmbeddingML:
    """Stands in for roberta-large in FusionEnsemble: no encoder, no training. Its 'embedding' is a constant zero vector, so the
    fusion MLP (and everything around it in the library) is exactly the one of the other runs, but the LLM one-hot is its only
    source of information. Provides what FusionEnsemble asks of an ML model."""

    def __init__(self, dim: int = 1):
        self.text_column = base.TEXT_COLUMN
        self.label_columns = base.LABEL_COLUMNS
        self.classes_ = base.LABEL_COLUMNS
        self.multi_label = False
        self.is_trained = True
        self.model_name = "zero-embedding"
        self.auto_save_path = None
        self.embedding_dim = dim
        self.model = nn.Linear(1, 1)  # FusionEnsemble only touches ml_model.model.parameters() (it never trains them here)

    def _result(self, df: pd.DataFrame) -> ClassificationResult:
        n = len(df)
        return ClassificationResult(
            predictions=[base.LABEL_COLUMNS[-1]] * n,  # meaningless: there is no classifier, only the constant embedding
            embeddings=[[0.0] * self.embedding_dim for _ in range(n)],
            model_name="zero_embedding",
            classification_type=ClassificationType.MULTI_CLASS,
        )

    def predict_without_saving(self, df: pd.DataFrame, mode: str = "test") -> ClassificationResult:
        return self._result(df)

    def predict(self, df: pd.DataFrame, mode: str = "test") -> ClassificationResult:
        return self._result(df)


def lookup_predictions(regime_labels: list[str], regime_true: list[str], test_labels: list[str]) -> list[str]:
    """LLM label -> the most frequent true class among the regime's sentences with that LLM label."""
    table = pd.DataFrame({"llm": regime_labels, "true": regime_true})
    mapping = {llm_label: group["true"].value_counts().idxmax() for llm_label, group in table.groupby("llm")}
    return [mapping.get(label, label) for label in test_labels]  # a label the regime never produced stays as it is


def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args, ml) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = plr.prepare(test_path, train_path, llm_path, percent)
    train_df, val_df = plr.split_regime(regime_df)
    print(f"\n=== {name}: regime {len(regime_df)} (train {len(train_df)} / val {len(val_df)}) / test {len(test_df)} ===")
    row.update({"train_rows": len(train_df), "val_rows": len(val_df)})

    regime_true = regime_df["label"].map(base.LABEL_MAP).tolist()
    out["llm_lookup_pred"] = lookup_predictions(llm_train.lookup(regime_df), regime_true, llm_test_preds)

    for who, data in (("llm_mlp_all", regime_df), ("llm_mlp_80", train_df), ("llm_mlp_20", val_df)):
        fname = f"{name}_{who}"
        try:
            if args.mlp_trials > 0:
                mlp, mlp_val_f1 = base.tune_mlp(ml, llm_train, data, data, fname, args)
            else:
                mlp, mlp_val_f1 = base.default_mlp_params(args), None
            out[f"{who}_pred"] = base.mlp_fit_predict(ml, llm_train, mlp, data, data, test_df, llm_test_preds, fname, "final", verbose=False)
            row.update({f"{who}_mlp_lr": mlp["fusion_lr"], f"{who}_mlp_epochs": mlp["num_epochs"],
                        f"{who}_mlp_batch_size": mlp["batch_size"], f"{who}_mlp_val_f1": mlp_val_f1, f"{who}_train_rows": len(data)})
        except Exception as e:  # e.g. a training set too small for the library's hold-out: the other settings still count
            traceback.print_exc()
            row[f"{who}_error"] = f"{type(e).__name__}: {e}"
        gc.collect()
    return plr.finish(row, out, t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--llm-cache-dir", type=Path, default=plr.LLM_CACHE_DIR, help="the LLM caches (default: outputs/llm_cache)")
    parser.add_argument("--llm-model", default=plr.LLM_MODEL, help="the LLM whose cache folders are read (e.g. gpt-5-nano)")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="Optuna trials per run and training-data setting tuning the MLP; 0 = the MLP's default configuration")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip runs that already have a result file in the output folder (default; --no-resume repeats them)")
    args = parser.parse_args()
    args.ml_proj_dim = 0  # the constant zero embedding is not projected (the other scripts project roberta-large's 1024 dims to 768)
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
        done = {re.sub(r"_llm_mlp_\d{8}_\d{6}$", "", p.stem) for p in RESULT_DIR.glob("lab-manual-*_llm_mlp_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs on {len({j[0] for j in todo})} test files; models: llm, llm_lookup, llm_mlp_all / _80 / _20 "
          f"(only the MLP is trained, {args.mlp_trials} Optuna trials each)")
    if args.dry_run or not todo:
        for test_path, train_path, llm_path, percent in todo[:10]:
            print(f"  {test_path.stem}_{percent}: regime {train_path.name}, LLM cache {llm_path.name}")
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    os.chdir(WORK_DIR)  # the library writes its prediction caches to ./cache: this run's own folder
    staging = WORK_DIR / "out"
    staging.mkdir()
    base.RESULT_DIR = plr.files.FUSION_DIR = staging

    ml = ZeroEmbeddingML()
    plr.install_checkpoint_wrappers()
    rows, failures = [], []
    for test_path, train_path, llm_path, percent in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name = f"{test_path.stem}_{percent}"
        plr.CKPT.start(name, {"regime": hashlib.md5(train_path.read_bytes()).hexdigest(), "test": test_path.name, "percent": percent,
                              "llm_model": plr.LLM_MODEL, "kind": "llm_mlp", "v": 1})
        try:
            row, out = run_job(test_path, train_path, llm_path, percent, args, ml)
        except Exception as e:  # one failing run must not stop the sweep
            traceback.print_exc()
            plr.CKPT.finish(delete=False)
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
        plr.CKPT.finish(delete=True)
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
