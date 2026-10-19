"""Low-data regimes with FROZEN encoders: only the fusion MLP is trained (and tuned with Optuna).

For every test file (data/test_data_dated/<dataset>-<seed>.xlsx, always the FULL test set) and every regime p
(10, 20, ..., 100 %) of its training file (data/training_data_dated_low_data_regimes/<train name>_<p>.csv, made by
dataset_scripts/make_low_data_regimes.py) one run trains a small MLP on the regime and predicts the test set. Three
MLPs differ only in their input, four columns of predictions come out:

  llm           the LLM labels of the regime: GPT-5 nano with 2 examples per 10 % of the training data
                (predict_all_files_llm_low_data_regimes.py, outputs/predictions_low_data_regimes/...). These are
                also the LLM labels the fusion models get for the test sentences.
  roberta       MLP on the frozen roberta-large [CLS] embedding alone (the LLM input is a constant, i.e. no information).
  fusion        LabelFusion: MLP on the frozen roberta-large embedding + the LLM label. The LLM labels of the regime's
                sentences are the zero-shot ones of predict_all_files_llm.py (outputs/predictions/train_gpt-5-nano_zero_shot).
  fusion_ts     the same with the time series branch: DGS10 / T10Y2Y / VIXCLS (data/3 Ts/daily_series.csv), the last
                --ts-window days up to the date of the sentence, one frozen TimesFM encoder per series, pooled over the
                series (--ts-pooling) and concatenated to the MLP input. Skipped with --no-ts.

Nothing but the MLP is trained: roberta-large is the pretrained model (never fine-tuned), the TimesFM encoders and the
time series transformer are frozen. The MLP is therefore trained on the WHOLE regime (no train/validation split is
needed to protect a fine-tuned encoder) and scored on the full test set: weighted F1 (primary), macro-F1, accuracy.
Optuna (--mlp-trials, default 30, 0 = the MLP's default configuration) tunes the MLP's learning rate, epochs, hidden
layers and batch size per run and model, on one half of the regime scored on the other half, as in
predict_all_labelFusion.py; the best parameters then train the MLP on the whole regime. Because the encoders are frozen,
every sentence is embedded once for the whole sweep (roberta-large and TimesFM), so a run costs only MLP trainings.

Several runs (e.g. one sbatch job per group of datasets) can run at the same time without influencing each other:
every run has its own work folder (.work/<job id>) for the library's prediction caches, and every result file carries
the start time of the run in its name. Writes to ./outputs/predict_low_data_regimes/:
    <test file>_<p>_low_data_regimes_<date_time>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred, fusion_pred, fusion_ts_pred
    summary_seeds_low_data_regimes_<date_time>.csv             one row per run (all metrics, MLP parameters used)
    summary_low_data_regimes_<date_time>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_low_data_regimes_<date_time>.csv                  runs that raised an error (the sweep goes on)
    trials/...                                                 Optuna trials of the MLPs

    python testing/predict_low_data_regimes.py --dry-run                                  # which runs, nothing is trained
    python testing/predict_low_data_regimes.py --datasets pc-split-test --percents 10 100 # a first look
    python testing/predict_low_data_regimes.py                                            # everything
    python testing/predict_low_data_regimes.py --no-ts --mlp-trials 10                    # without the time series, fewer trials
"""

from __future__ import annotations

import argparse
import atexit
import collections
import gc
import hashlib
import os
import re
import shutil
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

# Everything this script writes lives in RESULT_DIR. Every run gets its own work folder (.work/<job id>) there for the
# saved roberta-large models and the library's prediction caches (the library names them after the train texts, so two
# runs on the same data would otherwise write and delete the same directory); it is removed when the run ends.
RESULT_DIR = TESTING_DIR.parent / "outputs" / "predict_low_data_regimes"
RUN_ID = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
WORK_DIR = RESULT_DIR / ".work" / RUN_ID
os.environ["LABELFUSION_MODEL_CACHE"] = str(WORK_DIR / "models")  # set before the modules below read it
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)
SUFFIX = f"low_data_regimes_{datetime.now():%Y%m%d_%H%M%S}"

import predict_all_files_labelFusion as files  # noqa: E402
import predict_all_labelFusion as base  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from test_textclassify_llm import _add_onehot_labels  # noqa: E402
from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402
from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder  # noqa: E402

REPO_ROOT = base.TEST_DIR.parent.parent
TRAIN_REGIME_DIR = REPO_ROOT / "data" / "training_data_dated_low_data_regimes"
TEST_DATED_DIR = REPO_ROOT / "data" / "test_data_dated"
LLM_REGIME_DIR = base.OUT_DIR / "predictions_low_data_regimes" / "test_gpt-5-nano_2_per_10_percent"
TS_FILE = REPO_ROOT / "data" / "3 Ts" / "daily_series.csv"
TS_SERIES = ["DGS10", "T10Y2Y", "VIXCLS"]
DATE_COLUMN = "date"
FULL_HISTORY = 10**6  # window_days that reaches back to the beginning of every series
WHO = ("llm", "roberta", "fusion", "fusion_ts")
METRICS = [f"{who}_{m}" for who in WHO for m in ("f1_weighted", "f1_macro", "accuracy")]
REGIME_RE = re.compile(r"_(?P<percent>\d+)$")


# ---- time series branch (as in predict_all_labelFusion_TS_fixed_window.py) -------------------------------------------

class DailySeries(TS):
    """textclassify's TS loader on one CSV with a DATE column and one column per series."""

    def __init__(self, csv_path: Path, series_names: list[str]):
        super().__init__(data_dir=csv_path.parent, stock_symbols=list(series_names), date_column="DATE", price_column="value")
        self.csv_path = csv_path

    def load_all(self) -> None:
        raw = pd.read_csv(self.csv_path, parse_dates=["DATE"]).sort_values("DATE")
        for name in self.stock_symbols:
            series = raw[["DATE", name]].rename(columns={name: "value"})
            series["value"] = series["value"].ffill()  # holidays: the last known value
            self.data[name] = series.dropna().reset_index(drop=True)


class CachedEmbedder(nn.Module):
    """A FROZEN TSEmbedder that remembers its embedding per input series, shared by all copies of the encoder and by
    all runs of the sweep (the same window always gives the same embedding)."""

    def __init__(self, inner: TSEmbedder, cache: dict):
        super().__init__()
        self.inner = inner
        self.cache = cache

    def forward(self, past_values, **kwargs):
        keys = [hashlib.blake2b(np.asarray(v, dtype=np.float32).tobytes(), digest_size=16).digest() for v in past_values]
        todo = {k: v for k, v in zip(keys, past_values) if k not in self.cache}
        if todo:
            with torch.no_grad():
                embeddings = self.inner(list(todo.values()))["embeddings"].detach().cpu()
            self.cache.update(zip(todo, embeddings))
        device = next(self.inner.parameters()).device
        return {"embeddings": torch.stack([self.cache[k] for k in keys]).to(device)}


class TSBranch:
    """Loads the series and the pretrained TimesFM encoders once. Everything is frozen: the encoders (their embeddings
    are cached for the whole sweep) and the CrossTSTransformer, which FusionEnsemble.add_ts_model needs but which the
    pooled branch never calls."""

    def __init__(self, args):
        self.loader = DailySeries(TS_FILE, TS_SERIES)
        self.loader.load_all()
        encoders = {name: TSEmbedder(pooling="mean") for name in TS_SERIES}
        for enc in encoders.values():
            enc.requires_grad_(False)
        self.hidden_size = next(iter(encoders.values())).hidden_size
        self.caches = {name: {} for name in TS_SERIES}
        self.encoders = {name: CachedEmbedder(enc, self.caches[name]) for name, enc in encoders.items()}

    def fresh(self):
        torch.manual_seed(base.RANDOM_STATE)
        transformer = CrossTSTransformer(series_names=TS_SERIES, hidden_size=self.hidden_size, output_dim=self.hidden_size)
        transformer.requires_grad_(False)
        return self.loader, self.encoders, transformer


def run_pooled_ts_branch(self, dates):
    """FusionEnsemble._run_ts_branch without the CrossTSTransformer: the per-series embeddings are only pooled over
    the series (mean or max) and passed on as they are."""
    tr = self.ts_transformer
    batch = self.ts_loader.get_series_batch(dates, window_days=self.ts_window_days, stocks=tr.series_names)
    tokens = torch.stack(
        [self.ts_embedders[name](batch[name])["embeddings"].to(self.device) for name in tr.series_names], dim=1
    )  # (B, n_series, hidden_size)
    return tokens.mean(dim=1) if self.ts_pooling == "mean" else tokens.max(dim=1).values


# ---- data -------------------------------------------------------------------------------------------------------------

def with_dates(raw: pd.DataFrame) -> pd.DataFrame:
    """sentence + label (+ one-hot) as the other scripts use them, plus the date of every sentence."""
    out = _add_onehot_labels(raw[[base.TEXT_COLUMN, "label"]].copy()).reset_index(drop=True)
    out[DATE_COLUMN] = pd.to_datetime(raw[DATE_COLUMN]).dt.strftime("%Y-%m-%d").values
    return out


def load_regime(path: Path) -> pd.DataFrame:
    return with_dates(pd.read_csv(path))


def load_test(test_path: Path) -> pd.DataFrame:
    return with_dates(pd.read_excel(TEST_DATED_DIR / test_path.name))


def jobs(args) -> list[tuple[Path, Path, Path, int]]:
    """(test file, train regime file, the regime's LLM predictions, percent) for every run that can be made."""
    found = []
    for test_path in sorted(base.TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        m = base.SEED_RE.match(test_path.stem)
        if not m or (args.datasets and not any(d in m["dataset"] for d in args.datasets)):
            continue
        train_name = base.paired_train(test_path).stem
        for train_path in sorted(TRAIN_REGIME_DIR.glob(f"{train_name}_*.csv")):
            r = REGIME_RE.search(train_path.stem)
            if not r or (args.percents and int(r["percent"]) not in args.percents):
                continue
            percent = int(r["percent"])
            llm_path = args.llm_test_dir / f"{test_path.stem}_{percent}.csv"
            missing = [p.name for p in (TEST_DATED_DIR / test_path.name, llm_path) if not p.exists()]
            if missing:
                print(f"skip {test_path.stem}_{percent}: missing {', '.join(missing)}")
                continue
            found.append((test_path, train_path, llm_path, percent))
    return sorted(found, key=lambda j: (j[0].stem, j[3]))


# ---- one run ----------------------------------------------------------------------------------------------------------

class ConstantLLM(base.PrecomputedLLM):
    """An 'LLM' that always says the same: its one-hot carries no information, so the fusion MLP effectively sees the
    roberta-large embedding alone."""

    def __init__(self):
        super().__init__(collections.defaultdict(lambda: base.LABEL_COLUMNS[-1]))


def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args, roberta, branches) -> tuple[dict, pd.DataFrame]:
    name = f"{test_path.stem}_{percent}"
    t0 = time.time()
    regime_df = load_regime(train_path)
    test_df = load_test(test_path)
    print(f"\n=== {name}: regime {len(regime_df)} / test {len(test_df)} ===")

    llm_train = base.PrecomputedLLM(base.llm_predictions(base.LLM_TRAIN_DIR / f"{base.paired_train(test_path).stem}.csv"))
    llm_test = base.llm_predictions(llm_path)
    llm_test_preds = [llm_test[t] for t in test_df[base.TEXT_COLUMN]]
    constant = ConstantLLM()
    constant_test_preds = [base.LABEL_COLUMNS[-1]] * len(test_df)

    out = pd.DataFrame({
        base.TEXT_COLUMN: test_df[base.TEXT_COLUMN], "label": test_df["label"], "true": test_df["label"].map(base.LABEL_MAP),
        "llm_pred": llm_test_preds,
    })
    m = base.SEED_RE.match(test_path.stem)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "percent": percent,
           "regime_rows": len(regime_df), "test_rows": len(test_df),
           "llm_examples": max(1, round(2 * percent / 10))}  # the examples of the regime's LLM labels

    # (model, LLM input, test LLM labels, fusion builder): the MLP is the same, only its input differs
    models = [("roberta", constant, constant_test_preds, branches["plain"]),
              ("fusion", llm_train, llm_test_preds, branches["plain"])]
    if "ts" in branches:
        models.append(("fusion_ts", llm_train, llm_test_preds, branches["ts"]))
    for who, llm, test_labels, build in models:
        base.build_fusion = build
        fname = name if who == "fusion" else f"{name}_{who}"  # trial files / cache names per model
        if args.mlp_trials > 0:
            mlp, mlp_val_f1 = base.tune_mlp(roberta, llm, regime_df, regime_df, fname, args)
        else:
            mlp, mlp_val_f1 = base.default_mlp_params(args), None
        out[f"{who}_pred"] = base.mlp_fit_predict(roberta, llm, mlp, regime_df, regime_df, test_df, test_labels, fname, "final", verbose=False)
        row.update({f"{who}_mlp_lr": mlp["fusion_lr"], f"{who}_mlp_epochs": mlp["num_epochs"],
                    f"{who}_mlp_batch_size": mlp["batch_size"], f"{who}_mlp_val_f1": mlp_val_f1})
        gc.collect()

    for who in WHO:
        if f"{who}_pred" in out:
            row.update({f"{who}_{k}": v for k, v in base.scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row["seconds"] = round(time.time() - t0)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k.endswith(("f1_weighted", "seconds"))})
    return row, out


def aggregate(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and regime: mean and sample std (ddof=1) of every metric over its seeds."""
    grouped = seed_rows.groupby(["dataset", "percent"])
    summary = pd.DataFrame({
        "n_seeds": grouped.size(),
        "seeds": grouped["seed"].apply(lambda s: " ".join(sorted(s, key=int))),
    })
    for metric in METRICS:
        if metric in seed_rows:
            summary[f"{metric}_mean"] = grouped[metric].mean()
            summary[f"{metric}_std"] = grouped[metric].std(ddof=1)
    return summary.reset_index()


def publish(staging: Path) -> None:
    """Move what the run wrote into its staging folder to RESULT_DIR, with the start time of the run in every file name."""
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            target_dir = RESULT_DIR / path.parent.relative_to(staging)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target_dir / f"{path.stem}_{SUFFIX}{path.suffix}")


def build_frozen_roberta(model_name: str):
    """The pretrained roberta-large, never fine-tuned and frozen, with its embeddings memoised per sentence for the
    whole sweep (FusionEnsemble asks for them on every fit() / predict())."""
    config = ModelConfig(model_name="roberta-large", model_type=ModelType.TRADITIONAL_ML,
                         parameters={"model_name": model_name, "max_length": 128, "batch_size": 32})
    roberta = RoBERTaLargeClassifier(config=config, text_column=base.TEXT_COLUMN, label_columns=base.LABEL_COLUMNS,
                                     multi_label=False, auto_save_results=False, cache_dir=str(WORK_DIR / "models"))
    roberta.load_pretrained()
    base.EmbeddingMemo(roberta)
    return roberta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--llm-test-dir", type=Path, default=LLM_REGIME_DIR,
                        help="the regimes' LLM predictions (<test file>_<p>.csv) of predict_all_files_llm_low_data_regimes.py")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action="store_true", help="skip runs that already have a result file in the output folder")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="Optuna trials per run and model tuning the fusion MLP; 0 = the MLP's default configuration")
    parser.add_argument("--ml-proj-dim", type=int, default=768,
                        help="project roberta-large's 1024-dim [CLS] embedding down to this size before the MLP; 0 disables")
    parser.add_argument("--roberta-model", default="roberta-large", help="the (frozen) encoder; roberta-large unless testing")
    parser.add_argument("--no-ts", action="store_true", help="skip the fusion model with the time series branch")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="number of daily values of every series up to the sentence's date; 0 = the whole series")
    parser.add_argument("--ts-pooling", choices=("mean", "max"), default="mean", help="pooling of the three series embeddings")
    args = parser.parse_args()

    todo = jobs(args)
    if not todo:
        raise SystemExit("no runs found (are the regimes of dataset_scripts/make_low_data_regimes.py and the LLM "
                         "predictions of predict_all_files_llm_low_data_regimes.py there?)")
    if args.resume:
        done = {re.sub(r"_low_data_regimes_\d{8}_\d{6}$", "", p.stem) for p in RESULT_DIR.glob("lab-manual-*_low_data_regimes_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs ({', '.join(sorted({str(j[3]) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j[0] for j in todo})} test files; models: roberta, fusion{'' if args.no_ts else ', fusion_ts'} "
          f"(frozen encoders, MLP tuned with {args.mlp_trials} Optuna trials)")
    if args.dry_run:
        for test_path, train_path, llm_path, percent in todo[:10]:
            print(f"  {test_path.stem}_{percent}: train {train_path.name}, LLM labels {llm_path.name}")
        print("  ..." if len(todo) > 10 else "")
        return
    if not todo:
        return

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # FusionEnsemble writes its prediction caches to ./cache: this run's own folder
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    os.chdir(WORK_DIR)
    staging = WORK_DIR / "out"
    staging.mkdir()
    base.RESULT_DIR = files.FUSION_DIR = staging

    roberta = build_frozen_roberta(args.roberta_model)
    plain_build_fusion = base.build_fusion
    branches = {"plain": plain_build_fusion}
    if not args.no_ts:
        branch = TSBranch(args)
        ts_tag = f"ts{args.ts_window or 'full'}pool{args.ts_pooling}"

        def build_fusion_ts(ml_model, llm_model, fus, name, tag, save):
            # the TS settings are part of the tag: cache files of the plain and the time series fusion never collide
            fusion = plain_build_fusion(ml_model, llm_model, fus, name, f"{tag}_{ts_tag}", save)
            loader, encoders, transformer = branch.fresh()
            fusion.add_ts_model(loader, encoders, transformer, date_column=DATE_COLUMN, window_days=args.ts_window or FULL_HISTORY)
            fusion.ts_pooling = args.ts_pooling
            fusion._run_ts_branch = types.MethodType(run_pooled_ts_branch, fusion)
            return fusion

        branches["ts"] = build_fusion_ts
        print(f"time series: {', '.join(TS_SERIES)}, window {args.ts_window or 'full history'}"
              f"{'' if not args.ts_window else ' days'}, frozen TimesFM encoders, pooled ({args.ts_pooling})")

    rows, failures = [], []
    for test_path, train_path, llm_path, percent in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name = f"{test_path.stem}_{percent}"
        try:
            row, out = run_job(test_path, train_path, llm_path, percent, args, roberta, branches)
        except Exception as e:  # one failing run must not stop the sweep
            traceback.print_exc()
            failures.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(failures).to_csv(RESULT_DIR / f"failures_{SUFFIX}.csv", index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        out.to_csv(staging / f"{name}.csv", index=False)
        out.to_json(staging / f"{name}.json", orient="records", force_ascii=False, indent=2)
        publish(staging)
        rows.append(row)
        seed_rows = pd.DataFrame(rows)
        seed_rows.to_csv(RESULT_DIR / f"summary_seeds_{SUFFIX}.csv", index=False)
        aggregate(seed_rows).to_csv(RESULT_DIR / f"summary_{SUFFIX}.csv", index=False)

    if rows:
        summary = aggregate(pd.DataFrame(rows))
        print("\n=== weighted F1 mean +- std over seeds ===")
        for _, r in summary.iterrows():
            cells = "  ".join(f"{who} {r[f'{who}_f1_weighted_mean']:.3f}+-{r[f'{who}_f1_weighted_std']:.3f}"
                              for who in WHO if f"{who}_f1_weighted_mean" in r)
            print(f"{r['dataset']} {int(r['percent']):>3}% (n={r['n_seeds']}): {cells}")
    if failures:
        print(f"\n{len(failures)} runs failed: {RESULT_DIR / f'failures_{SUFFIX}.csv'}")
    print(f"\nSaved in {RESULT_DIR}: summary_{SUFFIX}.csv, summary_seeds_{SUFFIX}.csv and one "
          f"<test file>_<p>_{SUFFIX}.csv/.json per run")


if __name__ == "__main__":
    main()
