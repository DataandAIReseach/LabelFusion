"""Low-data regimes: roberta-large alone, LabelFusion (roberta-large + LLM) and LabelFusion with the time series.

For every test file (data/test_data_dated/<dataset>-<seed>.xlsx, always the FULL test set) and every regime p
(10, 20, ..., 100 %) of its training file (data/training_data_dated_low_data_regimes/<train name>_<p>.csv, made by
dataset_scripts/make_low_data_regimes.py) one run trains on the regime and predicts the test set with four models:

  llm           the LLM labels of that regime: GPT-5 nano with 2 examples per 10 % of the training data
                (predict_all_files_llm_low_data_regimes.py, outputs/predictions_low_data_regimes/...). These are
                also the LLM labels the fusion models get for the test sentences.
  roberta       roberta-large alone, fine-tuned on the regime (train 80 % / validation 20 %, stratified).
  fusion        LabelFusion as in predict_all_labelFusion.py: the fine-tuned roberta-large is frozen, a fusion MLP
                learns from its [CLS] embedding + the LLM label on the validation split. The LLM labels of the
                train/validation sentences are the zero-shot ones of predict_all_files_llm.py
                (outputs/predictions/train_gpt-5-nano_zero_shot).
  fusion_ts     the same fusion MLP with the time series branch of predict_all_labelFusion_TS_fixed_window.py:
                DGS10 / T10Y2Y / VIXCLS (data/3 Ts/daily_series.csv), the last --ts-window days up to the date of
                the sentence, TimesFM encoders, pooled over the series (--ts-pool-only, default; --no-ts-pool-only
                switches to the CrossTSTransformer's attention). Skipped with --no-ts.

roberta-large is fine-tuned once per run and shared by the three roberta-based models, so they differ only in
what the fusion MLP sees. Everything is scored on the full test set: weighted F1 (primary), macro-F1, accuracy.

Tuning is OFF by default, because a sweep has 240 runs: roberta-large uses the default configuration (lr 1e-5,
batch 16, up to --epochs epochs with best-epoch selection and --retries on a collapse) and the MLPs their defaults.
--trials N / --mlp-trials N switch on the Optuna searches of predict_all_labelFusion.py per run, and
--roberta-params-from <summary_seeds.csv of predict_all_labelFusion.py> takes the learning rate / batch size / epochs
tuned there for the same test file (at 100 % of the data) for all regimes of it.

Several runs (e.g. one sbatch job per group of datasets) can run at the same time without influencing each other:
every run has its own work folder (.work/<job id>) for the library's prediction caches and the saved roberta-large
models, and every result file carries the start time of the run in its name. Writes to ./outputs/predict_low_data_regimes/:
    <test file>_<p>_low_data_regimes_<date_time>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred, fusion_pred, fusion_ts_pred
    summary_seeds_low_data_regimes_<date_time>.csv             one row per run (all metrics, parameters used)
    summary_low_data_regimes_<date_time>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_low_data_regimes_<date_time>.csv                  runs that raised an error (the sweep goes on)
    trials/...                                                 Optuna trials, if tuning is on

    python testing/predict_low_data_regimes.py --dry-run                                  # which runs, nothing is trained
    python testing/predict_low_data_regimes.py --datasets pc-split-test --percents 10 100 # a first look
    python testing/predict_low_data_regimes.py                                            # everything
    python testing/predict_low_data_regimes.py --no-ts                                    # without the time series model
    python testing/predict_low_data_regimes.py --roberta-params-from outputs/labelfusion_roberta_large/summary_seeds.csv --mlp-trials 10
"""

from __future__ import annotations

import argparse
import atexit
import copy
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
from sklearn.model_selection import train_test_split  # noqa: E402

from test_textclassify_llm import _add_onehot_labels  # noqa: E402
from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder  # noqa: E402

stage1 = base.stage1
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
    """Loads the series and the pretrained TimesFM encoders once; fresh() hands every fusion MLP its own copy."""

    def __init__(self, args):
        self.loader = DailySeries(TS_FILE, TS_SERIES)
        self.loader.load_all()
        self.encoders = {name: TSEmbedder(pooling="mean") for name in TS_SERIES}
        self.caches = {name: {} for name in TS_SERIES}
        self.hidden_size = next(iter(self.encoders.values())).hidden_size
        self.output_dim = args.ts_dim
        self.finetune = args.ts_finetune

    def fresh(self):
        encoders = {name: copy.deepcopy(enc) for name, enc in self.encoders.items()}
        if not self.finetune:
            for enc in encoders.values():
                enc.requires_grad_(False)
            encoders = {name: CachedEmbedder(enc, self.caches[name]) for name, enc in encoders.items()}
        torch.manual_seed(base.RANDOM_STATE)
        transformer = CrossTSTransformer(series_names=TS_SERIES, hidden_size=self.hidden_size, output_dim=self.output_dim)
        return self.loader, encoders, transformer


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


def split_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regime -> (train, validation), 80/20 stratified, as base.split_pair does for a whole train file."""
    try:
        train_df, val_df = train_test_split(df, train_size=0.8, random_state=base.RANDOM_STATE, stratify=df["label"])
    except ValueError:  # a class too small to stratify
        train_df, val_df = train_test_split(df, train_size=0.8, random_state=base.RANDOM_STATE)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


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

def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args, fusion_builders) -> tuple[dict, pd.DataFrame]:
    name = f"{test_path.stem}_{percent}"
    t0 = time.time()
    train_df, val_df = split_regime(load_regime(train_path))
    test_df = load_test(test_path)
    print(f"\n=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = base.PrecomputedLLM(base.llm_predictions(base.LLM_TRAIN_DIR / f"{base.paired_train(test_path).stem}.csv"))
    llm_test = base.llm_predictions(llm_path)
    llm_test_preds = [llm_test[t] for t in test_df[base.TEXT_COLUMN]]

    # roberta-large alone, once for all three roberta-based models
    if args.trials > 0:
        rob, roberta_val_f1 = stage1.tune(train_df, val_df, name, args)
    else:
        rob, roberta_val_f1 = stage1.default_params(args), None
    roberta, memo = base.train_roberta(rob, train_df, val_df, retries=args.retries)
    roberta_preds = roberta.predict_without_saving(test_df).predictions

    out = pd.DataFrame({
        base.TEXT_COLUMN: test_df[base.TEXT_COLUMN], "label": test_df["label"], "true": test_df["label"].map(base.LABEL_MAP),
        "llm_pred": llm_test_preds, "roberta_pred": roberta_preds,
    })
    row = {"dataset": base.SEED_RE.match(test_path.stem)["dataset"], "seed": base.SEED_RE.match(test_path.stem)["seed"],
           "file": name, "percent": percent, "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df),
           "llm_examples": max(1, round(2 * percent / 10)),  # the LLM labels of the regime, see predict_all_files_llm_low_data_regimes.py
           "roberta_lr": rob["learning_rate"], "roberta_batch_size": rob["batch_size"], "roberta_epochs": rob["num_epochs"],
           "roberta_val_f1": roberta_val_f1}

    # the fusion MLPs (frozen roberta-large embedding + LLM label), with and without the time series branch
    for who, build in fusion_builders.items():
        base.build_fusion = build
        fname = name if who == "fusion" else f"{name}_ts"
        if args.mlp_trials > 0:
            mlp, mlp_val_f1 = base.tune_mlp(roberta, llm, train_df, val_df, fname, args)
        else:
            mlp, mlp_val_f1 = base.default_mlp_params(args), None
        out[f"{who}_pred"] = base.mlp_fit_predict(roberta, llm, mlp, train_df, val_df, test_df, llm_test_preds, fname, "final", verbose=False)
        row.update({f"{who}_mlp_lr": mlp["fusion_lr"], f"{who}_mlp_epochs": mlp["num_epochs"],
                    f"{who}_mlp_batch_size": mlp["batch_size"], f"{who}_mlp_val_f1": mlp_val_f1})
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--llm-test-dir", type=Path, default=LLM_REGIME_DIR,
                        help="the regimes' LLM predictions (<test file>_<p>.csv) of predict_all_files_llm_low_data_regimes.py")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action="store_true", help="skip runs that already have a result file in the output folder")
    parser.add_argument("--epochs", type=int, default=10,
                        help="roberta-large (maximum) epochs of the default configuration and of the grid points")
    parser.add_argument("--retries", type=int, default=2,
                        help="repeat the final roberta-large training with another seed up to N times if it collapses to one class")
    parser.add_argument("--trials", type=int, default=0,
                        help="roberta-large Optuna trials per run (the 16 grid points first); 0 = default configuration")
    parser.add_argument("--mlp-trials", type=int, default=0,
                        help="fusion MLP Optuna trials per run and model; 0 = default configuration")
    parser.add_argument("--ml-proj-dim", type=int, default=768,
                        help="project roberta-large's 1024-dim [CLS] embedding down to this size before the fusion MLP; 0 disables")
    parser.add_argument("--no-ts", action="store_true", help="skip the fusion model with the time series branch")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="number of daily values of every series up to the sentence's date; 0 = the whole series")
    parser.add_argument("--ts-dim", type=int, default=64, help="size of the time series embedding (attention variant)")
    parser.add_argument("--ts-pool-only", action=argparse.BooleanOptionalAction, default=True,
                        help="pool the per-series embeddings and pass them straight to the MLP (default); "
                             "--no-ts-pool-only uses the CrossTSTransformer's attention")
    parser.add_argument("--ts-pooling", choices=("mean", "max"), default="mean", help="pooling over the series")
    parser.add_argument("--ts-finetune", action="store_true", help="also train the TimesFM encoders (default: frozen)")
    parser.add_argument("--roberta-params-from", type=Path,
                        help="summary_seeds.csv of a predict_all_labelFusion.py run: roberta-large parameters per test file")
    args = parser.parse_args()

    todo = jobs(args)
    if not todo:
        raise SystemExit("no runs found (are the regimes of dataset_scripts/make_low_data_regimes.py and the LLM "
                         "predictions of predict_all_files_llm_low_data_regimes.py there?)")
    if args.resume:
        done = {re.sub(r"_low_data_regimes_\d{8}_\d{6}$", "", p.stem) for p in RESULT_DIR.glob("lab-manual-*_low_data_regimes_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs ({', '.join(sorted({str(j[3]) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j[0] for j in todo})} test files; models: roberta, fusion{'' if args.no_ts else ', fusion_ts'}")
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
    base.RESULT_DIR = stage1.RESULT_DIR = files.FUSION_DIR = staging

    plain_build_fusion = base.build_fusion
    fusion_builders = {"fusion": plain_build_fusion}
    if not args.no_ts:
        branch = TSBranch(args)
        ts_out = f"pool{args.ts_pooling}" if args.ts_pool_only else f"d{args.ts_dim}"
        ts_tag = f"ts{args.ts_window or 'full'}{ts_out}{'ft' if args.ts_finetune else ''}"

        def build_fusion_ts(ml_model, llm_model, fus, name, tag, save):
            # the TS settings are part of the tag: cache files of the plain and the time series fusion never collide
            fusion = plain_build_fusion(ml_model, llm_model, fus, name, f"{tag}_{ts_tag}", save)
            loader, encoders, transformer = branch.fresh()
            fusion.add_ts_model(loader, encoders, transformer, date_column=DATE_COLUMN, window_days=args.ts_window or FULL_HISTORY)
            if args.ts_pool_only:
                transformer.output_dim = transformer.hidden_size  # the fusion MLP reads its TS input size from here
                fusion.ts_pooling = args.ts_pooling
                fusion._run_ts_branch = types.MethodType(run_pooled_ts_branch, fusion)
            return fusion

        fusion_builders["fusion_ts"] = build_fusion_ts
        print(f"time series: {', '.join(TS_SERIES)}, window {args.ts_window or 'full history'}"
              f"{'' if not args.ts_window else ' days'}, "
              f"{'pooled (' + args.ts_pooling + ')' if args.ts_pool_only else 'attention'}, TimesFM {'fine-tuned' if args.ts_finetune else 'frozen'}")

    if args.roberta_params_from:
        reuse = pd.read_csv(args.roberta_params_from).set_index("file")
        plain_tune = stage1.tune

        def tune_or_reuse(train_df, val_df, name, targs):
            pair = REGIME_RE.sub("", name)  # <test file>_<p> -> <test file>
            if pair not in reuse.index:
                return plain_tune(train_df, val_df, name, targs)
            r = reuse.loc[pair]
            params = {**stage1.default_params(targs), "learning_rate": float(r["roberta_lr"]),
                      "batch_size": int(r["roberta_batch_size"]), "num_epochs": int(r["roberta_epochs"])}
            val_f1 = r.get("roberta_val_f1")
            return params, (None if pd.isna(val_f1) else float(val_f1))

        stage1.tune = tune_or_reuse
        args.trials = max(args.trials, 1)  # tune() now only looks the parameters up

    rows, failures = [], []
    for test_path, train_path, llm_path, percent in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name = f"{test_path.stem}_{percent}"
        try:
            row, out = run_job(test_path, train_path, llm_path, percent, args, fusion_builders)
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
