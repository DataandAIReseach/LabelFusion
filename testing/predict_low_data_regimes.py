"""Low-data regimes: the full program at --full-percents (default: every regime), frozen encoders and only a tuned MLP at the others.

For every test file (data/test_data_dated/<dataset>-<seed>.xlsx, always the FULL test set) and every regime p
(10, 20, ..., 100 %) of its training file (data/training_data_dated_low_data_regimes/<train name>_<p>.csv, made by
dataset_scripts/make_low_data_regimes.py) one run trains on the regime and predicts the test set. Four columns of
predictions come out; "training" in the summary says how they were made:

  llm           the LLM labels of the regime: GPT-5 nano with 2 examples per 10 % of the training data, read from the
                regime's LLM cache (below). These are also the LLM labels the fusion models get for the test sentences.
  roberta       frozen regimes: MLP on the frozen, pretrained roberta-large [CLS] embedding alone (the LLM input is a
                constant, i.e. no information).  full regimes: roberta-large fine-tuned alone.
  fusion        LabelFusion: MLP on the roberta-large embedding + the LLM label. The LLM labels of the regime's
                sentences are the zero-shot ones (also from the LLM cache, below).
  fusion_ts     the same with the time series branch: DGS10 / T10Y2Y / VIXCLS (data/3 Ts/daily_series.csv), the last
                --ts-window days up to the date of the sentence, one TimesFM encoder per series. Frozen regimes: frozen
                encoders, pooled over the series (--ts-pooling). Full regimes: FINE-TUNED encoders and a trained
                CrossTSTransformer (attention over the series, --ts-dim outputs). Skipped with --no-ts.

FROZEN regimes (all but --full-percents; none by default): nothing but the MLP is trained. roberta-large is the pretrained model
(never fine-tuned), the TimesFM encoders and the time series transformer are frozen. The MLP is trained on the WHOLE
regime (no train/validation split is needed to protect a fine-tuned encoder). Optuna (--mlp-trials, default 30,
0 = the MLP's default configuration) tunes its learning rate, epochs, hidden layers and batch size per run and model,
on one half of the regime scored on the other half; the best parameters then train the MLP on the whole regime. Every
sentence is embedded once for the whole sweep (roberta-large and TimesFM), so such a run costs only MLP trainings.

FULL regimes (--full-percents, default every regime found; none with --full-percents and no value): the full program of
predict_all_labelFusion.py. The regime is split 80/20 (stratified) into train and validation. Stage 1: roberta-large is
fine-tuned alone; Optuna (--trials, default 24: the 16-point learning-rate x batch-size grid, then free refinement)
over lr / batch size / epochs, weighted F1 on the validation split, with class weights, 10 % warmup, best-epoch
selection and --retries (default 2) on a one-class collapse. Stage 2: the fine-tuned roberta-large is frozen and the
fusion MLPs are tuned and trained on the VALIDATION split, which roberta-large never trained on: fusion with the
fine-tuned roberta-large + the LLM label (--mlp-trials), fusion_ts additionally with the fine-tuned time series model,
whose TimesFM encoders and CrossTSTransformer are trained together with the MLP (--ts-mlp-trials, default 10, because
every trial fine-tunes three TimesFM encoders; the library trains them with the MLP's learning rate). --roberta-params-from <summary_seeds.csv of predict_all_labelFusion.py> takes the learning rate / batch size /
epochs tuned there for the same test file instead of tuning stage 1 again. The fine-tuned "roberta" and the frozen
"roberta" of the other regimes are different models: compare the curve across that boundary with care.

Everything is scored on the full test set: weighted F1 (primary), macro-F1, accuracy.

LLM cache: no LLM is called. The predictions are read from the LLM caches of predict_all_files_llm.py /
predict_all_files_llm_low_data_regimes.py, which live INSIDE this script's output folder:

    outputs/predict_low_data_regimes/llm_cache/
        predict_all_train_gpt-5-nano_zero_shot/                      the zero-shot labels of all training sentences
        predict_all_test_gpt-5-nano_<k>_shot_<test file>_<p>/        the labels of the full test set for regime p
                                                                     (k = 2 examples per 10 % drawn from that regime)
A cache folder that is not there yet is copied once from the legacy outputs/llm_cache (--legacy-llm-cache-dir); the
legacy folder itself is never changed.

Several runs (e.g. one sbatch job per group of datasets) can run at the same time without influencing each other:
every run has its own work folder (.work/<job id>) for the saved roberta-large models and the library's prediction
caches, and every result file carries the start time of the run in its name. Everything lives in ./outputs/predict_low_data_regimes/:
    llm_cache/                                                 the LLM caches of the regimes (above)
    <test file>_<p>_low_data_regimes_<date_time>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred, fusion_pred, fusion_ts_pred
    summary_seeds_low_data_regimes_<date_time>.csv             one row per run (all metrics, "training", parameters used)
    summary_low_data_regimes_<date_time>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_low_data_regimes_<date_time>.csv                  runs that raised an error (the sweep goes on)
    trials/...                                                 Optuna trials

    python testing/predict_low_data_regimes.py --dry-run                                  # which runs, nothing is trained
    python testing/predict_low_data_regimes.py --datasets pc-split-test --percents 10 80  # a first look: two regimes
    python testing/predict_low_data_regimes.py                                            # everything
    python testing/predict_low_data_regimes.py --no-ts --mlp-trials 10 --full-percents 100   # without the time series, fewer trials
    python testing/predict_low_data_regimes.py --roberta-params-from outputs/labelfusion_roberta_large/summary_seeds.csv
"""

from __future__ import annotations

import argparse
import atexit
import collections
import copy
import gc
import hashlib
import os
import pickle
import re
import shutil
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR / "old"))  # the helper modules (predict_all_labelFusion.py, ...) live in testing/old

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
from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402
from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder  # noqa: E402

stage1 = base.stage1
REPO_ROOT = base.TEST_DIR.parent.parent
TRAIN_REGIME_DIR = REPO_ROOT / "data" / "training_data_dated_low_data_regimes"
TEST_DATED_DIR = REPO_ROOT / "data" / "test_data_dated"
LLM_CACHE_DIR = RESULT_DIR / "llm_cache"  # the LLM caches of the regimes (set from --llm-cache-dir)
LEGACY_LLM_CACHE_DIR = base.OUT_DIR / "llm_cache"  # where predict_all_files_llm*.py wrote them (set from --legacy-llm-cache-dir)
LLM_MODEL = "gpt-5-nano"  # the model name in the cache folder names (set from --llm-model)
SHOTS_PER_10_PERCENT = 2
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
    """Loads the series and the pretrained TimesFM encoders once and hands out two kinds of branches.

    fresh_frozen():    FROZEN regimes. The encoders and the CrossTSTransformer are frozen (the transformer is only needed
                       by FusionEnsemble.add_ts_model, the pooled branch never calls it); the encoders' embeddings are
                       cached for the whole sweep.
    fresh_finetuned(): FULL regimes. Own copies of the pretrained encoders and a new CrossTSTransformer (attention over the
                       three series, --ts-dim outputs); FusionEnsemble fine-tunes them together with the fusion MLP. Every
                       Optuna trial and the final MLP start from the pretrained TimesFM weights."""

    def __init__(self, args):
        self.loader = DailySeries(TS_FILE, TS_SERIES)
        self.loader.load_all()
        self.pretrained = {name: TSEmbedder(pooling="mean") for name in TS_SERIES}
        for enc in self.pretrained.values():
            enc.requires_grad_(False)
        self.hidden_size = next(iter(self.pretrained.values())).hidden_size
        self.caches = {name: {} for name in TS_SERIES}
        self.cached = {name: CachedEmbedder(enc, self.caches[name]) for name, enc in self.pretrained.items()}
        self.ts_dim = args.ts_dim

    def fresh_frozen(self):
        torch.manual_seed(base.RANDOM_STATE)
        transformer = CrossTSTransformer(series_names=TS_SERIES, hidden_size=self.hidden_size, output_dim=self.hidden_size)
        transformer.requires_grad_(False)
        return self.loader, self.cached, transformer

    def fresh_finetuned(self):
        encoders = {name: copy.deepcopy(enc) for name, enc in self.pretrained.items()}
        for enc in encoders.values():
            enc.requires_grad_(True)
        torch.manual_seed(base.RANDOM_STATE)
        transformer = CrossTSTransformer(series_names=TS_SERIES, hidden_size=self.hidden_size, output_dim=self.ts_dim)
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


def shots_for(percent: int) -> int:
    """2 examples per 10 % of the training data: 10 % -> 2, 20 % -> 4, ..., 100 % -> 20."""
    return max(1, round(SHOTS_PER_10_PERCENT * percent / 10))


def train_cache_name() -> str:
    return f"train_{LLM_MODEL}_zero_shot"


def regime_cache_name(test_stem: str, percent: int) -> str:
    return f"test_{LLM_MODEL}_{shots_for(percent)}_shot_{test_stem}_{percent}"


def find_cache(run_name: str) -> Path | None:
    """The cache folder of an LLM run: in this script's llm_cache, else in the legacy outputs/llm_cache."""
    for root in (LLM_CACHE_DIR, LEGACY_LLM_CACHE_DIR):
        folder = root / f"predict_all_{run_name}"
        if folder.is_dir() and any(folder.glob("cache_*.pkl")):
            return folder
    return None


def import_cache(folder: Path) -> Path:
    """Make sure the folder is inside this script's llm_cache: a legacy folder is copied once, never moved."""
    target = LLM_CACHE_DIR / folder.name
    if folder != target and not target.exists():
        LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, target)
    return target


def read_cache(folder: Path) -> dict[str, str]:
    """sentence -> predicted class name, from the classifier's prediction cache (only successful, unambiguous calls)."""
    preds: dict[str, str] = {}
    for pkl in sorted(folder.glob("cache_*.pkl")):
        with open(pkl, "rb") as fh:
            for entry in pickle.load(fh).values():
                vec = entry.get("prediction")
                if entry.get("success") and vec is not None and sum(vec) == 1:
                    preds[entry["text"]] = base.LABEL_COLUMNS[list(vec).index(1)]
    return preds


_TRAIN_LABELS: dict[str, str] = {}


def train_labels() -> dict[str, str]:
    """The zero-shot labels of every training sentence (one cache for all training files, read once)."""
    if not _TRAIN_LABELS:
        folder = find_cache(train_cache_name())
        if folder is None:
            raise FileNotFoundError(f"no zero-shot cache predict_all_{train_cache_name()} in {LLM_CACHE_DIR} or {LEGACY_LLM_CACHE_DIR}")
        _TRAIN_LABELS.update(read_cache(import_cache(folder)))
    return _TRAIN_LABELS


def jobs(args) -> list[tuple[Path, Path, Path, int]]:
    """(test file, train regime file, the regime's LLM cache folder, percent) for every run that can be made."""
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
            llm_path = find_cache(regime_cache_name(test_path.stem, percent))
            missing = ([TEST_DATED_DIR / test_path.name] if not (TEST_DATED_DIR / test_path.name).exists() else []) \
                + ([f"LLM cache predict_all_{regime_cache_name(test_path.stem, percent)}"] if llm_path is None else [])
            if missing:
                print(f"skip {test_path.stem}_{percent}: missing {', '.join(map(str, missing))}")
                continue
            found.append((test_path, train_path, llm_path, percent))
    return sorted(found, key=lambda j: (j[0].stem, j[3]))


# ---- one run ----------------------------------------------------------------------------------------------------------

class ConstantLLM(base.PrecomputedLLM):
    """An 'LLM' that always says the same: its one-hot carries no information, so the fusion MLP effectively sees the
    roberta-large embedding alone."""

    def __init__(self):
        super().__init__(collections.defaultdict(lambda: base.LABEL_COLUMNS[-1]))


def prepare(test_path: Path, train_path: Path, llm_path: Path, percent: int):
    """The regime, the full test set, the LLM labels, the start of the output frame and the summary row."""
    name = f"{test_path.stem}_{percent}"
    regime_df = load_regime(train_path)
    test_df = load_test(test_path)
    train_labels_ = train_labels()
    missing = [t for t in regime_df[base.TEXT_COLUMN] if t not in train_labels_]
    if missing:
        raise KeyError(f"{len(missing)} sentences of the regime have no zero-shot label in the LLM cache")
    llm_train = base.PrecomputedLLM(train_labels_)
    llm_test = read_cache(import_cache(llm_path))  # the regime's labels of the test sentences
    missing = [t for t in test_df[base.TEXT_COLUMN] if t not in llm_test]
    if missing:
        raise KeyError(f"{len(missing)} test sentences have no label in the LLM cache {llm_path.name}")
    llm_test_preds = [llm_test[t] for t in test_df[base.TEXT_COLUMN]]
    out = pd.DataFrame({
        base.TEXT_COLUMN: test_df[base.TEXT_COLUMN], "label": test_df["label"], "true": test_df["label"].map(base.LABEL_MAP),
        "llm_pred": llm_test_preds,
    })
    m = base.SEED_RE.match(test_path.stem)
    row = {"dataset": m["dataset"], "seed": m["seed"], "file": name, "percent": percent,
           "regime_rows": len(regime_df), "test_rows": len(test_df),
           "llm_examples": shots_for(percent)}  # the examples of the regime's LLM labels
    return name, regime_df, test_df, llm_train, llm_test_preds, out, row


def finish(row: dict, out: pd.DataFrame, t0: float) -> tuple[dict, pd.DataFrame]:
    for who in WHO:
        if f"{who}_pred" in out:
            row.update({f"{who}_{k}": v for k, v in base.scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    row["seconds"] = round(time.time() - t0)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if k.endswith(("f1_weighted", "seconds"))})
    return row, out


def run_job(test_path: Path, train_path: Path, llm_path: Path, percent: int, args, roberta, branches) -> tuple[dict, pd.DataFrame]:
    """FROZEN regime: only MLPs are trained, on the whole regime."""
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = prepare(test_path, train_path, llm_path, percent)
    print(f"\n=== {name} (frozen encoders): regime {len(regime_df)} / test {len(test_df)} ===")
    row["training"] = "frozen"
    constant = ConstantLLM()
    constant_test_preds = [base.LABEL_COLUMNS[-1]] * len(test_df)

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
    return finish(row, out, t0)


def run_job_full(test_path: Path, train_path: Path, llm_path: Path, percent: int, args, branches) -> tuple[dict, pd.DataFrame]:
    """FULL regime: the program of predict_all_labelFusion.py (+ the time series model) on the regime."""
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = prepare(test_path, train_path, llm_path, percent)
    train_df, val_df = split_regime(regime_df)
    print(f"\n=== {name} (full program): train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")
    row.update({"training": "finetuned", "train_rows": len(train_df), "val_rows": len(val_df)})

    # Stage 1: roberta-large alone (Optuna over lr x batch size x epochs, class weights, warmup, best epoch, retries)
    if args.trials > 0:
        rob, roberta_val_f1 = stage1.tune(train_df, val_df, name, args)
    else:
        rob, roberta_val_f1 = stage1.default_params(args), None
    roberta, memo = base.train_roberta(rob, train_df, val_df, retries=args.retries)
    out["roberta_pred"] = roberta.predict_without_saving(test_df).predictions
    row.update({"roberta_lr": rob["learning_rate"], "roberta_batch_size": rob["batch_size"],
                "roberta_epochs": rob["num_epochs"], "roberta_val_f1": roberta_val_f1})

    # Stage 2: roberta-large frozen, the fusion MLPs are tuned and trained on the validation split
    # fusion: the fine-tuned roberta-large + LLM label. fusion_ts: additionally the FINE-TUNED time series model
    # (TimesFM encoders + CrossTSTransformer, trained with the MLP); that is far more expensive per MLP trial.
    models = [("fusion", branches["plain"], args.mlp_trials)]
    if "ts_finetuned" in branches:
        models.append(("fusion_ts", branches["ts_finetuned"], args.ts_mlp_trials))
    for who, build, trials in models:
        base.build_fusion = build
        fname = name if who == "fusion" else f"{name}_{who}"
        if trials > 0:
            mlp, mlp_val_f1 = base.tune_mlp(roberta, llm_train, train_df, val_df, fname, argparse.Namespace(**{**vars(args), "mlp_trials": trials}))
        else:
            mlp, mlp_val_f1 = base.default_mlp_params(args), None
        out[f"{who}_pred"] = base.mlp_fit_predict(roberta, llm_train, mlp, train_df, val_df, test_df, llm_test_preds, fname, "final", verbose=False)
        row.update({f"{who}_mlp_lr": mlp["fusion_lr"], f"{who}_mlp_epochs": mlp["num_epochs"],
                    f"{who}_mlp_batch_size": mlp["batch_size"], f"{who}_mlp_val_f1": mlp_val_f1})
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return finish(row, out, t0)


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
    global LLM_CACHE_DIR, LEGACY_LLM_CACHE_DIR, LLM_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--llm-cache-dir", type=Path, default=LLM_CACHE_DIR,
                        help="the LLM caches of the regimes (default: outputs/predict_low_data_regimes/llm_cache)")
    parser.add_argument("--legacy-llm-cache-dir", type=Path, default=LEGACY_LLM_CACHE_DIR,
                        help="where predict_all_files_llm*.py wrote them; cache folders missing in --llm-cache-dir are copied from here")
    parser.add_argument("--llm-model", default=LLM_MODEL,
                        help="the LLM whose cache folders are read (name as in the folder names, e.g. gpt-5-nano)")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action="store_true", help="skip runs that already have a result file in the output folder")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="Optuna trials per run and model tuning the fusion MLP; 0 = the MLP's default configuration")
    parser.add_argument("--full-percents", nargs="*", type=int, default=None,
                        help="regimes that get the full program (fine-tuned roberta-large with Optuna, warmup, class weights, "
                             "best epoch, retries; MLPs on the validation split). Default: every regime; give no value for none")
    parser.add_argument("--ts-mlp-trials", type=int, default=10,
                        help="full regimes: Optuna trials for the fusion MLP WITH the fine-tuned time series model (every trial "
                             "fine-tunes three TimesFM encoders, so this is much more expensive than --mlp-trials); 0 = defaults")
    parser.add_argument("--ts-dim", type=int, default=64,
                        help="full regimes: size of the fine-tuned CrossTSTransformer's output fed to the MLP")
    parser.add_argument("--trials", type=int, default=24,
                        help="full regimes: roberta-large Optuna trials per run (the 16 grid points first, the rest is free); 0 = defaults")
    parser.add_argument("--epochs", type=int, default=10,
                        help="full regimes: roberta-large (maximum) epochs of the default configuration and of the grid points")
    parser.add_argument("--retries", type=int, default=2,
                        help="full regimes: repeat the final roberta-large training with another seed up to N times on a one-class collapse")
    parser.add_argument("--roberta-params-from", type=Path,
                        help="full regimes: summary_seeds.csv of a predict_all_labelFusion.py run, lr / batch size / epochs per test "
                             "file instead of tuning stage 1")
    parser.add_argument("--ml-proj-dim", type=int, default=768,
                        help="project roberta-large's 1024-dim [CLS] embedding down to this size before the MLP; 0 disables")
    parser.add_argument("--roberta-model", default="roberta-large", help="the (frozen) encoder; roberta-large unless testing")
    parser.add_argument("--no-ts", action="store_true", help="skip the fusion model with the time series branch")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="number of daily values of every series up to the sentence's date; 0 = the whole series")
    parser.add_argument("--ts-pooling", choices=("mean", "max"), default="mean", help="pooling of the three series embeddings")
    args = parser.parse_args()
    # absolute before main() changes into the work folder
    for key in ("llm_cache_dir", "legacy_llm_cache_dir", "roberta_params_from"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())

    LLM_CACHE_DIR, LEGACY_LLM_CACHE_DIR, LLM_MODEL = args.llm_cache_dir, args.legacy_llm_cache_dir, args.llm_model
    if find_cache(train_cache_name()) is None:
        raise SystemExit(f"no zero-shot cache predict_all_{train_cache_name()} in {LLM_CACHE_DIR} or {LEGACY_LLM_CACHE_DIR}")

    todo = jobs(args)
    if not todo:
        raise SystemExit("no runs found (are the regimes of dataset_scripts/make_low_data_regimes.py and the LLM "
                         "LLM caches of predict_all_files_llm_low_data_regimes.py there?)")
    if args.full_percents is None:  # default: the full program at every regime
        args.full_percents = sorted({j[3] for j in todo})
    if args.resume:
        done = {re.sub(r"_low_data_regimes_\d{8}_\d{6}$", "", p.stem) for p in RESULT_DIR.glob("lab-manual-*_low_data_regimes_*.csv")}
        todo = [j for j in todo if f"{j[0].stem}_{j[3]}" not in done]
    print(f"{len(todo)} runs ({', '.join(sorted({str(j[3]) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j[0] for j in todo})} test files; models: roberta, fusion{'' if args.no_ts else ', fusion_ts'} "
          f"(full program at {', '.join(str(p) + '%' for p in args.full_percents) or 'no regime'}, frozen encoders + "
          f"MLP tuned with {args.mlp_trials} Optuna trials at the others)")
    if args.dry_run:
        for test_path, train_path, llm_path, percent in todo[:10]:
            print(f"  {test_path.stem}_{percent} ({'full program' if percent in args.full_percents else 'frozen'}): "
                  f"train {train_path.name}, LLM cache {llm_path.parent.name}/{llm_path.name}")
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

    frozen = {}  # the frozen roberta-large is only built (and its embeddings cached) if a frozen regime is in the sweep
    plain_build_fusion = base.build_fusion
    branches = {"plain": plain_build_fusion}
    if not args.no_ts:
        branch = TSBranch(args)
        ts_tag = f"ts{args.ts_window or 'full'}pool{args.ts_pooling}"
        tsft_tag = f"tsft{args.ts_window or 'full'}d{args.ts_dim}"
        window_days = args.ts_window or FULL_HISTORY

        def build_fusion_ts(ml_model, llm_model, fus, name, tag, save):
            # FROZEN regimes. The TS settings are part of the tag: cache files of the different fusion models never collide
            fusion = plain_build_fusion(ml_model, llm_model, fus, name, f"{tag}_{ts_tag}", save)
            loader, encoders, transformer = branch.fresh_frozen()
            fusion.add_ts_model(loader, encoders, transformer, date_column=DATE_COLUMN, window_days=window_days)
            fusion.ts_pooling = args.ts_pooling
            fusion._run_ts_branch = types.MethodType(run_pooled_ts_branch, fusion)
            return fusion

        def build_fusion_ts_finetuned(ml_model, llm_model, fus, name, tag, save):
            # FULL regimes: the library's own branch (TimesFM encoders -> CrossTSTransformer attention), all trained
            fusion = plain_build_fusion(ml_model, llm_model, fus, name, f"{tag}_{tsft_tag}", save)
            loader, encoders, transformer = branch.fresh_finetuned()
            fusion.add_ts_model(loader, encoders, transformer, date_column=DATE_COLUMN, window_days=window_days)
            return fusion

        branches["ts"] = build_fusion_ts
        branches["ts_finetuned"] = build_fusion_ts_finetuned
        print(f"time series: {', '.join(TS_SERIES)}, window {args.ts_window or 'full history'}"
              f"{'' if not args.ts_window else ' days'}; frozen regimes: frozen TimesFM encoders, pooled ({args.ts_pooling}); "
              f"full regimes ({', '.join(str(p) + '%' for p in args.full_percents) or 'none'}): fine-tuned TimesFM encoders + "
              f"CrossTSTransformer ({args.ts_dim} dims)")

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
            if percent in args.full_percents:
                row, out = run_job_full(test_path, train_path, llm_path, percent, args, branches)
            else:
                if "roberta" not in frozen:
                    frozen["roberta"] = build_frozen_roberta(args.roberta_model)
                row, out = run_job(test_path, train_path, llm_path, percent, args, frozen["roberta"], branches)
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
