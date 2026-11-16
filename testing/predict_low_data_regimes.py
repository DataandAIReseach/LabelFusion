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

FULL regimes (--full-percents, default every regime found; none with --full-percents and no value): the regime is split 80/20
(stratified) into train and validation. roberta-large is fine-tuned ALONE on the train split (column roberta); Optuna (--trials,
default 24: the 16-point learning-rate x batch-size grid, then free refinement) over lr / batch size / epochs, weighted F1 on the
validation split, with class weights, 10 % warmup, best-epoch selection and --retries (default 2) on a one-class collapse.
--roberta-params-from <summary_seeds.csv of predict_all_labelFusion.py> takes the learning rate / batch size / epochs tuned there
for the same test file instead of tuning that again. The LabelFusion models of the full regimes are the JOINT ones (below). The
two-stage fusion (frozen fine-tuned roberta-large, MLP on the validation split: fusion, and fusion_ts with the fine-tuned time series
model, --mlp-trials / --ts-mlp-trials) is only computed with --two-stage. The fine-tuned "roberta" and the frozen "roberta" of the
other regimes are different models: compare the curve across that boundary with care.

JOINT models start from the roberta-large that stage 1 fine-tuned on the same train split (--joint-init finetuned, default; --joint-init
pretrained starts them from the pretrained model), so they are at least as good a starting point as the column roberta. They then
continue to train with the fusion loss. (An end-to-end model that starts from the pretrained model has to learn the task from
scratch with fewer epochs and no class weights, and tends to end up below the fine-tuned roberta.)
JOINT models (--joint, on by default, --no-joint skips them; full regimes only; every trial is a complete fine-tuning of roberta-large): roberta-large
and the fusion network are trained TOGETHER, end to end, on the TRAIN split (80 % of the regime): the fusion loss (cross-entropy on
[roberta-large [CLS] embedding, LLM label as one-hot]) is back-propagated into roberta-large (own learning rate ml_lr) and the fusion MLP
(fusion_lr), via FusionEnsemble's joint_training.
  fusion_joint     roberta-large + LLM one-hot.
  fusion_joint_ts  the same plus the time series model: the TimesFM encoders and the CrossTSTransformer (attention over DGS10 / T10Y2Y /
                   VIXCLS) are trained in the same loop (all three parts with the same loss; needs much GPU memory, roughly roberta-large
                   plus three TimesFM models with their optimizer states). Skipped with --no-ts.
The validation split (20 %) is not trained on; it scores every Optuna trial (--joint-trials: ml_lr, fusion_lr, epochs, batch size, hidden
layers) and the best trial's test predictions are taken, so the test file is never used for the choice. The LLM labels are the same
as for fusion.

Everything is scored on the full test set: weighted F1 (primary), macro-F1, accuracy.

Interruptions: a job has a time limit (e.g. 48 h on the cluster), so nothing is lost when it is cut. Finished runs are
skipped when the same command is started again (--no-resume repeats them), and the expensive calls of the run that was
running are kept on disk after every call, in outputs/predict_low_data_regimes/checkpoints/<test file>_<p>.json: every
Optuna trial of roberta-large (its validation predictions) and every fit of a fusion MLP (its predictions). Started
again, the run replays its Optuna studies from these records (they are deterministic, so the same trials come up, each
answered from the file in an instant) and trains only what was missing. Only the final fine-tuning of roberta-large
of an interrupted run is done again. A checkpoint file is deleted when its run is finished, and ignored if the regime
or the settings it was made with changed.

LLM cache: no LLM is called. The predictions are read from the LLM caches of predict_all_files_llm.py /
predict_all_files_llm_low_data_regimes.py, which live in the shared outputs/llm_cache (--llm-cache-dir):

    outputs/llm_cache/
        predict_all_train_gpt-5-nano_zero_shot/                      the zero-shot labels of all training sentences
        predict_all_test_gpt-5-nano_<k>_shot_<test file>_<p>/        the labels of the full test set for regime p
                                                                     (k = 2 examples per 10 % drawn from that regime)
The cache is only read, never copied or changed. (It is not in git, /outputs/ is ignored: copy it to the cluster by hand.)

Several runs (e.g. one sbatch job per group of datasets) can run at the same time without influencing each other:
every run has its own work folder (.work/<job id>) for the saved roberta-large models and the library's prediction
caches, and every result file carries the start time of the run in its name. The results live in ./outputs/predict_low_data_regimes/:
    <test file>_<p>_low_data_regimes_<date_time>_<job id>.csv / .json   per run: sentence, label, true, llm_pred, roberta_pred, fusion_pred, fusion_ts_pred
    summary_seeds_low_data_regimes_<date_time>_<job id>.csv             one row per run (all metrics, "training", parameters used)
    summary_low_data_regimes_<date_time>_<job id>.csv                   per dataset and regime: mean / std (ddof=1) over the seeds
    failures_low_data_regimes_<date_time>_<job id>.csv                  runs that raised an error (the sweep goes on)
    trials/...                                                 Optuna trials

JSON mode (--json <articles json>): instead of the Excel / CSV files the texts come from an articles JSON of
dataset_scripts/data_gen/article_creation.ipynb, read by dataset_scripts/json_dataset.py into the same table (sentence, label,
one-hot, date). For every seed in --json-seeds the JSON is split 80/20 (stratified, --json-test-size) into a training and a
test part, and the training part into the nested regimes (same algorithm as make_low_data_regimes.py); the runs are named
<json name>-test-<seed>_<p>. The LLM labels come from caches of their own,
    predict_all_train_<model>_zero_shot_<json name>/               the zero-shot labels of all texts of the JSON
    predict_all_test_<model>_<k>_shot_<json name>-test-<seed>_<p>/  the labels of the test part for regime p
which --make-llm-labels creates (OpenRouter, --llm-model-id; this calls the paid API, only for texts that are not cached yet).
The time series default to the synthetic inflation / unemployment series of data_gen.ipynb (--ts-file, --ts-series,
--ts-date-column). The per-run CSV/JSON also carries the columns filler and label_flipped of the JSON.

    python testing/predict_low_data_regimes.py --dry-run                                  # which runs, nothing is trained
    python testing/predict_low_data_regimes.py --json dataset_scripts/data/articles/jsons/<file>.json --dry-run
    python testing/predict_low_data_regimes.py --json dataset_scripts/data/articles/jsons/<file>.json --make-llm-labels --percents 10 50 100
    python testing/predict_low_data_regimes.py --datasets pc-split-test --percents 10 80  # a first look: two regimes
    python testing/predict_low_data_regimes.py                                            # everything
    python testing/predict_low_data_regimes.py --no-ts --mlp-trials 10 --full-percents 100   # without the time series, fewer trials
    python testing/predict_low_data_regimes.py --roberta-params-from outputs/labelfusion_roberta_large/summary_seeds.csv
"""

from __future__ import annotations

import argparse
import atexit
import collections
import contextlib
import copy
import dataclasses
import gc
import io
import hashlib
import json
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
sys.path.insert(0, str(TESTING_DIR.parent / "dataset_scripts"))  # json_dataset.py (JSON mode)

# Everything this script writes lives in RESULT_DIR. Every run gets its own work folder (.work/<job id>) there for the
# saved roberta-large models and the library's prediction caches (the library names them after the train texts, so two
# runs on the same data would otherwise write and delete the same directory); it is removed when the run ends.
RESULT_DIR = TESTING_DIR.parent / "outputs" / "predict_low_data_regimes"
RUN_ID = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
WORK_DIR = RESULT_DIR / ".work" / RUN_ID
os.environ["LABELFUSION_MODEL_CACHE"] = str(WORK_DIR / "models")  # set before the modules below read it
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)
SUFFIX = f"low_data_regimes_{datetime.now():%Y%m%d_%H%M%S}_{RUN_ID}"  # start time + job id: parallel jobs never share a file name

import predict_all_files_labelFusion as files  # noqa: E402
import predict_all_labelFusion as base  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import optuna  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from test_textclassify_llm import _add_onehot_labels  # noqa: E402
import json_dataset  # noqa: E402  (dataset_scripts/json_dataset.py: the articles JSON as a table)
from textclassify import RoBERTaLargeClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402
from textclassify.timeseries import CrossTSTransformer, TS, TSEmbedder  # noqa: E402

stage1 = base.stage1
REPO_ROOT = base.TEST_DIR.parent.parent
TRAIN_REGIME_DIR = REPO_ROOT / "data" / "training_data_dated_low_data_regimes"
TEST_DATED_DIR = REPO_ROOT / "data" / "test_data_dated"
LLM_CACHE_DIR = base.OUT_DIR / "llm_cache"  # the LLM caches (where predict_all_files_llm*.py wrote them; set from --llm-cache-dir)
LLM_MODEL = "gpt-5-nano"  # the model name in the cache folder names (set from --llm-model)
SHOTS_PER_10_PERCENT = 2
TS_FILE = REPO_ROOT / "data" / "3 Ts" / "daily_series.csv"
TS_SERIES = ["DGS10", "T10Y2Y", "VIXCLS"]
TS_DATE_COLUMN = "DATE"  # the date column of TS_FILE (set from --ts-file / --ts-series / --ts-date-column)
# JSON mode: the synthetic series the articles were generated from (data_gen.ipynb)
JSON_TS_FILE = REPO_ROOT / "dataset_scripts" / "data" / "stock_prices" / "full_series" / "full_baseline_inflation_unemployment_trial_seed7.csv"
JSON_TS_SERIES = ["inflation_price", "unemployment_price"]
JSON_TS_DATE_COLUMN = "date"
JSON_NAME = ""  # JSON mode: the name of the JSON in its runs and LLM cache folders (empty in file mode)
DATE_COLUMN = "date"
FULL_HISTORY = 10**6  # window_days that reaches back to the beginning of every series
WHO = ("llm", "roberta", "fusion", "fusion_ts", "fusion_joint", "fusion_joint_ts")
METRICS = [f"{who}_{m}" for who in WHO for m in ("f1_weighted", "f1_macro", "accuracy")]
REGIME_RE = re.compile(r"_(?P<percent>\d+)$")


# ---- time series branch (as in predict_all_labelFusion_TS_fixed_window.py) -------------------------------------------

class DailySeries(TS):
    """textclassify's TS loader on one CSV with a DATE column and one column per series."""

    def __init__(self, csv_path: Path, series_names: list[str], date_col: str = "DATE"):
        super().__init__(data_dir=csv_path.parent, stock_symbols=list(series_names), date_column="DATE", price_column="value")
        self.csv_path = csv_path
        self.date_col = date_col  # the file's date column; the loaded frames always call it DATE

    def load_all(self) -> None:
        raw = (pd.read_csv(self.csv_path, parse_dates=[self.date_col])
               .rename(columns={self.date_col: "DATE"}).sort_values("DATE"))
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
        self.loader = DailySeries(TS_FILE, TS_SERIES, TS_DATE_COLUMN)
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


# ---- checkpoints: the expensive calls of the run in progress survive an interruption ----------------------------------

CHECKPOINT_DIR = RESULT_DIR / "checkpoints"


def text_hash(df: pd.DataFrame) -> str:
    return hashlib.md5("\x1f".join(df[base.TEXT_COLUMN]).encode()).hexdigest()[:12]


class Checkpoint:
    """Results of the expensive calls of ONE run (a roberta-large Optuna trial, a fusion-MLP fit), kept in a json file that
    is rewritten after every new result (temp file + rename, so a kill never leaves half a file)."""

    def __init__(self):
        self.path: Path | None = None
        self.data: dict = {}
        self.hits = self.misses = 0

    def start(self, name: str, fingerprint: dict) -> None:
        self.path = CHECKPOINT_DIR / f"{name}.json"
        self.hits = self.misses = 0
        self.fingerprint = hashlib.md5(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()
        self.data = {}
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text())
                if saved.get("fingerprint") == self.fingerprint:
                    self.data = saved["results"]
                    print(f"### checkpoint: {len(self.data)} results of an interrupted run of {name} are reused")
                else:
                    print(f"### checkpoint of {name} was made with other settings or data: ignored")
            except (json.JSONDecodeError, KeyError, OSError):
                print(f"### checkpoint of {name} is unreadable: ignored")

    def get(self, key: str):
        if self.path is None or key not in self.data:
            return None
        self.hits += 1
        return self.data[key]

    def put(self, key: str, value) -> None:
        if self.path is None:
            return
        self.misses += 1
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "results": self.data}))
        os.replace(tmp, self.path)

    def finish(self, delete: bool) -> None:
        if delete and self.path is not None and self.path.exists():
            self.path.unlink()
        self.path = None


CKPT = Checkpoint()


def install_checkpoint_wrappers() -> None:
    """Route the two expensive calls through the checkpoint: a roberta-large trial (stage1.fit_predict, used by the Optuna
    objective) and a fusion-MLP fit (base.mlp_fit_predict, used by its Optuna objective and for the final fit)."""
    plain_stage1_fit_predict = stage1.fit_predict
    plain_mlp_fit_predict = base.mlp_fit_predict

    def stage1_fit_predict(params, train_df, val_df, pred_df, retries=0):
        key = f"roberta|{json.dumps(params, sort_keys=True, default=str)}|{text_hash(train_df)}|{text_hash(pred_df)}"
        cached = CKPT.get(key)
        if cached is not None:
            return cached
        preds = plain_stage1_fit_predict(params, train_df, val_df, pred_df, retries)
        CKPT.put(key, list(preds))
        return preds

    def mlp_fit_predict(roberta, llm, mlp, rob_train_df, mlp_train_df, pred_df, pred_llm, name, tag, verbose):
        key = f"mlp|{name}|{tag}|{json.dumps(mlp, sort_keys=True, default=str)}|{text_hash(mlp_train_df)}|{text_hash(pred_df)}"
        cached = CKPT.get(key)
        if cached is not None:
            return cached
        preds = plain_mlp_fit_predict(roberta, llm, mlp, rob_train_df, mlp_train_df, pred_df, pred_llm, name, tag, verbose)
        CKPT.put(key, list(preds))
        return preds

    stage1.fit_predict = stage1_fit_predict
    base.mlp_fit_predict = mlp_fit_predict


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
    # JSON mode: one zero-shot cache per JSON (its texts are not in the shared cache of the FOMC training files)
    return f"train_{LLM_MODEL}_zero_shot" + (f"_{JSON_NAME}" if JSON_NAME else "")


def regime_cache_name(test_stem: str, percent: int) -> str:
    return f"test_{LLM_MODEL}_{shots_for(percent)}_shot_{test_stem}_{percent}"


def find_cache(run_name: str) -> Path | None:
    """The cache folder of an LLM run in outputs/llm_cache (it must hold at least one cache_*.pkl)."""
    folder = LLM_CACHE_DIR / f"predict_all_{run_name}"
    return folder if folder.is_dir() and any(folder.glob("cache_*.pkl")) else None


def read_cache(folder: Path) -> dict[str, str]:
    """sentence -> predicted class name, from the classifier's prediction cache (only successful, unambiguous calls)."""
    preds: dict[str, str] = {}
    for pkl in sorted(folder.glob("cache_*.pkl")):
        try:
            with open(pkl, "rb") as fh:
                entries = pickle.load(fh)
        except (EOFError, pickle.UnpicklingError) as e:
            raise RuntimeError(f"{pkl} is empty or truncated ({pkl.stat().st_size} bytes, {type(e).__name__}): the copy to this machine "
                               f"was probably interrupted. Copy it again, e.g. rsync -av --checksum, and compare `ls -l` with the original.") from e
        for entry in entries.values():
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
            raise FileNotFoundError(f"no zero-shot cache predict_all_{train_cache_name()} with a cache_*.pkl in {LLM_CACHE_DIR}")
        _TRAIN_LABELS.update(read_cache(folder))
    return _TRAIN_LABELS


@dataclasses.dataclass
class Job:
    """One run: a regime of a training set and the full test set it is scored on. The tables come either from the
    Excel / CSV files (file mode) or from an articles JSON (JSON mode); everything after prepare() is the same."""
    test_stem: str                         # <dataset>-<seed>, e.g. lab-manual-mm-test-5768 or <json name>-test-5768
    percent: int
    regime: callable                       # () -> regime table (sentence, label, one-hot, date, ...)
    test: callable                         # () -> full test table
    fingerprint: str                       # changes when the regime's data change (checkpoint validity)
    source: str                            # where the regime comes from (for --dry-run)
    test_path: Path | None = None          # file mode only: the test file and the regime CSV
    train_path: Path | None = None

    @property
    def name(self) -> str:
        return f"{self.test_stem}_{self.percent}"

    @property
    def llm_cache(self) -> Path | None:
        return find_cache(regime_cache_name(self.test_stem, self.percent))

    # The scripts built on this one (testing/old/predict_low_data_regimes_logit.py, ..._llm_mlp.py,
    # roberta_llm_label_compare.py) use the jobs of file mode as (test file, regime file, LLM cache, percent) tuples.
    def _legacy(self) -> tuple:
        return (self.test_path, self.train_path, self.llm_cache, self.percent)

    def __iter__(self):
        return iter(self._legacy())

    def __getitem__(self, i):
        return self._legacy()[i]


def file_job(test_path: Path, train_path: Path, percent: int) -> Job:
    """The Job of one test file and one regime CSV (file mode)."""
    return Job(test_stem=test_path.stem, percent=percent,
               regime=lambda: load_regime(train_path), test=lambda: load_test(test_path),
               fingerprint=hashlib.md5(train_path.read_bytes()).hexdigest(), source=train_path.name,
               test_path=test_path, train_path=train_path)


def jobs(args) -> list[Job]:
    """A Job for every run that can be made from the files (test file, regime CSV, the regime's LLM cache)."""
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
            found.append(file_job(test_path, train_path, percent))
    return sorted(found, key=lambda j: (j.test_stem, j.percent))


JSON_KEEP = json_dataset.MODEL_COLUMNS + ["obs_id", "filler", "label_flipped"]  # the models' table + columns for split and results


def json_jobs(args) -> tuple[list[Job], pd.DataFrame]:
    """JSON mode: a Job for every seed in --json-seeds and every regime, from the articles JSON (no files are written).
    Per seed: stratified train / test split, then the nested regimes of the training part. Also returns the whole table
    (its texts are what the zero-shot LLM cache must cover)."""
    df = json_dataset.read_articles_json(args.json, drop_meta_text=args.json_drop_meta_text)[JSON_KEEP]
    percents = sorted(args.percents) if args.percents else list(range(10, 101, 10))
    found = []
    for seed in args.json_seeds:
        train_df, test_df = json_dataset.split_train_test(df, test_size=args.json_test_size, seed=seed)
        test_df = test_df[JSON_KEEP]
        for percent, regime_df in json_dataset.low_data_regimes(train_df, percents=percents, seed=seed).items():
            regime_df = regime_df[JSON_KEEP]
            found.append(Job(test_stem=f"{JSON_NAME}-test-{seed}", percent=percent,
                             regime=lambda r=regime_df: r.copy(), test=lambda t=test_df: t.copy(),
                             fingerprint=hashlib.md5(regime_df.to_csv(index=False).encode()).hexdigest(),
                             source=f"{Path(args.json).name}, seed {seed}, {len(regime_df)} of {len(train_df)} training texts"))
    return found, df


def make_llm_labels(todo: list[Job], all_texts: pd.DataFrame, args) -> None:
    """JSON mode, --make-llm-labels: create the missing LLM labels via OpenRouter, as predict_all_files_llm.py and
    predict_all_files_llm_low_data_regimes.py do for the FOMC files -- zero-shot for all texts of the JSON, few-shot
    (2 examples per 10 %, drawn from the regime) for the test part of every regime. Texts already in a cache are not
    sent again (the classifier's own prediction cache)."""
    import predict_all_files_llm as llm_files  # OpenRouter classifier with the cache layout read by this script

    known = read_cache(find_cache(train_cache_name())) if find_cache(train_cache_name()) else {}
    zero = all_texts[~all_texts[base.TEXT_COLUMN].isin(known)][json_dataset.MODEL_COLUMNS]
    few = []
    for job in todo:
        cached = read_cache(job.llm_cache) if job.llm_cache else {}
        n_missing = int((~job.test()[base.TEXT_COLUMN].isin(cached)).sum())
        if n_missing:
            few.append((job, n_missing))
    calls = len(zero) + sum(n for _, n in few)
    print(f"--make-llm-labels: {len(zero)} zero-shot texts + {sum(n for _, n in few)} few-shot test texts "
          f"({len(few)} regimes) = {calls} calls to {args.llm_model_id}")
    if len(zero):
        llm_files.build(args.llm_model_id, train_cache_name(), 0).predict(train_df=None, test_df=zero)
    for job, _ in few:
        regime = job.regime()
        k = min(shots_for(job.percent), len(regime))
        shots = regime.sample(n=k, random_state=base.RANDOM_STATE)[json_dataset.MODEL_COLUMNS].reset_index(drop=True)
        print(f"[{job.name}] {k} examples from the regime ({len(regime)} texts)")
        clf = llm_files.build(args.llm_model_id, regime_cache_name(job.test_stem, job.percent), k)
        clf.fit(shots)
        clf.predict(train_df=shots, test_df=job.test()[json_dataset.MODEL_COLUMNS])
    _TRAIN_LABELS.clear()  # re-read the zero-shot cache


# ---- one run ----------------------------------------------------------------------------------------------------------

class ConstantLLM(base.PrecomputedLLM):
    """An 'LLM' that always says the same: its one-hot carries no information, so the fusion MLP effectively sees the
    roberta-large embedding alone."""

    def __init__(self):
        super().__init__(collections.defaultdict(lambda: base.LABEL_COLUMNS[-1]))


def prepare(job: Job | Path, train_path: Path | None = None, llm_path: Path | None = None, percent: int | None = None):
    """The regime, the full test set, the LLM labels, the start of the output frame and the summary row.
    Takes a Job, or (as the scripts built on this one call it) test file, regime file, LLM cache and percent."""
    if not isinstance(job, Job):
        job = file_job(job, train_path, percent)
    name, percent = job.name, job.percent
    regime_df = job.regime()
    test_df = job.test()
    llm_path = job.llm_cache
    if llm_path is None:
        raise FileNotFoundError(f"no LLM cache predict_all_{regime_cache_name(job.test_stem, percent)} in {LLM_CACHE_DIR}"
                                + (" (create it with --make-llm-labels)" if JSON_NAME else ""))
    train_labels_ = train_labels()
    missing = [t for t in regime_df[base.TEXT_COLUMN] if t not in train_labels_]
    if missing:
        raise KeyError(f"{len(missing)} sentences of the regime have no zero-shot label in the LLM cache")
    llm_train = base.PrecomputedLLM(train_labels_)
    llm_test = read_cache(llm_path)  # the regime's labels of the test sentences
    missing = [t for t in test_df[base.TEXT_COLUMN] if t not in llm_test]
    if missing:
        raise KeyError(f"{len(missing)} test sentences have no label in the LLM cache {llm_path.name}")
    llm_test_preds = [llm_test[t] for t in test_df[base.TEXT_COLUMN]]
    out = pd.DataFrame({
        base.TEXT_COLUMN: test_df[base.TEXT_COLUMN], "label": test_df["label"], "true": test_df["label"].map(base.LABEL_MAP),
        "llm_pred": llm_test_preds,
    })
    for col in ("filler", "label_flipped"):  # JSON mode: which test texts were content-free / have a wrong label
        if col in test_df:
            out[col] = test_df[col].values
    m = base.SEED_RE.match(job.test_stem)
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


def run_job(job: Job, args, roberta, branches) -> tuple[dict, pd.DataFrame]:
    """FROZEN regime: only MLPs are trained, on the whole regime."""
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = prepare(job)
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


def run_job_full(job: Job, args, branches) -> tuple[dict, pd.DataFrame]:
    """FULL regime: the program of predict_all_labelFusion.py (+ the time series model) on the regime."""
    t0 = time.time()
    name, regime_df, test_df, llm_train, llm_test_preds, out, row = prepare(job)
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

    # Two-stage fusion (only with --two-stage): the fine-tuned roberta-large is frozen, the fusion MLPs are tuned and trained on the
    # validation split. fusion: + LLM label. fusion_ts: additionally the FINE-TUNED time series model (TimesFM encoders +
    # CrossTSTransformer, trained with the MLP); far more expensive per MLP trial.
    models = []
    if args.two_stage:
        models.append(("fusion", branches["plain"], args.mlp_trials))
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
    # the weights of the fine-tuned roberta-large (CPU copy): the joint models start from them (--joint-init finetuned)
    fine_tuned_state = ({k: v.detach().cpu().clone() for k, v in roberta.model.state_dict().items()}
                        if args.joint and args.joint_init == "finetuned" else None)
    del roberta, memo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Joint models (the LabelFusion models of the full regimes), trained end to end on the train split and chosen on the validation split:
    #   fusion_joint     roberta-large + LLM one-hot -> fusion MLP
    #   fusion_joint_ts  the same + the time series model (TimesFM encoders + CrossTSTransformer), all trained together
    if args.joint:
        row["joint_init"] = args.joint_init
        joint_models = [("fusion_joint", branches["plain"])]
        if "ts_finetuned" in branches:
            joint_models.append(("fusion_joint_ts", branches["ts_finetuned"]))
        for who, build in joint_models:
            preds, jp, jv = tune_joint(f"{name}_{who}", train_df, val_df, test_df, llm_train, llm_test_preds, args, build, fine_tuned_state)
            out[f"{who}_pred"] = preds
            row.update({f"{who}_ml_lr": jp["ml_lr"], f"{who}_mlp_lr": jp["fusion_lr"], f"{who}_epochs": jp["epochs"],
                        f"{who}_batch_size": jp["batch_size"], f"{who}_hidden": jp["hidden_dims"], f"{who}_val_f1": jv})
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


def build_frozen_roberta(model_name: str, memoise: bool = True):
    """The pretrained roberta-large, never fine-tuned, with its embeddings memoised per sentence for the whole sweep
    (FusionEnsemble asks for them on every fit() / predict()). Without the memo (joint training) its weights change
    during training, so nothing may be cached."""
    config = ModelConfig(model_name="roberta-large", model_type=ModelType.TRADITIONAL_ML,
                         parameters={"model_name": model_name, "max_length": 128, "batch_size": 32})
    roberta = RoBERTaLargeClassifier(config=config, text_column=base.TEXT_COLUMN, label_columns=base.LABEL_COLUMNS,
                                     multi_label=False, auto_save_results=False, cache_dir=str(WORK_DIR / "models"))
    roberta.load_pretrained()
    if memoise:
        base.EmbeddingMemo(roberta)
    return roberta


def joint_trial(params: dict, name: str, tag: str, train_df, val_df, test_df, llm_train, llm_test_preds, args, build, init_state=None) -> dict:
    """One end-to-end training: roberta-large and the fusion MLP trained together on train_df; returns the predictions for the
    validation split and for the test set. roberta-large starts from init_state (the weights fine-tuned on train_df in stage 1,
    --joint-init finetuned) or, without it, from the pretrained model. Checkpointed like the other expensive calls."""
    key = f"joint|{name}|{json.dumps(params, sort_keys=True, default=str)}|{text_hash(train_df)}|{text_hash(val_df)}"
    cached = CKPT.get(key)
    if cached is not None:
        return cached
    fus = {"fusion_hidden_dims": base.HIDDEN_DIMS[params["hidden_dims"]], "fusion_lr": params["fusion_lr"],
           "ml_lr": params["ml_lr"], "num_epochs": params["epochs"], "batch_size": params["batch_size"], "joint_training": True}
    if args.ml_proj_dim:
        fus["ml_proj_dim"] = args.ml_proj_dim
    roberta = build_frozen_roberta(args.roberta_model, memoise=False)
    if init_state is not None:
        roberta.model.load_state_dict(init_state)  # start from the roberta-large fine-tuned on this regime's train split
    fusion = build(roberta, llm_train, fus, name, tag, False)
    with contextlib.redirect_stdout(io.StringIO()):  # the library's progress prints
        fusion.fit(train_df, train_df)  # roberta counts as trained -> the fusion trains on train_df, roberta-large with it
        val_preds = fusion.predict(val_df, train_df=train_df, test_llm_predictions=llm_train.lookup(val_df)).predictions
        test_preds = fusion.predict(test_df, train_df=train_df, test_llm_predictions=llm_test_preds).predictions
    del fusion, roberta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"val": list(val_preds), "test": list(test_preds)}
    CKPT.put(key, result)
    return result


def tune_joint(name: str, train_df, val_df, test_df, llm_train, llm_test_preds, args, build, init_state=None) -> tuple[list, dict, float]:
    """Optuna over the joint model: every trial trains roberta-large + fusion MLP on train_df and is scored by weighted F1 on
    val_df. Returns the test predictions of the best trial, its parameters and its validation F1."""
    val_true = val_df["label"].map(base.LABEL_MAP).tolist()
    results = {}
    print(f"### joint training on {name}: train {len(train_df)} / val {len(val_df)}, {args.joint_trials} trials")

    def objective(trial: optuna.Trial) -> float:
        params = {"ml_lr": trial.suggest_float("ml_lr", 1e-6, 5e-5, log=True),
                  "fusion_lr": trial.suggest_float("fusion_lr", 1e-4, 1e-2, log=True),
                  "epochs": trial.suggest_int("epochs", 2, args.joint_max_epochs),
                  "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                  "hidden_dims": trial.suggest_categorical("hidden_dims", list(base.HIDDEN_DIMS))}
        result = joint_trial(params, name, f"joint{trial.number}", train_df, val_df, test_df, llm_train, llm_test_preds, args, build, init_state)
        results[trial.number] = result
        return f1_score(val_true, result["val"], average="weighted")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=base.RANDOM_STATE))
    study.enqueue_trial({"ml_lr": 5e-6 if init_state is not None else 1e-5, "fusion_lr": 1e-3, "epochs": min(4, args.joint_max_epochs), "batch_size": 16, "hidden_dims": "64-32"})
    study.optimize(objective, n_trials=args.joint_trials)

    path = base.RESULT_DIR / "trials" / f"{name}_joint_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(attrs=("number", "value", "params", "state", "duration")).rename(
        columns={"value": "val_f1_weighted"}).to_csv(path, index=False)
    best = study.best_trial
    print(f"### best joint weighted F1 {study.best_value:.3f} after {len(study.trials)} trials: {best.params}")
    return results[best.number]["test"], best.params, study.best_value


def main() -> None:
    global LLM_CACHE_DIR, LLM_MODEL, TS_FILE, TS_SERIES, TS_DATE_COLUMN, JSON_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path,
                        help="JSON mode: an articles JSON of article_creation.ipynb instead of the Excel / CSV files (see the docstring)")
    parser.add_argument("--json-name", help="JSON mode: name of the dataset in the run names and LLM cache folders (default: the JSON's file name)")
    parser.add_argument("--json-seeds", nargs="*", type=int, default=[5768],
                        help="JSON mode: one train / test split (and set of regimes) per seed (default: 5768)")
    parser.add_argument("--json-test-size", type=float, default=0.2, help="JSON mode: share of the texts in the test part")
    parser.add_argument("--json-drop-meta-text", action="store_true",
                        help="JSON mode: drop the few texts in which the generator left notes about its own answer")
    parser.add_argument("--make-llm-labels", action="store_true",
                        help="JSON mode: create the missing LLM labels via OpenRouter before the runs (calls the paid API, "
                             "only for texts that are not cached yet)")
    parser.add_argument("--llm-model-id", help="JSON mode, --make-llm-labels: OpenRouter id of the LLM (default: openai/<--llm-model>)")
    parser.add_argument("--ts-file", type=Path, help="time series CSV (default: data/3 Ts/daily_series.csv; JSON mode: the synthetic "
                                                     "inflation / unemployment series of data_gen.ipynb)")
    parser.add_argument("--ts-series", nargs="*", help="the series columns of --ts-file used by the time series branch")
    parser.add_argument("--ts-date-column", help="the date column of --ts-file")
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--llm-cache-dir", type=Path, default=LLM_CACHE_DIR,
                        help="the LLM caches (default: outputs/llm_cache)")
    parser.add_argument("--llm-model", default=LLM_MODEL,
                        help="the LLM whose cache folders are read (name as in the folder names, e.g. gpt-5-nano)")
    parser.add_argument("--dry-run", action="store_true", help="list the runs, nothing is trained")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip runs that already have a result file in the output folder (default; --no-resume repeats them)")
    parser.add_argument("--mlp-trials", type=int, default=30,
                        help="Optuna trials per run and model tuning the fusion MLP; 0 = the MLP's default configuration")
    parser.add_argument("--full-percents", nargs="*", type=int, default=None,
                        help="regimes that get the full program (fine-tuned roberta-large with Optuna, warmup, class weights, "
                             "best epoch, retries; MLPs on the validation split). Default: every regime; give no value for none")
    parser.add_argument("--two-stage", action="store_true",
                        help="full regimes: also compute the two-stage fusion models fusion / fusion_ts (frozen fine-tuned roberta-large, "
                             "MLP on the validation split). Off by default: the fusion models of the full regimes are the joint ones")
    parser.add_argument("--joint", action=argparse.BooleanOptionalAction, default=True,
                        help="full regimes: train roberta-large, the LLM one-hot and the fusion MLP TOGETHER, end to end, on the train split "
                             "(columns fusion_joint, and fusion_joint_ts with the time series model). On by default; --no-joint skips them. "
                             "Every trial is a complete fine-tuning")
    parser.add_argument("--joint-trials", type=int, default=12, help="Optuna trials of the joint model per run")
    parser.add_argument("--joint-max-epochs", type=int, default=8, help="upper bound of the joint model's epochs in the search")
    parser.add_argument("--joint-init", choices=("finetuned", "pretrained"), default="finetuned",
                        help="where roberta-large starts in the joint models: from the roberta-large fine-tuned on the train split in "
                             "stage 1 (default), or from the pretrained model")
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
    parser.add_argument("--ml-proj-dim", type=int, default=32,
                        help="project roberta-large's 1024-dim [CLS] embedding down to this size before the MLP (default 32: a small bottleneck gives the 3-dim LLM label more weight); 0 disables")
    parser.add_argument("--roberta-model", default="roberta-large", help="the (frozen) encoder; roberta-large unless testing")
    parser.add_argument("--no-ts", action="store_true", help="skip the fusion model with the time series branch")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="number of daily values of every series up to the sentence's date; 0 = the whole series")
    parser.add_argument("--ts-pooling", choices=("mean", "max"), default="mean", help="pooling of the three series embeddings")
    args = parser.parse_args()
    # absolute before main() changes into the work folder
    for key in ("llm_cache_dir", "roberta_params_from", "json", "ts_file"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())

    LLM_CACHE_DIR, LLM_MODEL = args.llm_cache_dir, args.llm_model
    if args.json:  # JSON mode: texts from the articles JSON, the synthetic series as time series
        JSON_NAME = args.json_name or args.json.stem
        TS_FILE = args.ts_file or JSON_TS_FILE
        TS_SERIES = args.ts_series or JSON_TS_SERIES
        TS_DATE_COLUMN = args.ts_date_column or JSON_TS_DATE_COLUMN
        args.llm_model_id = args.llm_model_id or f"openai/{LLM_MODEL}"
        todo, all_texts = json_jobs(args)
        print(f"JSON mode: {args.json.name} as '{JSON_NAME}', {len(all_texts)} texts, seeds {args.json_seeds}, "
              f"time series {', '.join(TS_SERIES)} from {TS_FILE.name}")
    else:
        if args.make_llm_labels:
            raise SystemExit("--make-llm-labels only works in JSON mode (--json); in file mode the caches come from "
                             "predict_all_files_llm.py / predict_all_files_llm_low_data_regimes.py")
        TS_FILE = args.ts_file or TS_FILE
        TS_SERIES = args.ts_series or TS_SERIES
        TS_DATE_COLUMN = args.ts_date_column or TS_DATE_COLUMN
        todo = jobs(args)
    if not todo:
        raise SystemExit("no runs found (are the regimes of dataset_scripts/make_low_data_regimes.py and the LLM "
                         "LLM caches of predict_all_files_llm_low_data_regimes.py there?)")
    if args.full_percents is None:  # default: the full program at every regime
        args.full_percents = sorted({j.percent for j in todo})
    if args.resume:
        done = {re.sub(r"_low_data_regimes_\d{8}_\d{6}(_\w+)?$", "", p.stem) for p in RESULT_DIR.glob("*_low_data_regimes_*.csv")
                if not p.name.startswith(("summary", "failures"))}
        todo = [j for j in todo if j.name not in done]

    if args.json and args.make_llm_labels and not args.dry_run and todo:
        make_llm_labels(todo, all_texts, args)
    if find_cache(train_cache_name()) is None:
        msg = f"no zero-shot cache predict_all_{train_cache_name()} with a cache_*.pkl in {LLM_CACHE_DIR}"
        if not args.dry_run:
            raise SystemExit(msg + (" -- create it with --make-llm-labels" if args.json else ""))
        print(f"WARNING: {msg}")
    else:
        try:
            print(f"zero-shot LLM labels of the training sentences: {len(train_labels())} (from {find_cache(train_cache_name())})")
        except RuntimeError as e:  # an empty / truncated pickle would otherwise fail every single run in the same way
            raise SystemExit(f"ERROR: {e}")

    print(f"{len(todo)} runs ({', '.join(sorted({str(j.percent) + '%' for j in todo}, key=lambda s: int(s[:-1])))}) "
          f"on {len({j.test_stem for j in todo})} test files; models: roberta, fusion{'' if args.no_ts else ', fusion_ts'} "
          f"(full program at {', '.join(str(p) + '%' for p in args.full_percents) or 'no regime'}, frozen encoders + "
          f"MLP tuned with {args.mlp_trials} Optuna trials at the others)")
    if args.dry_run:
        for job in todo[:10]:
            cache = job.llm_cache
            print(f"  {job.name} ({'full program' if job.percent in args.full_percents else 'frozen'}): "
                  f"train {job.source}, LLM cache {cache.name if cache else 'MISSING predict_all_' + regime_cache_name(job.test_stem, job.percent)}")
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

    install_checkpoint_wrappers()
    rows, failures = [], []
    for job in todo:
        shutil.rmtree(WORK_DIR / "cache", ignore_errors=True)
        name, percent = job.name, job.percent
        CKPT.start(name, {"regime": job.fingerprint, "test": job.test_stem, "percent": percent,
                          "full": percent in args.full_percents, "epochs": args.epochs, "ml_proj_dim": args.ml_proj_dim,
                          "llm_model": LLM_MODEL, "ts": [args.ts_window, args.ts_pooling, args.ts_dim], "no_ts": args.no_ts, "joint": [args.joint, args.joint_max_epochs, args.joint_init], "two_stage": args.two_stage, "v": 1})
        try:
            if percent in args.full_percents:
                row, out = run_job_full(job, args, branches)
            else:
                if "roberta" not in frozen:
                    frozen["roberta"] = build_frozen_roberta(args.roberta_model)
                row, out = run_job(job, args, frozen["roberta"], branches)
        except Exception as e:  # one failing run must not stop the sweep
            traceback.print_exc()
            CKPT.finish(delete=False)  # keep what was computed: the next start of this run continues from it
            failures.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            pd.DataFrame(failures).to_csv(RESULT_DIR / f"failures_{SUFFIX}.csv", index=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        out.to_csv(staging / f"{name}.csv", index=False)
        out.to_json(staging / f"{name}.json", orient="records", force_ascii=False, indent=2)
        publish(staging)
        print(f"### {name}: {CKPT.hits} results from the checkpoint, {CKPT.misses} newly computed")
        CKPT.finish(delete=True)  # the run is finished and published: its checkpoint is not needed any more
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
