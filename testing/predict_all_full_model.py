"""LLM predictions + roberta-large embeddings fused by an MLP, on fixed-test training regimes.

For every text of an articles JSON (dataset_scripts/article_creation.ipynb, read by dataset_scripts/json_dataset.py):

  embedding   the [CLS] embedding of roberta-large. LEARNABLE by default: roberta-large is fine-tuned together with the MLP head
              inside every model that uses it (--roberta-lr, --ft-*; each training starts from the pretrained weights).
              --no-finetune-roberta: the pretrained, FROZEN encoder instead (embeddings computed once, cached on disk)
  LLM label   the zero-shot label of GPT-5 nano (one OpenRouter call per text, cached; missing ones are created unless
              --no-make-llm-labels). The same zero-shot setting for training and test texts, so the fusion is trained on the
              kind of LLM label it later sees.
  time series the DAILY real series of data/TSs_macro (default DFF, DGS10, DGS2, NASDAQCOM): the last --ts-window observations up to the
              date of the text, one frozen TimesFM encoder embedding per series, mean-pooled over the series (computed once,
              cached on disk). Switch off with --no-ts.

The data is split once into a stratified training set and a held-out test set. Each model is then trained
on 10, 20, ..., 100 percent of the training set and evaluated on the same full test set.

  llm          the LLM label itself (no training)
  logreg       TF-IDF + logistic regression (standard settings), the text-only reference
  xgboost      TF-IDF + XGBoost with the fixed baseline configuration
  roberta_mlp  MLP on the roberta-large embedding alone (the LLM input is zeroed)
  fusion_mlp   MLP on the roberta-large embedding + the LLM label as one-hot: LabelFusion
  fusion_ts_mlp  the same plus the time series embedding (LabelFusion with the time series branch; not with --no-ts)
  fusion_ts_cross_mlp  the same, but the series embeddings are not just averaged: a TRAINABLE CrossTSTransformer (attention
                 over the series, see textclassify/timeseries/CrossTSTransformer.py) turns them into the time series
                 embedding. The TimesFM encoders stay FROZEN (their embeddings are cached); only the transformer and the
                 MLP are trained (not with --no-ts or --no-ts-cross)

The MLPs have the same architecture (as textclassify's FusionWrapper), TWO layers: the 1024-dim roberta [CLS] embedding
(raw) is concatenated with the 3-dim LLM one-hot (and, for the time series models, the time series embedding projected to
--ts-proj-dim (default 32) by Linear + ReLU) = 1027 values -> Linear(--hidden, default 8) + ReLU + Dropout -> Linear(3 classes).
(--proj-dim N would first project the roberta embedding to N dims.) Trained with AdamW and cross-entropy on 90 % of each
regime, early stopping on the other 10 %
(--patience); the fixed test set is never used for any choice.

Scores: accuracy, macro-F1, and weighted-F1 per training regime, also split by text type (clean content texts,
content-free fillers with a random label, content texts with a deliberately flipped label).

Writes to ./outputs/predict_all_full_model/ (repo root; <name> gets the suffix _finetuned when roberta learns):
    <name>_regime_predictions.csv  one row per test text, model, and training percentage
    <name>_regime_scores.csv       one row per training percentage, model, and text type
    embeddings/<model>_<hash>.npy   the cached embeddings (keyed by model and texts)
    embeddings/tsx_<series>_w<window>_<hash>.npy   the cached per-series time series embeddings (texts x series x 1280;
                                                   keyed by series, window and dates)

    python testing/predict_all_full_model.py                        # missing LLM labels via OpenRouter (key from .env), then the rest
    python testing/predict_all_full_model.py --no-make-llm-labels   # never call the API: stop if LLM labels are missing
    python testing/predict_all_full_model.py --json data/<articles>.json --roberta-model roberta-base --limit 300   # smoke test
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import pickle
import sys
import time
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent
sys.path.insert(0, str(REPO_ROOT))                        # textclassify
sys.path.insert(0, str(TESTING_DIR / "old"))              # predict_all_files_llm.py (OpenRouter classifier, cache layout)
sys.path.insert(0, str(REPO_ROOT / "dataset_scripts"))   # json_dataset.py

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import json_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")                           # OPENROUTER_API_KEY for the LLM labels

LABELS = list(json_dataset.LABEL_MAP.values())           # dovish, hawkish, neutral (index = label id)
TEXT = json_dataset.TEXT_COLUMN
OUT_DIR = REPO_ROOT / "outputs" / "predict_all_full_model"
LLM_CACHE_DIR = REPO_ROOT / "outputs" / "llm_cache"
DATA_DIR = REPO_ROOT / "data"                            # default: the newest articles JSON here
TS_DIR = DATA_DIR / "TSs_macro"                          # <series>.csv with the columns Date,value
TS_SERIES = ["DFF", "DGS10", "DGS2", "NASDAQCOM"]        # the DAILY series (fed funds rate, 10-year and 2-year Treasury yield, Nasdaq);
                                                         # the monthly ones are not used. DGS2 starts 1976-06 and NASDAQCOM 1971-02, see
                                                         # series_windows(early=...) for the texts dated before that
TS_MODEL = "google/timesfm-2.5-200m-transformers"
MODELS = ("llm", "logreg", "xgboost", "roberta_mlp", "fusion_mlp")
TS_MODEL_NAME = "fusion_ts_mlp"
TS_CROSS_MODEL_NAME = "fusion_ts_cross_mlp"
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
}


# ---- LLM labels -------------------------------------------------------------------------------------------------------

def llm_cache_folder(llm_model: str, name: str) -> Path:
    """The zero-shot cache of this JSON -- the same folder predict_low_data_regimes.py --json uses."""
    return LLM_CACHE_DIR / f"predict_all_train_{llm_model}_zero_shot_{name}"


def read_llm_cache(folder: Path) -> dict[str, int]:
    """sentence -> predicted label id, from the classifier's prediction cache (only successful, unambiguous calls)."""
    preds: dict[str, int] = {}
    for pkl in sorted(folder.glob("cache_*.pkl")):
        with open(pkl, "rb") as fh:
            entries = pickle.load(fh)
        for entry in entries.values():
            vec = entry.get("prediction")
            if entry.get("success") and vec is not None and sum(vec) == 1:
                preds[entry["text"]] = int(list(vec).index(1))
    return preds


def llm_labels(df: pd.DataFrame, args) -> np.ndarray:
    folder = llm_cache_folder(args.llm_model, args.name)
    cached = read_llm_cache(folder) if folder.is_dir() else {}
    missing = df[~df[TEXT].isin(cached)]
    if len(missing) and args.make_llm_labels:
        import predict_all_files_llm as llm_files  # OpenRouter; writes the cache into `folder`
        print(f"LLM: {len(missing)} texts without a label -> {len(missing)} calls to {args.llm_model_id} (zero-shot)")
        llm_files.build(args.llm_model_id, folder.name.removeprefix("predict_all_"), 0).predict(
            train_df=None, test_df=missing[json_dataset.MODEL_COLUMNS].reset_index(drop=True))
        cached = read_llm_cache(folder)
        missing = df[~df[TEXT].isin(cached)]
    if len(missing):
        raise SystemExit(f"{len(missing)} of {len(df)} texts have no LLM label in {folder} -- run without --no-make-llm-labels "
                         "(calls the paid OpenRouter API for the missing texts only)")
    return df[TEXT].map(cached).to_numpy()


# ---- roberta-large embeddings -----------------------------------------------------------------------------------------

def embeddings(df: pd.DataFrame, args) -> np.ndarray:
    """[CLS] embeddings of the frozen, pretrained encoder, cached per model and set of texts."""
    key = hashlib.md5(("\x1f".join(df[TEXT]) + f"|{args.max_length}").encode()).hexdigest()[:12]
    path = OUT_DIR / "embeddings" / f"{args.roberta_model.replace('/', '_')}_{key}.npy"
    if path.exists():
        print(f"embeddings: {path.name} (cached)")
        return np.load(path)
    from textclassify import RoBERTaLargeClassifier
    from textclassify.core.types import ModelConfig, ModelType
    print(f"embeddings: {args.roberta_model} on {len(df)} texts (once, then cached) ...")
    t0 = time.time()
    config = ModelConfig(model_name=args.roberta_model, model_type=ModelType.TRADITIONAL_ML,
                         parameters={"model_name": args.roberta_model, "max_length": args.max_length, "batch_size": 32})
    roberta = RoBERTaLargeClassifier(config=config, text_column=TEXT, label_columns=LABELS, multi_label=False,
                                     auto_save_results=False, cache_dir=str(OUT_DIR / "models"))
    roberta.load_pretrained()
    emb = np.stack(roberta.predict_without_saving(df[json_dataset.MODEL_COLUMNS]).embeddings).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"embeddings: {emb.shape} in {time.time() - t0:.0f}s -> {path.name}")
    return emb


# ---- time series embeddings (data/TSs_macro) ---------------------------------------------------------------------------

def series_windows(path: Path, dates: np.ndarray, window: int, early: str = "first") -> tuple[list[np.ndarray], int, int]:
    """For every date the last `window` observations of the series up to and including that date (window 0 = all of them).
    Returns (windows, n_short, n_early).

    Where fewer than `window` observations exist yet (the first weeks of the series), the window is simply the maximum
    available (shorter; TimesFM pads and masks it): n_short counts these texts.
    Dates BEFORE the series' first observation have no data at all. early="first" gives them the first `window`
    observations of the series -- the earliest data there is, i.e. data from AFTER the text's date (a look-ahead, only
    sensible because the dates of the generated texts are synthetic); n_early counts these texts. early="error" stops instead.

    The window counts observations, not calendar days: with the daily series (DFF every calendar day, DGS10 / DGS2 /
    NASDAQCOM on business days) 60 = about two months (DFF) or twelve weeks. A monthly series would count months."""
    raw = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").ffill()       # gaps (holidays): the last known value
    raw = raw.dropna(subset=["value"])
    stamps, values = raw["Date"].to_numpy().astype("datetime64[ns]"), raw["value"].to_numpy(np.float32)
    ends = np.searchsorted(stamps, dates.astype("datetime64[ns]"), side="right")   # observations <= date
    before_start = ends == 0
    if before_start.any():
        if early != "first":
            raise SystemExit(f"{path.name} starts at {pd.Timestamp(stamps[0]).date()}, after {int(before_start.sum())} texts "
                             f"(the first is {pd.Timestamp(dates.min()).date()}); use --ts-early first or other --ts-series")
        ends = np.where(before_start, min(window, len(values)) if window else len(values), ends)
    windows = [values[max(0, e - window) if window else 0:e] for e in ends]
    n_short = int(sum(len(w) < window for w, early_text in zip(windows, before_start) if not early_text)) if window else 0
    return windows, n_short, int(before_start.sum())


def ts_embeddings(df: pd.DataFrame, args) -> np.ndarray:
    """One frozen TimesFM embedding per series and text (identical windows are embedded once): (texts, series, 1280).
    fusion_ts_mlp averages it over the series, fusion_ts_cross_mlp feeds it to the trainable CrossTSTransformer."""
    dates = pd.to_datetime(df[json_dataset.DATE_COLUMN]).to_numpy()
    key = hashlib.md5(("\x1f".join(df[json_dataset.DATE_COLUMN]) + f"|{args.ts_model}").encode()).hexdigest()[:12]
    path = OUT_DIR / "embeddings" / f"tsx_{'-'.join(args.ts_series)}_w{args.ts_window}_{key}.npy"
    if path.exists():
        print(f"time series embeddings: {path.name} (cached)")
        return np.load(path)
    from textclassify.timeseries import TSEmbedder
    print(f"time series embeddings: {args.ts_model} on {', '.join(args.ts_series)} (window {args.ts_window or 'full'}) ...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TSEmbedder(args.ts_model, pooling="mean").to(device).eval()
    per_series = []
    for name in args.ts_series:
        windows, n_short, n_early = series_windows(args.ts_dir / f"{name}.csv", dates, args.ts_window, args.ts_early)
        unique: dict[bytes, int] = {}
        index = np.array([unique.setdefault(w.tobytes(), len(unique)) for w in windows])
        first = {i: k for k, i in reversed(list(enumerate(index)))}           # unique id -> first text with that window
        todo = [windows[first[i]] for i in range(len(unique))]
        out = []
        for start in range(0, len(todo), 64):
            with torch.no_grad():
                out.append(encoder(todo[start:start + 64])["embeddings"].float().cpu().numpy())
        per_series.append(np.concatenate(out)[index])
        print(f"  {name}: {len(unique)} distinct windows for {len(windows)} texts"
              + (f"; {n_short} with a shorter window (the series had fewer observations yet)" if n_short else "")
              + (f"; {n_early} dated before the series starts -> its first {args.ts_window or 'all'} observations (look-ahead)"
                 if n_early else ""))
    emb = np.stack(per_series, axis=1).astype(np.float32)             # (texts, series, hidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"time series embeddings: {emb.shape} in {time.time() - t0:.0f}s -> {path.name}")
    return emb


# ---- the fusion MLP ---------------------------------------------------------------------------------------------------

class CrossTSHead(nn.Module):
    """Trainable time series branch on the FROZEN, cached TimesFM embeddings.

    (B, series, 1280) -> Linear(1280 -> hidden), shared by the series -> CrossTSTransformer (learnable series positions,
    attention over the series, mean pool, projection) -> (B, out_dim).
    The 1280 -> hidden projection keeps the transformer small (the default CrossTSTransformer at 1280 dimensions has
    ~24M parameters, far too many for a few hundred texts)."""

    def __init__(self, series_names: list[str], in_dim: int, hidden: int, layers: int, heads: int, out_dim: int,
                 dropout: float):
        super().__init__()
        from textclassify.timeseries import CrossTSTransformer
        self.series_names = list(series_names)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.Dropout(dropout))
        self.transformer = CrossTSTransformer(series_names=self.series_names, hidden_size=hidden, output_dim=out_dim,
                                              num_heads=heads, num_layers=layers, dropout=dropout,
                                              feedforward_dim=4 * hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)                                                   # (B, series, hidden)
        return self.transformer({name: h[:, i] for i, name in enumerate(self.series_names)})["output"]


class FusionMLP(nn.Module):
    """embedding (raw, or Linear(proj_dim) + ReLU), concatenated with the LLM one-hot (and, with ts_dim, the time series
    embedding projected the same way) -> [hidden layer, only if hidden > 0] -> 3 classes. With ts_head the time series input is the
    (B, series, 1280) tensor of per-series embeddings, passed through the trainable CrossTSHead first."""

    def __init__(self, emb_dim: int, proj_dim: int, hidden: int, dropout: float, ts_dim: int = 0,
                 ts_head: CrossTSHead | None = None, ts_proj_dim: int = 32):
        super().__init__()
        self.ts_head = ts_head
        if ts_head is not None:
            ts_dim = ts_head.transformer.output_dim
        # proj_dim 0: the roberta embedding goes in RAW (all emb_dim values), no projection
        self.proj = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU()) if proj_dim else nn.Identity()
        self.ts_proj = nn.Sequential(nn.Linear(ts_dim, ts_proj_dim), nn.ReLU()) if ts_dim else None
        width = (proj_dim or emb_dim) + (ts_proj_dim if ts_dim else 0) + len(LABELS)
        if hidden:                                          # projection -> hidden layer -> classes
            self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, len(LABELS)))
        else:                                               # shallow: projection -> classes (no hidden layer)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, len(LABELS)))

    def forward(self, emb, llm_onehot, ts=None):
        if self.ts_head is not None:
            ts = self.ts_head(ts)
        parts = [self.proj(emb), llm_onehot] + ([self.ts_proj(ts)] if self.ts_proj is not None else [])
        return self.head(torch.cat(parts, dim=1))


def make_ts_head(args, in_dim: int) -> CrossTSHead:
    return CrossTSHead(args.ts_series, in_dim, args.cross_hidden, args.cross_layers, args.cross_heads,
                       out_dim=args.ts_proj_dim, dropout=args.dropout)


def onehot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(LABELS), dtype=np.float32)[labels]


def train_predict_mlp(emb_tr, llm_tr, y_tr, emb_te, llm_te, args, seed: int, ts_tr=None, ts_te=None) -> np.ndarray:
    """Train on 90 % of a regime with early stopping on the other 10 %, then predict the test set.
    ts_tr / ts_te: the time series embeddings (None = a model without the time series branch)."""
    torch.manual_seed(seed)
    idx_fit, idx_val = train_test_split(np.arange(len(y_tr)), test_size=0.1, random_state=seed, stratify=y_tr)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731

    def standardiser(x):                                                  # fitted on the fitting part only
        mean, std = x[idx_fit].mean(0), x[idx_fit].std(0) + 1e-6
        return lambda a: t((a - mean) / std)

    E = standardiser(emb_tr)
    e_fit, l_fit, y_fit = E(emb_tr[idx_fit]), t(llm_tr[idx_fit]), torch.tensor(y_tr[idx_fit])
    e_val, l_val, y_val = E(emb_tr[idx_val]), t(llm_tr[idx_val]), y_tr[idx_val]
    if ts_tr is not None:
        T = standardiser(ts_tr)
        t_fit, t_val, t_te = T(ts_tr[idx_fit]), T(ts_tr[idx_val]), T(ts_te)
    else:
        t_fit = t_val = t_te = None

    ts_head = None
    if ts_tr is not None and ts_tr.ndim == 3:                             # (texts, series, 1280): the trainable transformer
        ts_head = make_ts_head(args, ts_tr.shape[2])
    model = FusionMLP(emb_tr.shape[1], args.proj_dim, args.hidden, args.dropout,
                      ts_dim=ts_tr.shape[1] if ts_tr is not None and ts_head is None else 0, ts_head=ts_head,
                      ts_proj_dim=args.ts_proj_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(args.max_epochs):
        model.train()
        for b in torch.randperm(len(y_fit), generator=g).split(args.batch_size):
            opt.zero_grad()
            loss_fn(model(e_fit[b], l_fit[b], t_fit[b] if t_fit is not None else None), y_fit[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(y_val, model(e_val, l_val, t_val).argmax(1).numpy(), average="macro")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(E(emb_te), t(llm_te), t_te).argmax(1).numpy()


# ---- the fusion MLP with a LEARNABLE roberta-large --------------------------------------------------------------------

_PRETRAINED: dict = {}


def pretrained_encoder(name: str) -> nn.Module:
    """A fresh copy of the pretrained encoder (read from disk once): every training starts from the pretrained weights."""
    if name not in _PRETRAINED:
        from transformers import AutoModel
        _PRETRAINED[name] = AutoModel.from_pretrained(name, add_pooling_layer=False)
    return copy.deepcopy(_PRETRAINED[name])


def tokenize(df: pd.DataFrame, args) -> dict:
    """input_ids / attention_mask of all texts (padded to --max-length; batches are trimmed to their longest text)."""
    from transformers import AutoTokenizer
    enc = AutoTokenizer.from_pretrained(args.roberta_model)(
        list(df[TEXT]), truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class JointFusion(nn.Module):
    """roberta (trained) -> [CLS] -> the same FusionMLP head as above: LLM one-hot (and time series embedding) are added there."""

    def __init__(self, encoder: nn.Module, args, ts_dim: int = 0, ts_head: CrossTSHead | None = None):
        super().__init__()
        self.encoder = encoder
        self.head = FusionMLP(encoder.config.hidden_size, args.proj_dim, args.hidden, args.dropout, ts_dim=ts_dim,
                              ts_head=ts_head, ts_proj_dim=args.ts_proj_dim)

    def forward(self, input_ids, attention_mask, llm_onehot, ts=None):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head(cls.float(), llm_onehot, ts)


def train_predict_finetuned(enc: dict, tr: np.ndarray, te: np.ndarray, llm: np.ndarray, y: np.ndarray, args, seed: int,
                            ts: np.ndarray | None = None) -> np.ndarray:
    """roberta-large and the MLP head are trained TOGETHER on 90 % of the regime (tr = indices into the whole table),
    early stopping on the other 10 % (macro-F1), then the test texts (te) are predicted.

    The encoder starts from the pretrained weights and learns with --roberta-lr, the head with --lr; AdamW, linear warm-up
    and decay, gradient clipping, bf16 on a GPU. The time series embedding (ts) stays frozen, only its projection learns.
    llm is the LLM one-hot (all zeros = no LLM input)."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_fit, pos_val = train_test_split(np.arange(len(tr)), test_size=0.1, random_state=seed, stratify=y[tr])
    fit, val = tr[pos_fit], tr[pos_val]
    llm_t, y_t = torch.tensor(llm, dtype=torch.float32), torch.tensor(y)
    if ts is not None:                                                    # standardised on the fitting part only
        ts_t = torch.tensor((ts - ts[fit].mean(0)) / (ts[fit].std(0) + 1e-6), dtype=torch.float32)

    def batch(idx):
        idx = torch.as_tensor(idx)
        mask = enc["attention_mask"][idx]
        n = int(mask.sum(1).max())                                        # trim the padding of the batch
        return (enc["input_ids"][idx][:, :n].to(device), mask[:, :n].to(device), llm_t[idx].to(device),
                ts_t[idx].to(device) if ts is not None else None, y_t[idx].to(device))

    ts_head = make_ts_head(args, ts.shape[2]) if ts is not None and ts.ndim == 3 else None   # (texts, series, 1280)
    ts_dim = ts.shape[1] if ts is not None and ts_head is None else 0
    model = JointFusion(pretrained_encoder(args.roberta_model), args, ts_dim=ts_dim, ts_head=ts_head).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.roberta_lr},
                             {"params": model.head.parameters(), "lr": args.lr}], weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(fit) // args.ft_batch_size)
    total, warmup = steps_per_epoch * args.ft_max_epochs, max(1, int(args.ft_warmup * steps_per_epoch * args.ft_max_epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    loss_fn = nn.CrossEntropyLoss()
    autocast = lambda: torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")  # noqa: E731

    def predict(idx: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                ids, mask, l, t_, _ = batch(idx[s:s + 64])
                with autocast():
                    out.append(model(ids, mask, l, t_).argmax(1).cpu())
        return torch.cat(out).numpy()

    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(args.ft_max_epochs):
        model.train()
        for b in torch.randperm(len(fit), generator=g).split(args.ft_batch_size):
            ids, mask, l, t_, yb = batch(fit[b.numpy()])
            opt.zero_grad()
            with autocast():
                loss = loss_fn(model(ids, mask, l, t_).float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        val_f1 = f1_score(y[val], predict(val), average="macro")
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.ft_patience:
                break
    model.load_state_dict(best_state)
    pred = predict(te)
    del model, opt, sched, best_state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return pred


def stratified_regime_indices(train_idx: np.ndarray, y: np.ndarray, percent: int,
                              seed: int) -> np.ndarray:
    """Return a reproducible stratified subset of the fixed training set."""
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y[train_idx]):
        label_idx = train_idx[y[train_idx] == label].copy()
        rng.shuffle(label_idx)
        count = max(1, round(len(label_idx) * percent / 100))
        selected.append(label_idx[:count])
    return np.concatenate(selected)


# ---- evaluation -------------------------------------------------------------------------------------------------------

def scores(y, p) -> dict:
    return {"accuracy": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, p, average="weighted", zero_division=0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="articles JSON (default: the newest in data/)")
    parser.add_argument("--name", help="dataset name for the LLM cache and output files (default: the JSON's file name)")
    parser.add_argument("--make-llm-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="create missing zero-shot LLM labels via OpenRouter (paid; only texts not cached yet; "
                             "--no-make-llm-labels = cache only)")
    parser.add_argument("--llm-model", default="gpt-5-nano", help="LLM name in the cache folder")
    parser.add_argument("--llm-model-id", help="OpenRouter id for --make-llm-labels (default: openai/<--llm-model>)")
    parser.add_argument("--roberta-model", default="roberta-large", help="the frozen encoder (roberta-base for a quick test)")
    parser.add_argument("--max-length", type=int, default=128, help="tokens per text for the encoder")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction held out once as the full test set (default: 0.2)")
    parser.add_argument("--train-percents", type=int, nargs="+", default=list(range(10, 101, 10)),
                        help="training-set percentages to evaluate (default: 10 20 ... 100)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the fixed stratified split and regimes")
    parser.add_argument("--limit", type=int, help="only the first N texts (smoke test)")
    parser.add_argument("--drop-meta-text", action="store_true", help="drop texts in which the generator left notes")
    parser.add_argument("--ts", action=argparse.BooleanOptionalAction, default=True,
                        help="add the time series model fusion_ts_mlp (--no-ts: without, nothing is loaded from TimesFM)")
    parser.add_argument("--ts-cross", action=argparse.BooleanOptionalAction, default=True,
                        help="also train fusion_ts_cross_mlp: a trainable CrossTSTransformer on the frozen TimesFM "
                             "embeddings instead of their mean (needs --ts)")
    parser.add_argument("--cross-hidden", type=int, default=64, help="model size of the CrossTSTransformer (the 1280-dim embeddings are projected to it)")
    parser.add_argument("--cross-layers", type=int, default=1, help="encoder layers of the CrossTSTransformer")
    parser.add_argument("--cross-heads", type=int, default=4, help="attention heads of the CrossTSTransformer (divides --cross-hidden)")
    parser.add_argument("--ts-dir", type=Path, default=TS_DIR, help="folder with <series>.csv (columns Date,value)")
    parser.add_argument("--ts-series", nargs="+", default=TS_SERIES, help="series names = file names in --ts-dir")
    parser.add_argument("--ts-window", type=int, default=60,
                        help="last N observations up to the text's date, per series (daily series: days; 0 = whole history)")
    parser.add_argument("--ts-early", choices=("first", "error"), default="first",
                        help="texts dated before a series starts (DGS2 1976-06, NASDAQCOM 1971-02): 'first' = the series' first "
                             "--ts-window observations (look-ahead), 'error' = stop")
    parser.add_argument("--ts-model", default=TS_MODEL, help="frozen TimesFM encoder for the series")
    parser.add_argument("--finetune-roberta", action=argparse.BooleanOptionalAction, default=True,
                        help="roberta-large learns together with the MLP head in roberta_mlp / fusion_mlp / fusion_ts_mlp "
                             "(each training starts from the pretrained weights). --no-finetune-roberta: frozen, cached embeddings")
    parser.add_argument("--roberta-lr", type=float, default=1e-5, help="learning rate of roberta-large (--lr is the head's)")
    parser.add_argument("--ft-batch-size", type=int, default=16, help="batch size when roberta learns")
    parser.add_argument("--ft-max-epochs", type=int, default=10, help="maximum epochs when roberta learns")
    parser.add_argument("--ft-patience", type=int, default=3,
                        help="epochs without a better validation macro-F1 before stopping, when roberta learns")
    parser.add_argument("--ft-warmup", type=float, default=0.1, help="fraction of the steps with linear learning-rate warm-up")
    parser.add_argument("--proj-dim", type=int, default=0,
                        help="size the roberta embedding is projected to by Linear + ReLU BEFORE it meets the LLM one-hot; "
                             "0 (default) = raw: the 1024 values and the one-hot enter the first layer together")
    parser.add_argument("--ts-proj-dim", type=int, default=32, help="size the time series embedding is projected to")
    parser.add_argument("--hidden", type=int, default=8,
                        help="size of the first layer (Linear + ReLU) over the concatenated inputs: 1024 roberta + 3 LLM one-hot "
                             "= 1027 values; the second layer maps it to the 3 classes. 0 = a single linear layer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15, help="epochs without a better validation macro-F1 before stopping")
    args = parser.parse_args()
    if args.cross_hidden % args.cross_heads:
        parser.error("--cross-hidden must be divisible by --cross-heads")

    json_path = args.json if args.json else json_dataset.latest_json(DATA_DIR)
    df = json_dataset.read_articles_json(json_path, drop_meta_text=args.drop_meta_text)
    args.name = args.name or json_path.stem
    args.llm_model_id = args.llm_model_id or f"openai/{args.llm_model}"
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    y = df["label"].to_numpy()
    print(f"{json_path.name}: {len(df)} texts, {int(df['filler'].sum())} fillers, {int(df['label_flipped'].sum())} flipped labels")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    llm = llm_labels(df, args)
    if args.finetune_roberta:
        enc, emb = tokenize(df, args), None                  # roberta learns: no cached embeddings
    else:
        enc, emb = None, embeddings(df, args)
    out_name = args.name + ("_finetuned" if args.finetune_roberta else "")   # does not overwrite the frozen runs
    llm_1h, zeros = onehot(llm), np.zeros((len(df), len(LABELS)), dtype=np.float32)
    ts_series = ts_embeddings(df, args) if args.ts else None                # (texts, series, 1280), frozen TimesFM
    ts = ts_series.mean(axis=1) if args.ts else None                         # fusion_ts_mlp: averaged over the series
    use_cross = args.ts and args.ts_cross
    if use_cross:
        head = make_ts_head(args, ts_series.shape[2])
        n_cross = sum(p.numel() for p in head.parameters())
        print(f"{TS_CROSS_MODEL_NAME}: CrossTSTransformer head with {n_cross:,} trainable parameters "
              f"(series {', '.join(args.ts_series)}, {args.cross_layers} layer(s), {args.cross_hidden} dim); TimesFM frozen")
    models = MODELS + ((TS_MODEL_NAME,) if args.ts else ()) + ((TS_CROSS_MODEL_NAME,) if use_cross else ())
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=args.test_size, random_state=args.seed, stratify=y)
    groups = {
        "all": np.ones(len(test_idx), bool),
        "clean": ~(df["filler"].iloc[test_idx] | df["label_flipped"].iloc[test_idx]).to_numpy(),
        "filler": df["filler"].iloc[test_idx].to_numpy(),
        "flipped": df["label_flipped"].iloc[test_idx].to_numpy(),
    }

    prediction_rows, score_rows = [], []
    for percent in args.train_percents:
        if not 1 <= percent <= 100:
            raise SystemExit(f"--train-percents must be between 1 and 100: {percent}")
        tr = stratified_regime_indices(train_idx, y, percent, args.seed)
        t0 = time.time()
        preds = {"llm": llm[test_idx]}
        logreg = Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))]).fit(
            df[TEXT].iloc[tr], y[tr])
        preds["logreg"] = logreg.predict(df[TEXT].iloc[test_idx])
        xgboost = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("xgb", XGBClassifier(objective="multi:softprob", num_class=len(LABELS),
                                  eval_metric="mlogloss", random_state=args.seed,
                                  tree_method="hist", **XGBOOST_PARAMS)),
        ]).fit(df[TEXT].iloc[tr], y[tr])
        preds["xgboost"] = xgboost.predict(df[TEXT].iloc[test_idx])
        def fit_predict(llm_input, ts_input=None):
            seed = args.seed + percent
            if args.finetune_roberta:
                return train_predict_finetuned(enc, tr, test_idx, llm_input, y, args, seed, ts=ts_input)
            return train_predict_mlp(emb[tr], llm_input[tr], y[tr], emb[test_idx], llm_input[test_idx], args, seed=seed,
                                     ts_tr=ts_input[tr] if ts_input is not None else None,
                                     ts_te=ts_input[test_idx] if ts_input is not None else None)

        preds["roberta_mlp"] = fit_predict(zeros)
        preds["fusion_mlp"] = fit_predict(llm_1h)
        if args.ts:
            preds[TS_MODEL_NAME] = fit_predict(llm_1h, ts)
        if use_cross:
            preds[TS_CROSS_MODEL_NAME] = fit_predict(llm_1h, ts_series)
        for model in models:
            for group, mask in groups.items():
                if mask.any():
                    score_rows.append({
                        "train_percent": percent, "train_texts": len(tr),
                        "model": model, "texts": group,
                        **scores(y[test_idx][mask], preds[model][mask]),
                    })
            for row_idx, original_idx in enumerate(test_idx):
                prediction_rows.append({
                    "train_percent": percent, "model": model,
                    TEXT: df[TEXT].iloc[original_idx], "obs_id": df["obs_id"].iloc[original_idx],
                    "label": y[original_idx], "true": json_dataset.LABEL_MAP[y[original_idx]],
                    "prediction": json_dataset.LABEL_MAP[preds[model][row_idx]],
                    "filler": df["filler"].iloc[original_idx],
                    "label_flipped": df["label_flipped"].iloc[original_idx],
                })
        print(f"train {percent:3d}% ({len(tr)} texts, {time.time() - t0:.0f}s): " +
              "  ".join(f"{m} {scores(y[test_idx], preds[m])['f1_macro']:.3f}" for m in models))

    predictions = pd.DataFrame(prediction_rows)
    scores_df = pd.DataFrame(score_rows)
    predictions.to_csv(OUT_DIR / f"{out_name}_regime_predictions.csv", index=False)
    scores_df.to_csv(OUT_DIR / f"{out_name}_regime_scores.csv", index=False)
    print(f"\n=== Fixed test set ({len(test_idx)} texts), training regimes ===")
    for percent in args.train_percents:
        rows = scores_df[(scores_df["train_percent"] == percent) & (scores_df["texts"] == "all")].set_index("model")
        print(f"{percent:3d}% " + "  ".join(
            f"{m} {rows.loc[m, 'f1_macro']:.3f} ({rows.loc[m, 'accuracy']:.3f})"
            for m in models if m in rows.index))
    print(f"\nSaved in {OUT_DIR}: {out_name}_regime_predictions.csv, {out_name}_regime_scores.csv")

if __name__ == "__main__":
    main()
