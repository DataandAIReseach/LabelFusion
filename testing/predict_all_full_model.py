"""LLM predictions + roberta-large embeddings fused by an MLP, on fixed-test training regimes.

For every text of an articles JSON (dataset_scripts/article_creation.ipynb, read by dataset_scripts/json_dataset.py):

  embedding   the [CLS] embedding of the pretrained, FROZEN roberta-large (computed once, cached on disk)
  LLM label   the zero-shot label of GPT-5 nano (one OpenRouter call per text, cached; missing ones are created unless
              --no-make-llm-labels). The same zero-shot setting for training and test texts, so the fusion is trained on the
              kind of LLM label it later sees.
  time series the real macro series of data/TSs_macro (default CPIAUCSL + UNRATE): the last --ts-window observations up to the
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

The MLPs have the same architecture (as textclassify's FusionWrapper): the 1024-dim embedding is projected to
--proj-dim (default 32) by Linear + ReLU, concatenated with the 3-dim LLM one-hot (and, for fusion_ts_mlp, the time series
embedding projected to --proj-dim in the same way), one hidden layer (--hidden) and the 3-class output. Trained with AdamW and cross-entropy on 90 % of each regime, early stopping on the other 10 %
(--patience); the fixed test set is never used for any choice.

Scores: accuracy, macro-F1, and weighted-F1 per training regime, also split by text type (clean content texts,
content-free fillers with a random label, content texts with a deliberately flipped label).

Writes to ./outputs/predict_all_full_model/ (repo root):
    <name>_regime_predictions.csv  one row per test text, model, and training percentage
    <name>_regime_scores.csv       one row per training percentage, model, and text type
    embeddings/<model>_<hash>.npy   the cached embeddings (keyed by model and texts)
    embeddings/ts_<series>_w<window>_<hash>.npy   the cached time series embeddings (keyed by series, window and dates)

    python testing/predict_all_full_model.py                        # missing LLM labels via OpenRouter (key from .env), then the rest
    python testing/predict_all_full_model.py --no-make-llm-labels   # never call the API: stop if LLM labels are missing
    python testing/predict_all_full_model.py --json data/<articles>.json --roberta-model roberta-base --limit 300   # smoke test
"""

from __future__ import annotations

import argparse
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
TS_SERIES = ["CPIAUCSL", "UNRATE"]                       # inflation and unemployment: the two topics of the articles
TS_MODEL = "google/timesfm-2.5-200m-transformers"
MODELS = ("llm", "logreg", "xgboost", "roberta_mlp", "fusion_mlp")
TS_MODEL_NAME = "fusion_ts_mlp"
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

def series_windows(path: Path, dates: np.ndarray, window: int) -> list[np.ndarray]:
    """For every date the last `window` observations of the series up to and including that date (window 0 = all of them).

    The series have different frequencies (CPIAUCSL / UNRATE monthly, stamped on the 1st of the month): the window counts
    observations, not days, so 21 = the last 21 months here. A monthly value counts from the 1st of its month on."""
    raw = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").ffill()       # gaps (holidays): the last known value
    raw = raw.dropna(subset=["value"])
    stamps, values = raw["Date"].to_numpy().astype("datetime64[ns]"), raw["value"].to_numpy(np.float32)
    ends = np.searchsorted(stamps, dates.astype("datetime64[ns]"), side="right")   # observations <= date
    if (ends == 0).any():
        raise SystemExit(f"{path.name} starts at {pd.Timestamp(stamps[0]).date()}, after the first text "
                         f"({pd.Timestamp(dates.min()).date()})")
    return [values[max(0, e - window) if window else 0:e] for e in ends]


def ts_embeddings(df: pd.DataFrame, args) -> np.ndarray:
    """One frozen TimesFM embedding per series and text (identical windows are embedded once), mean-pooled over the series."""
    dates = pd.to_datetime(df[json_dataset.DATE_COLUMN]).to_numpy()
    key = hashlib.md5(("\x1f".join(df[json_dataset.DATE_COLUMN]) + f"|{args.ts_model}").encode()).hexdigest()[:12]
    path = OUT_DIR / "embeddings" / f"ts_{'-'.join(args.ts_series)}_w{args.ts_window}_{key}.npy"
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
        windows = series_windows(args.ts_dir / f"{name}.csv", dates, args.ts_window)
        unique: dict[bytes, int] = {}
        index = np.array([unique.setdefault(w.tobytes(), len(unique)) for w in windows])
        first = {i: k for k, i in reversed(list(enumerate(index)))}           # unique id -> first text with that window
        todo = [windows[first[i]] for i in range(len(unique))]
        out = []
        for start in range(0, len(todo), 64):
            with torch.no_grad():
                out.append(encoder(todo[start:start + 64])["embeddings"].float().cpu().numpy())
        per_series.append(np.concatenate(out)[index])
        print(f"  {name}: {len(unique)} distinct windows for {len(windows)} texts")
    emb = np.mean(per_series, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"time series embeddings: {emb.shape} in {time.time() - t0:.0f}s -> {path.name}")
    return emb


# ---- the fusion MLP ---------------------------------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """embedding -> Linear(proj_dim) + ReLU, concatenated with the LLM one-hot (and, with ts_dim, the time series
    embedding projected the same way) -> hidden layer -> 3 classes."""

    def __init__(self, emb_dim: int, proj_dim: int, hidden: int, dropout: float, ts_dim: int = 0):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU())
        self.ts_proj = nn.Sequential(nn.Linear(ts_dim, proj_dim), nn.ReLU()) if ts_dim else None
        self.head = nn.Sequential(nn.Linear(proj_dim * (2 if ts_dim else 1) + len(LABELS), hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, len(LABELS)))

    def forward(self, emb, llm_onehot, ts=None):
        parts = [self.proj(emb), llm_onehot] + ([self.ts_proj(ts)] if self.ts_proj is not None else [])
        return self.head(torch.cat(parts, dim=1))


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

    model = FusionMLP(emb_tr.shape[1], args.proj_dim, args.hidden, args.dropout,
                      ts_dim=ts_tr.shape[1] if ts_tr is not None else 0)
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
    parser.add_argument("--ts-dir", type=Path, default=TS_DIR, help="folder with <series>.csv (columns Date,value)")
    parser.add_argument("--ts-series", nargs="+", default=TS_SERIES, help="series names = file names in --ts-dir")
    parser.add_argument("--ts-window", type=int, default=21,
                        help="last N observations up to the text's date, per series (monthly series: months; 0 = whole history)")
    parser.add_argument("--ts-model", default=TS_MODEL, help="frozen TimesFM encoder for the series")
    parser.add_argument("--proj-dim", type=int, default=32, help="size the embeddings are projected to")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15, help="epochs without a better validation macro-F1 before stopping")
    args = parser.parse_args()

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
    emb = embeddings(df, args)
    llm_1h, zeros = onehot(llm), np.zeros((len(df), len(LABELS)), dtype=np.float32)
    ts = ts_embeddings(df, args) if args.ts else None
    models = MODELS + ((TS_MODEL_NAME,) if args.ts else ())
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
        preds["roberta_mlp"] = train_predict_mlp(
            emb[tr], zeros[tr], y[tr], emb[test_idx], zeros[test_idx], args, seed=args.seed + percent)
        preds["fusion_mlp"] = train_predict_mlp(
            emb[tr], llm_1h[tr], y[tr], emb[test_idx], llm_1h[test_idx], args, seed=args.seed + percent)
        if args.ts:
            preds[TS_MODEL_NAME] = train_predict_mlp(
                emb[tr], llm_1h[tr], y[tr], emb[test_idx], llm_1h[test_idx], args, seed=args.seed + percent,
                ts_tr=ts[tr], ts_te=ts[test_idx])
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
    predictions.to_csv(OUT_DIR / f"{args.name}_regime_predictions.csv", index=False)
    scores_df.to_csv(OUT_DIR / f"{args.name}_regime_scores.csv", index=False)
    print(f"\n=== Fixed test set ({len(test_idx)} texts), training regimes ===")
    for percent in args.train_percents:
        rows = scores_df[(scores_df["train_percent"] == percent) & (scores_df["texts"] == "all")].set_index("model")
        print(f"{percent:3d}% " + "  ".join(
            f"{m} {rows.loc[m, 'f1_macro']:.3f} ({rows.loc[m, 'accuracy']:.3f})"
            for m in models if m in rows.index))
    print(f"\nSaved in {OUT_DIR}: {args.name}_regime_predictions.csv, {args.name}_regime_scores.csv")

if __name__ == "__main__":
    main()
