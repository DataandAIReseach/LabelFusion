"""LLM predictions + roberta-large embeddings fused by an MLP, on the full synthetic dataset (5-fold cross-validation).

For every text of an articles JSON (dataset_scripts/article_creation.ipynb, read by dataset_scripts/json_dataset.py):

  embedding   the [CLS] embedding of the pretrained, FROZEN roberta-large (computed once, cached on disk)
  LLM label   the zero-shot label of GPT-5 nano (one OpenRouter call per text, cached; missing ones are created unless --no-make-llm-labels; the
              missing ones). The same zero-shot setting for training and test texts, so the fusion is trained on the
              kind of LLM label it later sees.

and per fold of a stratified 5-fold cross-validation (the folds of the logistic-regression benchmark:
StratifiedKFold(5, shuffle=True, random_state=42) over the texts in obs_id order) the columns

  llm          the LLM label itself (no training)
  logreg       TF-IDF + logistic regression (standard settings), the text-only reference
  roberta_mlp  MLP on the roberta-large embedding alone (the LLM input is zeroed)
  fusion_mlp   MLP on the roberta-large embedding + the LLM label as one-hot: LabelFusion

The two MLPs have the same architecture (as textclassify's FusionWrapper): the 1024-dim embedding is projected to
--proj-dim (default 32) by Linear + ReLU, concatenated with the 3-dim LLM one-hot, one hidden layer (--hidden) and the
3-class output. Trained with AdamW and cross-entropy on 90 % of the training folds, early stopping on the other 10 %
(--patience); the test fold is never used for any choice.

Scores: accuracy, macro-F1, weighted-F1 per fold and mean +- std, also split by text type (clean content texts,
content-free fillers with a random label, content texts with a deliberately flipped label).

Writes to ./outputs/predict_all_full_model/ (repo root):
    <name>_predictions.csv    one row per text: fold, true label, the prediction of every model, filler, label_flipped
    <name>_folds.csv          one row per fold and model: the metrics
    <name>_summary.csv        per model: mean and std over the folds, overall and per text type
    embeddings/<model>_<hash>.npy   the cached embeddings (keyed by model and texts)

    python testing/predict_all_full_model.py                        # missing LLM labels via OpenRouter (key from .env), then the rest
    python testing/predict_all_full_model.py --no-make-llm-labels   # never call the API: stop if LLM labels are missing
    python testing/predict_all_full_model.py --json data/<articles>.json --limit 300 --roberta-model roberta-base   # smoke test
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
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import json_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")                           # OPENROUTER_API_KEY for the LLM labels

LABELS = list(json_dataset.LABEL_MAP.values())           # dovish, hawkish, neutral (index = label id)
TEXT = json_dataset.TEXT_COLUMN
OUT_DIR = REPO_ROOT / "outputs" / "predict_all_full_model"
LLM_CACHE_DIR = REPO_ROOT / "outputs" / "llm_cache"
DATA_DIR = REPO_ROOT / "data"                            # default: the newest articles JSON here
MODELS = ("llm", "logreg", "roberta_mlp", "fusion_mlp")


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


# ---- the fusion MLP ---------------------------------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """embedding -> Linear(proj_dim) + ReLU, concatenated with the LLM one-hot -> hidden layer -> 3 classes."""

    def __init__(self, emb_dim: int, proj_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(proj_dim + len(LABELS), hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, len(LABELS)))

    def forward(self, emb, llm_onehot):
        return self.head(torch.cat([self.proj(emb), llm_onehot], dim=1))


def onehot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(LABELS), dtype=np.float32)[labels]


def train_predict_mlp(emb_tr, llm_tr, y_tr, emb_te, llm_te, args, seed: int) -> np.ndarray:
    """Train on 90 % of the training folds with early stopping on the other 10 %, predict the test fold."""
    torch.manual_seed(seed)
    idx_fit, idx_val = train_test_split(np.arange(len(y_tr)), test_size=0.1, random_state=seed, stratify=y_tr)
    mean, std = emb_tr[idx_fit].mean(0), emb_tr[idx_fit].std(0) + 1e-6   # standardised on the fitting part only
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    E = lambda e: t((e - mean) / std)  # noqa: E731
    e_fit, l_fit, y_fit = E(emb_tr[idx_fit]), t(llm_tr[idx_fit]), torch.tensor(y_tr[idx_fit])
    e_val, l_val, y_val = E(emb_tr[idx_val]), t(llm_tr[idx_val]), y_tr[idx_val]

    model = FusionMLP(emb_tr.shape[1], args.proj_dim, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(args.max_epochs):
        model.train()
        for b in torch.randperm(len(y_fit), generator=g).split(args.batch_size):
            opt.zero_grad()
            loss_fn(model(e_fit[b], l_fit[b]), y_fit[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_f1 = f1_score(y_val, model(e_val, l_val).argmax(1).numpy(), average="macro")
        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(E(emb_te), t(llm_te)).argmax(1).numpy()


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="seed of the folds (42 = the folds of the logistic-regression benchmark)")
    parser.add_argument("--limit", type=int, help="only the first N texts (smoke test)")
    parser.add_argument("--drop-meta-text", action="store_true", help="drop texts in which the generator left notes")
    parser.add_argument("--proj-dim", type=int, default=32, help="size the 1024-dim embedding is projected to")
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

    preds = {m: np.full(len(df), -1) for m in MODELS}
    fold_of = np.full(len(df), -1)
    folds = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, te) in enumerate(folds.split(df[TEXT], y)):
        t0 = time.time()
        fold_of[te] = k
        preds["llm"][te] = llm[te]
        logreg = Pipeline([("tfidf", TfidfVectorizer()), ("lr", LogisticRegression(max_iter=1000))]).fit(df[TEXT].iloc[tr], y[tr])
        preds["logreg"][te] = logreg.predict(df[TEXT].iloc[te])
        preds["roberta_mlp"][te] = train_predict_mlp(emb[tr], zeros[tr], y[tr], emb[te], zeros[te], args, seed=k)
        preds["fusion_mlp"][te] = train_predict_mlp(emb[tr], llm_1h[tr], y[tr], emb[te], llm_1h[te], args, seed=k)
        print(f"fold {k + 1}/{args.folds} ({time.time() - t0:.0f}s): " + "  ".join(
            f"{m} {f1_score(y[te], preds[m][te], average='macro'):.3f}" for m in MODELS))

    # ---- outputs ----
    out = df[[TEXT, "obs_id", "label", "filler", "label_flipped"]].assign(fold=fold_of, true=df["label"].map(json_dataset.LABEL_MAP))
    for m in MODELS:
        out[f"{m}_pred"] = pd.Series(preds[m]).map(json_dataset.LABEL_MAP)
    out.to_csv(OUT_DIR / f"{args.name}_predictions.csv", index=False)

    groups = {"all": np.ones(len(df), bool), "clean": ~(df["filler"] | df["label_flipped"]).to_numpy(),
              "filler": df["filler"].to_numpy(), "flipped": df["label_flipped"].to_numpy()}
    fold_rows = [{"fold": k, "model": m, "texts": g, **scores(y[(fold_of == k) & mask], preds[m][(fold_of == k) & mask])}
                 for k in range(args.folds) for m in MODELS for g, mask in groups.items() if ((fold_of == k) & mask).any()]
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT_DIR / f"{args.name}_folds.csv", index=False)
    summary = folds_df.groupby(["texts", "model"])[["accuracy", "f1_macro", "f1_weighted"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / f"{args.name}_summary.csv", index=False)

    print(f"\n=== {args.folds}-fold CV, macro-F1 mean +- std (accuracy) ===")
    for g in groups:
        rows = summary[summary["texts"] == g].set_index("model")
        if len(rows):
            print(f"{g:8s} " + "  ".join(f"{m} {rows.loc[m, 'f1_macro_mean']:.3f}+-{rows.loc[m, 'f1_macro_std']:.3f} "
                                          f"({rows.loc[m, 'accuracy_mean']:.3f})" for m in MODELS if m in rows.index))
    print(f"\nSaved in {OUT_DIR}: {args.name}_predictions.csv, {args.name}_folds.csv, {args.name}_summary.csv")


if __name__ == "__main__":
    main()
