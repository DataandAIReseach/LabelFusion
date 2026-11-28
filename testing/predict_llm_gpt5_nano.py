"""GPT-5 nano alone: zero-shot labels for every text of an articles JSON, and their scores.

For every text of an articles JSON (dataset_scripts/article_creation.ipynb, read by dataset_scripts/json_dataset.py)
GPT-5 nano (OpenRouter, key OPENROUTER_API_KEY from the repo's .env) predicts dovish / hawkish / neutral, zero-shot: the
prompt is built from the label names only, no labelled text reaches it. No model is trained, so this runs on a CPU.

Cache: the labels are kept in the same folder testing/predict_all_full_model.py uses,
    outputs/llm_cache/predict_all_train_<llm>_zero_shot_<json name>/
so the two scripts share them (whichever runs first pays, the other makes no calls). Only texts without a label there go
to the API. --reuse-caches (default) also takes labels for identical texts from the zero-shot caches of the OTHER JSONs
(e.g. the 2500 OUT texts the struct / var / rho JSONs share with the baseline); those are not written into this
JSON's cache folder, so predict_all_full_model.py would still ask for them. --no-api never calls the API.

Scores: accuracy, macro-F1 and weighted-F1 on all texts and on the held-out test part of predict_all_full_model.py
(the same stratified split: --test-size 0.2, --seed 42), each for all texts and per text type
(clean / filler / flipped) and period (IN / OUT).

Writes to ./outputs/predict_llm_gpt5_nano/ (repo root):
    <json name>_<llm>_predictions.csv  one row per text: obs_id, date, IN, topic, filler, label_flipped, in_test,
                                       sentence, true, pred (empty where the API gave no usable answer)
    <json name>_<llm>_scores.csv       one row per split (all / test) and group
    summary_<llm>.csv                  the scores of every JSON of this run in one table

    python testing/predict_llm_gpt5_nano.py                                         # baseline + every JSON in data/articles/jsons
    python testing/predict_llm_gpt5_nano.py --json data/articles/jsons/articles_inflation_unemployment_var_high_seed7_n5000.json
    python testing/predict_llm_gpt5_nano.py --no-api                                # only what is cached, no calls
    python testing/predict_llm_gpt5_nano.py --json <file>.json --limit 20           # smoke test (20 texts)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTING_DIR.parent
sys.path.insert(0, str(TESTING_DIR / "old"))              # predict_all_files_llm.py (OpenRouter classifier, cache layout)
sys.path.insert(0, str(REPO_ROOT / "dataset_scripts"))   # json_dataset.py

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import json_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")                           # OPENROUTER_API_KEY

TEXT = json_dataset.TEXT_COLUMN
LABEL_MAP = json_dataset.LABEL_MAP                        # 0 dovish, 1 hawkish, 2 neutral
JSONS_DIR = REPO_ROOT / "data" / "articles" / "jsons"
LLM_CACHE_DIR = REPO_ROOT / "outputs" / "llm_cache"
OUT_DIR = REPO_ROOT / "outputs" / "predict_llm_gpt5_nano"


# ---- the LLM cache (same layout as predict_all_full_model.py) ---------------------------------------------------------

def cache_folder(llm_model: str, name: str) -> Path:
    return LLM_CACHE_DIR / f"predict_all_train_{llm_model}_zero_shot_{name}"


def read_cache(folder: Path) -> dict[str, int]:
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


def drop_failed(folder: Path) -> int:
    """Remove the failed calls from the classifier's cache, so that they are asked again.

    textclassify stores a failed call (e.g. a connection error) with success=False but prediction [1, 0, 0] and, on the
    next run, takes that entry as cached instead of calling again. Its per-dataset result files (test_<hash>.json) of a
    run that produced no predictions would also be returned as they are, so those are removed too."""
    dropped = 0
    for pkl in sorted(folder.glob("cache_*.pkl")):
        with open(pkl, "rb") as fh:
            entries = pickle.load(fh)
        keep = {k: v for k, v in entries.items() if v.get("success")}
        if len(keep) < len(entries):
            dropped += len(entries) - len(keep)
            tmp = pkl.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(keep, fh)
            tmp.replace(pkl)
    for res in folder.glob("test_*.json"):
        try:
            if not json.load(open(res)).get("predictions"):
                res.unlink()
        except (OSError, ValueError):
            pass
    return dropped


def other_caches(llm_model: str, own: Path) -> dict[str, int]:
    """Labels from the zero-shot caches of the other articles JSONs of the same LLM (texts shared between datasets)."""
    preds: dict[str, int] = {}
    for folder in sorted(LLM_CACHE_DIR.glob(f"predict_all_train_{llm_model}_zero_shot_articles_*")):
        if folder != own and folder.is_dir():
            preds.update(read_cache(folder))
    return preds


def llm_labels(df: pd.DataFrame, name: str, args) -> tuple[pd.Series, dict]:
    """Predicted label id per text (NaN where none could be obtained) and where the labels came from."""
    folder = cache_folder(args.llm_model, name)
    own = read_cache(folder) if folder.is_dir() else {}
    reused = {t: p for t, p in other_caches(args.llm_model, folder).items() if t not in own} if args.reuse_caches else {}
    missing = df[~df[TEXT].isin(own) & ~df[TEXT].isin(reused)]
    n_cached = int(df[TEXT].isin(own).sum())
    n_reused = int((~df[TEXT].isin(own) & df[TEXT].isin(reused)).sum())
    print(f"  LLM labels: {n_cached} cached, {n_reused} from other JSONs' caches, {len(missing)} missing")
    called = 0
    if len(missing) and args.api:
        import predict_all_files_llm as llm_files          # OpenRouter; writes the cache into `folder`
        if folder.is_dir() and (n := drop_failed(folder)):
            print(f"  removed {n} failed calls from the cache (they are asked again)")
        print(f"  -> {len(missing)} calls to {args.llm_model_id} (zero-shot)")
        llm_files.build(args.llm_model_id, folder.name.removeprefix("predict_all_"), 0).predict(
            train_df=None, test_df=missing[json_dataset.MODEL_COLUMNS].reset_index(drop=True))
        called = len(missing)
        own = read_cache(folder)
    labels = df[TEXT].map({**reused, **own})              # this JSON's own cache wins
    source = {"cached": n_cached, "reused": n_reused, "called": called, "missing": int(labels.isna().sum())}
    return labels, source


# ---- scores -----------------------------------------------------------------------------------------------------------

def score_rows(df: pd.DataFrame, split: str) -> list[dict]:
    groups = {
        "all":     np.ones(len(df), bool),
        "clean":   ~(df["filler"] | df["label_flipped"]).to_numpy(),
        "filler":  df["filler"].to_numpy(),
        "flipped": df["label_flipped"].to_numpy(),
        "IN":      (df["IN"] == 1).to_numpy(),
        "OUT":     (df["IN"] == 0).to_numpy(),
    }
    rows = []
    for group, mask in groups.items():
        part = df[mask & df["pred_id"].notna().to_numpy()]
        row = {"split": split, "group": group, "n": int(mask.sum()), "n_scored": len(part)}
        if len(part):
            y, p = part["label"].astype(int), part["pred_id"].astype(int)
            row.update(accuracy=accuracy_score(y, p),
                       f1_macro=f1_score(y, p, average="macro", labels=list(LABEL_MAP), zero_division=0),
                       f1_weighted=f1_score(y, p, average="weighted", labels=list(LABEL_MAP), zero_division=0))
        rows.append(row)
    return rows


def run(json_path: Path, args) -> pd.DataFrame:
    name = json_path.stem
    df = json_dataset.read_articles_json(json_path, drop_meta_text=args.drop_meta_text)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"{json_path.name}: {len(df)} texts ({int(df['IN'].sum())} IN, {int(df['filler'].sum())} fillers, "
          f"{int(df['label_flipped'].sum())} flipped labels)")

    df["pred_id"], source = llm_labels(df, name, args)
    # the held-out test part of predict_all_full_model.py (same call, same seed)
    _, test_idx = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=args.seed, stratify=df["label"])
    df["in_test"] = False
    df.loc[test_idx, "in_test"] = True

    out = df[["obs_id", "date", "IN", "topic", "filler", "label_flipped", "in_test", TEXT]].copy()
    out["true"] = df["label"].map(LABEL_MAP)
    out["pred"] = df["pred_id"].map(lambda v: LABEL_MAP[int(v)] if pd.notna(v) else "")
    scores = pd.DataFrame(score_rows(df, "all") + score_rows(df[df["in_test"]].reset_index(drop=True), "test"))
    scores.insert(0, "dataset", name)
    for key, value in source.items():
        scores[f"labels_{key}"] = value

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{args.llm_model}" + (f"_limit{args.limit}" if args.limit else "")
    out.to_csv(OUT_DIR / f"{stem}_predictions.csv", index=False)
    scores.to_csv(OUT_DIR / f"{stem}_scores.csv", index=False)

    if source["missing"]:
        print(f"  WARNING: {source['missing']} texts without a label (not scored)"
              + ("" if args.api else " -- run without --no-api to create them"))
    for split in ("all", "test"):
        r = scores[(scores["split"] == split) & (scores["group"] == "all")].iloc[0]
        if r["n_scored"]:
            print(f"  {split:4s}: n={int(r['n_scored'])}  accuracy {r['accuracy']:.3f}  macro-F1 {r['f1_macro']:.3f}  "
                  f"weighted-F1 {r['f1_weighted']:.3f}")
    print(f"  -> {OUT_DIR / stem}_{{predictions,scores}}.csv\n")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, nargs="+",
                        help="articles JSON(s) (default: the baseline in data/, then every *.json in data/articles/jsons)")
    parser.add_argument("--llm-model", default="gpt-5-nano", help="LLM name in the cache folder and output file names")
    parser.add_argument("--llm-model-id", help="OpenRouter id (default: openai/<--llm-model>)")
    parser.add_argument("--api", action=argparse.BooleanOptionalAction, default=True,
                        help="create missing labels via OpenRouter (default); --no-api only uses what is cached")
    parser.add_argument("--reuse-caches", action=argparse.BooleanOptionalAction, default=True,
                        help="take labels of identical texts from the other articles JSONs' zero-shot caches (default)")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of the held-out test part (as predict_all_full_model.py)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the stratified split (as predict_all_full_model.py)")
    parser.add_argument("--drop-meta-text", action="store_true", help="drop texts in which the generator left notes")
    parser.add_argument("--limit", type=int, help="only the first N texts of every JSON (smoke test)")
    args = parser.parse_args()
    args.llm_model_id = args.llm_model_id or f"openai/{args.llm_model}"

    # the baseline first: the struct / var / rho JSONs share its OUT texts and then take its labels instead of asking
    paths = args.json or (sorted(REPO_ROOT.glob("data/articles_all_commodities_baseline_*.json"))
                          + sorted(p for p in JSONS_DIR.glob("*.json") if not p.name.endswith("_failures.json")))
    if not paths:
        raise SystemExit(f"no articles JSON in {JSONS_DIR}")
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(missing))

    summary = pd.concat([run(Path(p).resolve(), args) for p in paths], ignore_index=True)
    path = OUT_DIR / (f"summary_{args.llm_model}" + (f"_limit{args.limit}" if args.limit else ""))
    summary.to_csv(f"{path}.csv", index=False)
    table = summary[summary["group"] == "all"].pivot(index="dataset", columns="split", values="f1_macro")
    print("macro-F1 of the LLM labels (all texts / test part):")
    print(table.round(3).to_string())
    print(f"\nSaved: {path}.csv")


if __name__ == "__main__":
    main()
