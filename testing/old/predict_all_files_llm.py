"""Predict every file in data/training_data and data/test_data with GPT-5 nano
(best clean result in testing/run_llm_battery.py):

  * train: openai/gpt-5-nano, zero-shot, on all data/training_data files. No
    labelled rows reach the prompt (prompts are built from the label names only).
    Zero-shot prompts don't depend on the file, so every unique sentence is
    classified once and the prediction is copied to each file containing it.
  * test:  openai/gpt-5-nano, 5-shot, on all data/test_data files. The 5
    examples come from the matching train file (same name with -test- -> -train-).

Writes one CSV per input file (sentence, label, true, pred) plus a
summary.csv with accuracy / macro-F1 per file, all under ./outputs (repo root):

    outputs/<input file name>.json                 JSON, named exactly like the Excel file
    outputs/predictions/<run>/<input file name>.csv
    outputs/predictions/summary_{train,test}.csv

The classifier's own experiment logs and prediction cache (outputs/experiments,
outputs/llm_cache) live there too; a rerun reuses the cache and makes no API calls.

    python testing/predict_all_files_llm.py              # both parts
    python testing/predict_all_files_llm.py train        # only the training files
    python testing/predict_all_files_llm.py test --limit 20   # smoke test
    python testing/predict_all_files_llm.py test --from-cache # rebuild CSV/JSON from the
                                                              # prediction cache, no API calls
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

from test_textclassify_llm import (  # noqa: E402  (also loads .env)
    LABEL_COLUMNS, LABEL_MAP, RANDOM_STATE, REPO_ROOT, TEXT_COLUMN, _add_onehot_labels,
)
from textclassify import OpenRouterClassifier  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402

TRAIN_DIR = REPO_ROOT / "data" / "training_data"
TEST_DIR = REPO_ROOT / "data" / "test_data"
OUT_DIR = REPO_ROOT / "outputs"  # where the prediction files are saved
PRED_DIR = OUT_DIR / "predictions"

TRAIN_MODEL = "openai/gpt-5-nano"
TEST_MODEL = "openai/gpt-5-nano"
TEST_SHOTS = 5


def build(model: str, run_name: str, shots: int) -> OpenRouterClassifier:
    config = ModelConfig(
        model_name=model,
        model_type=ModelType.LLM,
        parameters={"model": model, "temperature": 0.1, "max_completion_tokens": 100},
    )
    return OpenRouterClassifier(
        config=config,
        text_column=TEXT_COLUMN,
        label_columns=LABEL_COLUMNS,
        multi_label=False,
        few_shot_mode="zero_shot" if shots == 0 else shots,
        output_dir=str(OUT_DIR),
        experiment_name=f"predict_all_{run_name}",
        cache_dir=str(OUT_DIR / "llm_cache" / f"predict_all_{run_name}"),
    )


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _add_onehot_labels(df[[TEXT_COLUMN, "label"]].copy()).reset_index(drop=True)


def save(df: pd.DataFrame, preds: list, out_dir: Path, name: str) -> dict:
    out = df[[TEXT_COLUMN, "label"]].copy()
    out["true"] = out["label"].map(LABEL_MAP)
    out["pred"] = preds
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{name}.csv", index=False)
    # JSON copy in ./outputs, named exactly like the source Excel file
    out.to_json(OUT_DIR / f"{name}.json", orient="records", force_ascii=False, indent=2)
    return {
        "file": name,
        "rows": len(out),
        "accuracy": accuracy_score(out["true"], out["pred"]),
        "f1_macro": f1_score(out["true"], out["pred"], average="macro"),
    }


def cached_predictions(run_name: str) -> dict[str, str]:
    """sentence -> predicted label name, read from the classifier's prediction cache
    (outputs/llm_cache/predict_all_<run_name>/*.pkl). Only successful calls count."""
    preds: dict[str, str] = {}
    for pkl in sorted((OUT_DIR / "llm_cache" / f"predict_all_{run_name}").glob("cache_*.pkl")):
        with open(pkl, "rb") as fh:
            for entry in pickle.load(fh).values():
                vec = entry.get("prediction")
                if entry.get("success") and vec is not None and sum(vec) == 1:
                    preds[entry["text"]] = LABEL_COLUMNS[list(vec).index(1)]
    return preds


def predict_train(limit: int | None, from_cache: bool = False) -> list[dict]:
    files = sorted(TRAIN_DIR.glob("lab-manual-*-train-*.xlsx"))
    dfs = {f.stem: load(f) for f in files}
    unique = pd.concat(dfs.values()).drop_duplicates(TEXT_COLUMN).reset_index(drop=True)
    if limit:
        unique = unique.head(limit)
    print(f"[train] {len(files)} files, {sum(map(len, dfs.values()))} rows, {len(unique)} unique sentences")

    if from_cache:
        pred_by_text = cached_predictions("train_gpt-5-nano_zero_shot")
        missing = [t for t in unique[TEXT_COLUMN] if t not in pred_by_text]
        if missing:
            print(f"[train] {len(missing)} of {len(unique)} sentences not in the cache; those rows are skipped")
    else:
        result = build(TRAIN_MODEL, "train_gpt-5-nano_zero_shot", 0).predict(train_df=None, test_df=unique)
        pred_by_text = dict(zip(unique[TEXT_COLUMN], result.predictions))

    out_dir = PRED_DIR / "train_gpt-5-nano_zero_shot"
    summary = []
    for name, df in dfs.items():
        df = df[df[TEXT_COLUMN].isin(pred_by_text)]
        if len(df):
            summary.append(save(df, df[TEXT_COLUMN].map(pred_by_text).tolist(), out_dir, name))
    return summary


def predict_test(limit: int | None, from_cache: bool = False) -> list[dict]:
    out_dir = PRED_DIR / f"test_gpt-5-nano_{TEST_SHOTS}_shot"
    summary = []
    for test_path in sorted(TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        train_path = TRAIN_DIR / test_path.name.replace("-test-", "-train-")
        if not train_path.exists():
            print(f"[test] skip {test_path.name}: no matching train file")
            continue
        test_df = load(test_path)
        if limit:
            test_df = test_df.head(limit)
        run_name = f"test_gpt-5-nano_{TEST_SHOTS}_shot_{test_path.stem}"
        if from_cache:
            cached = cached_predictions(run_name)
            missing = [t for t in test_df[TEXT_COLUMN] if t not in cached]
            if missing:
                print(f"[test] {test_path.stem}: {len(missing)} of {len(test_df)} rows not in the cache, skipped")
                continue
            print(f"[test] {test_path.stem}: {len(test_df)} rows from cache")
            summary.append(save(test_df, [cached[t] for t in test_df[TEXT_COLUMN]], out_dir, test_path.stem))
            continue
        shots = load(train_path).sample(n=TEST_SHOTS, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[test] {test_path.stem}: {len(test_df)} rows, {TEST_SHOTS} examples from {train_path.name}")

        model = build(TEST_MODEL, run_name, TEST_SHOTS)
        model.fit(shots)
        result = model.predict(train_df=shots, test_df=test_df)
        summary.append(save(test_df, result.predictions, out_dir, test_path.stem))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", nargs="?", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, help="only the first N sentences (smoke test)")
    parser.add_argument("--from-cache", action="store_true", help="rebuild outputs from the prediction cache, no API calls")
    args = parser.parse_args()

    for part, fn in (("train", predict_train), ("test", predict_test)):
        if args.part not in (part, "all"):
            continue
        summary = pd.DataFrame(fn(args.limit, args.from_cache))
        path = PRED_DIR / f"summary_{part}.csv"
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
        print(f"\n{summary.to_string(index=False)}\nSaved: {path}")


if __name__ == "__main__":
    main()
