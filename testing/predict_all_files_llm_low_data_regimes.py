"""LLM only (GPT-5 nano, few-shot) with the training data in low-data regimes: always the FULL test sets.

The test sets stay as they are (data/test_data/<test name>.xlsx, always the same sentences). What changes with the
regime p (10, 20, ..., 100 %) is the training data the prompt's examples are drawn from, and how many there are:
2 examples per 10 % of the training data, i.e. 10 % -> 2 examples, 20 % -> 4, ..., 100 % -> 20.

    data/training_data_dated_low_data_regimes/<train name>_<p>.csv   (made by dataset_scripts/make_low_data_regimes.py)

(<train name> is <test name> with -test- -> -train-). The examples are drawn at random (project seed) from that
regime. No model is trained: this is the LLM-only baseline for the low-data regimes of predict_all_labelFusion*.py.

Writes to ./outputs/predictions_low_data_regimes/:
    test_gpt-5-nano_2_per_10_percent/<test name>_<p>.csv   sentence, label, true, pred (the columns of predict_all_files_llm.py)
    summary_test.csv                                       per test file and regime: examples, rows, accuracy, macro-F1, weighted-F1
Finished files are skipped on a rerun (--force repeats them); the classifier's prediction cache lives in
outputs/llm_cache like for predict_all_files_llm.py, so no sentence is paid for twice.

    python testing/predict_all_files_llm_low_data_regimes.py --dry-run                       # what would be classified, no API calls
    python testing/predict_all_files_llm_low_data_regimes.py                                 # every test file, every regime
    python testing/predict_all_files_llm_low_data_regimes.py --datasets mm-test --percents 10 50
    python testing/predict_all_files_llm_low_data_regimes.py --limit 20                      # smoke test: first 20 sentences per file
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

import predict_all_files_llm as llm  # noqa: E402
from test_textclassify_llm import LABEL_MAP, RANDOM_STATE, REPO_ROOT, TEXT_COLUMN, _add_onehot_labels  # noqa: E402

TRAIN_REGIME_DIR = REPO_ROOT / "data" / "training_data_dated_low_data_regimes"
OUT_DIR = llm.OUT_DIR / "predictions_low_data_regimes"
RUN_DIR = OUT_DIR / "test_gpt-5-nano_2_per_10_percent"
SHOTS_PER_10_PERCENT = 2
REGIME_RE = re.compile(r"_(?P<percent>\d+)$")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _add_onehot_labels(df[[TEXT_COLUMN, "label"]].copy()).reset_index(drop=True)


def shots_for(percent: int) -> int:
    """2 examples per 10 % of the training data: 10 % -> 2, 20 % -> 4, ..., 100 % -> 20."""
    return max(1, round(SHOTS_PER_10_PERCENT * percent / 10))


def jobs(datasets: list[str] | None, percents: list[int] | None) -> list[tuple[Path, Path, int]]:
    """(test file, train regime file, percent) for every test file and every regime that exists."""
    found = []
    for test_path in sorted(llm.TEST_DIR.glob("lab-manual-*-test-*.xlsx")):
        if datasets and not any(d in test_path.stem for d in datasets):
            continue
        train_name = test_path.stem.replace("-test-", "-train-")
        for train_path in sorted(TRAIN_REGIME_DIR.glob(f"{train_name}_*.csv")):
            m = REGIME_RE.search(train_path.stem)
            if m and (not percents or int(m["percent"]) in percents):
                found.append((test_path, train_path, int(m["percent"])))
    return sorted(found, key=lambda j: (j[0].stem, j[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="only test files (name without seed) containing one of these")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--limit", type=int, help="only the first N sentences of every test file (smoke test)")
    parser.add_argument("--force", action="store_true", help="classify files again that already have an output")
    parser.add_argument("--dry-run", action="store_true", help="list what would be classified, no API calls")
    args = parser.parse_args()

    todo = jobs(args.datasets, args.percents)
    if not todo:
        raise SystemExit(f"no train regime files found in {TRAIN_REGIME_DIR} (run dataset_scripts/make_low_data_regimes.py)")

    sizes = {t: len(llm.load(t)) for t in {j[0] for j in todo}}
    calls = sum(sizes[j[0]] for j in todo)
    print(f"{len(todo)} test file x regime runs, {calls} sentences to classify with {llm.TEST_MODEL}, "
          f"{', '.join(str(shots_for(p)) for p in sorted({j[2] for j in todo}))} examples for the regimes "
          f"{', '.join(str(p) + '%' for p in sorted({j[2] for j in todo}))}")
    if args.dry_run:
        for test_path, train_path, percent in todo[:12]:
            print(f"  {test_path.stem}  <-  {shots_for(percent)} examples from {train_path.name}")
        print("  ..." if len(todo) > 12 else "")
        return

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary_test.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    for test_path, train_path, percent in todo:
        name = f"{test_path.stem}_{percent}"
        out_path = RUN_DIR / f"{name}.csv"
        if out_path.exists() and not args.force:
            print(f"skip {name}: already classified")
            continue
        test_df = llm.load(test_path)
        if args.limit:
            test_df = test_df.head(args.limit)
        train_df = load_csv(train_path)
        n_shots = min(shots_for(percent), len(train_df))
        shots = train_df.sample(n=n_shots, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[{name}] {len(test_df)} sentences, {n_shots} examples from {train_path.name} ({len(train_df)} rows)")

        model = llm.build(llm.TEST_MODEL, f"test_gpt-5-nano_{n_shots}_shot_{name}", n_shots)
        model.fit(shots)
        preds = model.predict(train_df=shots, test_df=test_df).predictions

        out = test_df[[TEXT_COLUMN, "label"]].copy()
        out["true"] = out["label"].map(LABEL_MAP)
        out["pred"] = preds
        out.to_csv(out_path, index=False)

        row = {"file": name, "dataset": test_path.stem, "percent": percent, "examples": n_shots, "rows": len(out), "shots_from": train_path.name,
               "accuracy": accuracy_score(out["true"], out["pred"]),
               "f1_macro": f1_score(out["true"], out["pred"], average="macro"),
               "f1_weighted": f1_score(out["true"], out["pred"], average="weighted")}
        if len(summary):
            summary = summary[summary["file"] != row["file"]]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.sort_values(["dataset", "percent"]).to_csv(summary_path, index=False)  # kept up to date after every file

    print(f"\nSaved: {RUN_DIR} and {summary_path}")


if __name__ == "__main__":
    main()
