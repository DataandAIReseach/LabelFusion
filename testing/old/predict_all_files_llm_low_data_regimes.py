"""LLM only (GPT-5 nano by default, or Claude; few-shot) with the training data in low-data regimes: always the FULL test sets.

The test sets stay as they are (data/test_data/<test name>.xlsx, always the same sentences). What changes with the
regime p (10, 20, ..., 100 %) is the training data the prompt's examples are drawn from, and how many there are:
2 examples per 10 % of the training data, i.e. 10 % -> 2 examples, 20 % -> 4, ..., 100 % -> 20.

    data/training_data_dated_low_data_regimes/<train name>_<p>.csv   (made by dataset_scripts/make_low_data_regimes.py)

(<train name> is <test name> with -test- -> -train-). The examples are drawn at random (project seed) from that
regime. No model is trained: this is the LLM-only baseline for the low-data regimes of predict_all_labelFusion*.py.

Provider / model: --provider openrouter (default; key OPENROUTER_API_KEY, model ids like openai/gpt-5-nano, also
anthropic/claude-sonnet-4.5) or --provider anthropic (the Claude API directly; key ANTHROPIC_API_KEY, or CLAUDE_API_KEY,
model ids like claude-haiku-5-5). --model overrides the default model of the provider.

Writes to ./outputs/predictions_low_data_regimes/ (<model> = the model id without the vendor prefix):
    test_<model>_2_per_10_percent/<test name>_<p>.csv   sentence, label, true, pred (the columns of predict_all_files_llm.py)
    summary_test_<model>.csv                            per test file and regime: examples, rows, accuracy, macro-F1, weighted-F1
Finished files are skipped on a rerun (--force repeats them); the classifier's prediction cache lives in
outputs/llm_cache like for predict_all_files_llm.py, so no sentence is paid for twice.

    python testing/old/predict_all_files_llm_low_data_regimes.py --dry-run                       # what would be classified, no API calls
    python testing/old/predict_all_files_llm_low_data_regimes.py                                 # every test file, every regime
    python testing/old/predict_all_files_llm_low_data_regimes.py --datasets mm-test --percents 10 50
    python testing/old/predict_all_files_llm_low_data_regimes.py --limit 20                      # smoke test: first 20 sentences per file
    python testing/old/predict_all_files_llm_low_data_regimes.py --provider anthropic            # Claude (claude-haiku-5-5) with an Anthropic key
    python testing/old/predict_all_files_llm_low_data_regimes.py --provider anthropic --model claude-sonnet-5-5
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
from test_textclassify_llm import (  # noqa: E402
    LABEL_COLUMNS, LABEL_MAP, RANDOM_STATE, REPO_ROOT, TEXT_COLUMN, _add_onehot_labels,
)
from textclassify import AnthropicClassifier  # noqa: E402
from textclassify.config.api_keys import APIKeyManager  # noqa: E402
from textclassify.core.types import ModelConfig, ModelType  # noqa: E402

TRAIN_REGIME_DIR = REPO_ROOT / "data" / "training_data_dated_low_data_regimes"
OUT_DIR = llm.OUT_DIR / "predictions_low_data_regimes"
SHOTS_PER_10_PERCENT = 2
DEFAULT_MODELS = {"openrouter": llm.TEST_MODEL, "anthropic": "claude-haiku-5-5"}
REGIME_RE = re.compile(r"_(?P<percent>\d+)$")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _add_onehot_labels(df[[TEXT_COLUMN, "label"]].copy()).reset_index(drop=True)


def model_slug(model: str) -> str:
    """openai/gpt-5-nano -> gpt-5-nano (used in file and folder names)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model.split("/")[-1])


def build_classifier(provider: str, model: str, run_name: str, n_shots: int):
    """The few-shot classifier of the chosen provider (OpenRouter as in predict_all_files_llm.py, or Claude directly)."""
    if provider == "openrouter":
        return llm.build(model, run_name, n_shots)
    config = ModelConfig(model_name=model, model_type=ModelType.LLM,
                         parameters={"model": model, "temperature": 0.1, "max_completion_tokens": 1024})
    return AnthropicClassifier(
        config=config, text_column=TEXT_COLUMN, label_columns=LABEL_COLUMNS, multi_label=False, few_shot_mode=n_shots,
        output_dir=str(llm.OUT_DIR), experiment_name=f"predict_all_{run_name}",
        cache_dir=str(llm.OUT_DIR / "llm_cache" / f"predict_all_{run_name}"),
    )


def check_api_key(provider: str) -> None:
    keys = APIKeyManager()
    names = {"openrouter": ("openrouter", "openai"), "anthropic": ("anthropic", "claude")}[provider]
    if not any(keys.get_key(n) for n in names):
        raise SystemExit(f"no API key for --provider {provider}: set {' or '.join(n.upper() + '_API_KEY' for n in names)} (e.g. in .env)")


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
    parser.add_argument("--provider", choices=("openrouter", "anthropic"), default="openrouter",
                        help="openrouter (OPENROUTER_API_KEY) or anthropic (ANTHROPIC_API_KEY or CLAUDE_API_KEY): the Claude API directly")
    parser.add_argument("--model", help="model id (default: openai/gpt-5-nano for openrouter, claude-haiku-5-5 for anthropic)")
    parser.add_argument("--percents", nargs="*", type=int, help="only these regimes (default: all found)")
    parser.add_argument("--limit", type=int, help="only the first N sentences of every test file (smoke test)")
    parser.add_argument("--force", action="store_true", help="classify files again that already have an output")
    parser.add_argument("--dry-run", action="store_true", help="list what would be classified, no API calls")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    slug = model_slug(model)
    run_dir = OUT_DIR / f"test_{slug}_{SHOTS_PER_10_PERCENT}_per_10_percent"
    todo = jobs(args.datasets, args.percents)
    if not todo:
        raise SystemExit(f"no train regime files found in {TRAIN_REGIME_DIR} (run dataset_scripts/make_low_data_regimes.py)")

    sizes = {t: len(llm.load(t)) for t in {j[0] for j in todo}}
    calls = sum(sizes[j[0]] for j in todo)
    print(f"{len(todo)} test file x regime runs, {calls} sentences to classify with {model} ({args.provider}), "
          f"{', '.join(str(shots_for(p)) for p in sorted({j[2] for j in todo}))} examples for the regimes "
          f"{', '.join(str(p) + '%' for p in sorted({j[2] for j in todo}))}")
    if args.dry_run:
        for test_path, train_path, percent in todo[:12]:
            print(f"  {test_path.stem}  <-  {shots_for(percent)} examples from {train_path.name}")
        print("  ..." if len(todo) > 12 else "")
        return

    check_api_key(args.provider)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"summary_test_{slug}.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    for test_path, train_path, percent in todo:
        name = f"{test_path.stem}_{percent}"
        out_path = run_dir / f"{name}.csv"
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

        classifier = build_classifier(args.provider, model, f"test_{slug}_{n_shots}_shot_{name}", n_shots)
        classifier.fit(shots)
        preds = classifier.predict(train_df=shots, test_df=test_df).predictions

        out = test_df[[TEXT_COLUMN, "label"]].copy()
        out["true"] = out["label"].map(LABEL_MAP)
        out["pred"] = preds
        out.to_csv(out_path, index=False)

        row = {"file": name, "dataset": test_path.stem, "percent": percent, "provider": args.provider, "model": model, "examples": n_shots, "rows": len(out), "shots_from": train_path.name,
               "accuracy": accuracy_score(out["true"], out["pred"]),
               "f1_macro": f1_score(out["true"], out["pred"], average="macro"),
               "f1_weighted": f1_score(out["true"], out["pred"], average="weighted")}
        if len(summary):
            summary = summary[summary["file"] != row["file"]]
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
        summary.sort_values(["dataset", "percent"]).to_csv(summary_path, index=False)  # kept up to date after every file

    print(f"\nSaved: {run_dir} and {summary_path}")


if __name__ == "__main__":
    main()
