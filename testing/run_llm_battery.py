"""Battery of LLM few-shot runs: models x number of training examples.

Runs testing/test_textclassify_llm.py once per (model, sample_size) as a
subprocess (the script reads its config from env vars at import time) and
prints/saves a summary table.

    python testing/run_llm_battery.py                  # all models
    python testing/run_llm_battery.py gpt-5-nano ...   # only these model keys

Results are merged into the CSV by (model, train_samples).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

TESTING_DIR = Path(__file__).resolve().parent
SCRIPT = TESTING_DIR / "test_textclassify_llm.py"
OUT_CSV = TESTING_DIR / "outputs" / "llm_battery_results.csv"

MODELS = {
    "haiku": "anthropic/claude-haiku-4.5",
    "sonnet": "anthropic/claude-sonnet-4.5",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5-mini": "openai/gpt-5-mini",
}
# Reasoning models need headroom for hidden reasoning tokens
MAX_TOKENS = {"gpt-5-nano": 2000, "gpt-5-mini": 2000, "gemini-2.5-flash": 1000}
SAMPLE_SIZES = [0, 1, 5]
MAX_PARALLEL = 3
METRIC_RE = re.compile(r"^\s{2}(accuracy|precision|recall|f1|auc): ([\d.]+)$", re.M)


def run_one(model_key: str, sample_size: int) -> dict:
    env = {
        **os.environ,
        "RUN_TEXTCLASSIFY_FUSION_TEST": "1",
        "TEXTCLASSIFY_SAMPLE_SIZE": str(sample_size),
        "TEXTCLASSIFY_LLM_MODEL": MODELS[model_key],
        "TEXTCLASSIFY_MAX_TOKENS": str(MAX_TOKENS.get(model_key, 100)),
    }
    proc = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
    row = {"model": model_key, "train_samples": sample_size}
    if proc.returncode != 0:
        row["error"] = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"
        return row
    # Metrics are printed after the "Test metrics:" header (last occurrence).
    tail = proc.stdout.rsplit("Test metrics:", 1)[-1]
    row.update({k: float(v) for k, v in METRIC_RE.findall(tail)})
    print(f"done: {model_key} n={sample_size}", flush=True)
    return row


def main() -> None:
    selected = sys.argv[1:] or list(MODELS)
    jobs = [(m, n) for m in selected for n in SAMPLE_SIZES]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        rows = list(pool.map(lambda job: run_one(*job), jobs))
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        keys = set(zip(df["model"], df["train_samples"]))
        old = old[[k not in keys for k in zip(old["model"], old["train_samples"])]]
        df = pd.concat([old, df], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
