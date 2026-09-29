"""LabelFusion on exactly one train/test pair, with standard (non-tuned) hyperparameters.

A minimal counterpart to predict_all_files_labelFusion.py: no Optuna, no loop over every
pair, no resume/skip logic. Useful to sanity-check the environment/pipeline or to get a
quick default-hyperparameter result on a single file without paying for tuning.

Reuses the building blocks (PrecomputedLLM, EmbeddingMemo, fit_predict, default_params, ...)
from predict_all_files_labelFusion.py so behaviour matches the --trials 0 case of that script.

Writes ./outputs/fusion/single/<test file name>.csv and .json (sentence, label, true, llm_pred,
roberta_pred, fusion_pred) and prints accuracy / macro-F1 for the LLM, RoBERTa and the fusion.

    python testing/predict_single_pair_labelFusion.py                                   # default pair
    python testing/predict_single_pair_labelFusion.py --test lab-manual-pc-test-5768     # by test file stem
    python testing/predict_single_pair_labelFusion.py --test lab-manual-pc-test-5768 --limit-train 60 --epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING_DIR))

from predict_all_files_labelFusion import (  # noqa: E402
    FUSION_DIR, LLM_TEST_DIR, LLM_TRAIN_DIR, TEST_DIR, TEXT_COLUMN, LABEL_MAP,
    PrecomputedLLM, default_params, fit_predict, llm_predictions, load, paired_train, scores, split_pair,
)

import pandas as pd  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="lab-manual-mm-test-5768", help="test file stem (in data/test_data)")
    parser.add_argument("--epochs", type=int, default=3, help="RoBERTa epochs of the default configuration")
    parser.add_argument("--fusion-epochs", type=int, default=30, help="fusion MLP epochs of the default configuration")
    parser.add_argument("--limit-train", type=int, help="subsample the train file to N rows (smoke test)")
    args = parser.parse_args()

    out_dir = FUSION_DIR / "single"
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(FUSION_DIR)  # FusionEnsemble writes its prediction caches to ./cache

    test_path = TEST_DIR / f"{args.test}.xlsx"
    train_path = paired_train(test_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    name = test_path.stem
    t0 = time.time()
    train_df, val_df = split_pair(train_path, args)
    test_df = load(test_path)
    print(f"=== {name}: train {len(train_df)} / val {len(val_df)} / test {len(test_df)} ===")

    llm = PrecomputedLLM(llm_predictions(LLM_TRAIN_DIR / f"{train_path.stem}.csv"))
    llm_test = llm_predictions(LLM_TEST_DIR / f"{test_path.stem}.csv")
    llm_test_preds = [llm_test[t] for t in test_df[TEXT_COLUMN]]

    rob, fus = default_params(args)
    fusion_preds, roberta_preds = fit_predict(
        rob, fus, train_df, val_df, test_df, llm_test_preds, llm, name, "single", save=True, verbose=True
    )

    out = pd.DataFrame({
        TEXT_COLUMN: test_df[TEXT_COLUMN],
        "label": test_df["label"],
        "true": test_df["label"].map(LABEL_MAP),
        "llm_pred": llm_test_preds,
        "roberta_pred": roberta_preds,
        "fusion_pred": fusion_preds,
    })
    shutil.rmtree(FUSION_DIR / "cache" / "predictions", ignore_errors=True)
    out.to_csv(out_dir / f"{name}.csv", index=False)
    out.to_json(out_dir / f"{name}.json", orient="records", force_ascii=False, indent=2)

    row = {"file": name, "rows": len(out), "seconds": round(time.time() - t0)}
    for who in ("llm", "roberta", "fusion"):
        row.update({f"{who}_{m}": v for m, v in scores(out["true"].tolist(), out[f"{who}_pred"].tolist()).items()})
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(f"Saved: {out_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
