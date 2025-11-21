"""
Smoke resume test for Reuters-like classes.

This mirrors `smoke_llm_resume.py` but uses Reuters-style class names and runs the same
interruption-and-resume validation for train/val/test datasets of size n=10.

It uses a `MockLLM` to avoid any external API calls.
"""

import os
import json
from pathlib import Path
import pandas as pd
import random
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from textclassify.ensemble.fusion import FusionEnsemble
from textclassify.core.types import EnsembleConfig


class MockLLM:
    def __init__(self, text_column='text', classes=None, fail_after=None):
        self.text_column = text_column
        self.classes_ = classes or [f"class_{i}" for i in range(4)]
        self.fail_after = fail_after
        self._counter = 0

    def predict(self, train_df, test_df):
        preds = []
        for _idx, row in test_df.iterrows():
            cls = self.classes_[self._counter % len(self.classes_)]
            preds.append(cls)
            self._counter += 1
            if self.fail_after is not None and self._counter > self.fail_after:
                raise KeyboardInterrupt("Simulated cancellation during LLM prediction")
        class R: pass
        r = R()
        r.predictions = preds
        return r


def make_reuters_df(n=10, text_col='text', classes=None):
    classes = classes or ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
    texts = [f"reuters sample {i}" for i in range(n)]
    labels = [random.choice(classes) for _ in range(n)]

    label_columns = [f"label_{i}" for i in range(len(classes))]
    rows = []
    for t, l in zip(texts, labels):
        vec = [1 if c == l else 0 for c in classes]
        row = {text_col: t}
        for col, v in zip(label_columns, vec):
            row[col] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, text_col, label_columns, classes


def run_resume_test(df, cache_base, classes, text_column='text', batch_size=4, fail_after=3):
    # Provide a non-empty models list to satisfy BaseEnsemble init
    cfg = EnsembleConfig(ensemble_method='fusion', models=[{"placeholder": True}], parameters={})
    fusion = FusionEnsemble(cfg)

    # Attach mock LLM
    mock = MockLLM(text_column=text_column, classes=classes, fail_after=fail_after)
    fusion.add_llm_model(mock)

    ndjson_path = f"{cache_base}.ndjson"

    if os.path.exists(ndjson_path):
        os.remove(ndjson_path)

    print(f"\n--- First (interrupted) pass for cache '{cache_base}' ---")
    try:
        fusion.generate_llm_predictions_incremental(df, cache_base, batch_size=batch_size)
        print("First pass completed without interruption (unexpected)")
    except KeyboardInterrupt as e:
        print(f"First pass interrupted as expected: {e}")

    processed = 0
    if os.path.exists(ndjson_path):
        with open(ndjson_path, 'r', encoding='utf8') as f:
            for _ in f:
                processed += 1
    print(f"After interruption, cache contains {processed}/{len(df)} entries")

    print(f"\n--- Second (resume) pass for cache '{cache_base}' ---")
    mock2 = MockLLM(text_column=text_column, classes=classes, fail_after=None)
    fusion.llm_model = mock2

    try:
        preds = fusion.generate_llm_predictions_incremental(df, cache_base, batch_size=batch_size)
        print(f"Second pass returned {len(preds)} items (including previously saved ones)")
    except Exception as e:
        print(f"Second pass failed unexpectedly: {e}")
        raise

    final_count = 0
    with open(ndjson_path, 'r', encoding='utf8') as f:
        for line in f:
            final_count += 1
    success = final_count == len(df)
    print(f"After resume, cache contains {final_count}/{len(df)} entries -> {'SUCCESS' if success else 'FAIL'}")
    return success


if __name__ == '__main__':
    base_cache_dir = Path('cache/smoke_llm_resume_reuters')
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    df_train, text_col, label_cols, classes = make_reuters_df(n=10)
    df_val, _, _, _ = make_reuters_df(n=10)
    df_test, _, _, _ = make_reuters_df(n=10)

    all_ok = True

    for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        cache_base = str(base_cache_dir / f"{name}_preds")
        ok = run_resume_test(df, cache_base, classes, text_column=text_col, batch_size=4, fail_after=3)
        all_ok = all_ok and ok

    if all_ok:
        print("\nALL REUTERS RESUME TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME REUTERS RESUME TESTS FAILED")
        sys.exit(2)
