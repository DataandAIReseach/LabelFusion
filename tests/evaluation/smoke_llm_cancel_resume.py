"""Smoke test: simulate cancelling and resuming LLM predictions for FusionEnsemble.

This script creates a small toy DataFrame, a `FusionEnsemble` instance with
lightweight mock ML/LLM models and demonstrates:
 - starting incremental LLM prediction generation
 - simulating a cancellation (KeyboardInterrupt) part-way
 - resuming the incremental generation and completing predictions
 - calling `predict` with the cached/passed LLM predictions

Run: `python tests/evaluation/smoke_llm_cancel_resume.py`
"""
import os
import json
import pandas as pd
from types import SimpleNamespace

from textclassify.ensemble.fusion import FusionEnsemble


class SimpleEnsembleConfig:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}


class MockML:
    def __init__(self, text_column='text', label_columns=None, classes=None):
        self.text_column = text_column
        self.label_columns = label_columns or ['label_a', 'label_b']
        self.is_trained = True
        self.classes_ = classes or ['class_a', 'class_b']
        # provide a dummy .model with parameters() for FusionWrapper compatibility
        class DummyModel:
            def parameters(self):
                return iter([])
        self.model = DummyModel()

    def predict(self, df):
        # return a simple object with .predictions attribute: binary vectors
        preds = [[1, 0] if i % 2 == 0 else [0, 1] for i in range(len(df))]
        return SimpleNamespace(predictions=preds, metadata={})

    def predict_without_saving(self, df):
        return self.predict(df)


class MockLLM:
    def __init__(self, classes=None, interrupt_at_call=None):
        self.classes_ = classes or ['class_a', 'class_b']
        self.text_column = 'text'
        self.call_count = 0
        self.interrupt_at_call = interrupt_at_call

    def predict(self, train_df=None, test_df=None):
        # Simulate incremental long-running predictions by raising KeyboardInterrupt
        # on a configured call to simulate cancellation.
        self.call_count += 1
        if self.interrupt_at_call is not None and self.call_count == self.interrupt_at_call:
            raise KeyboardInterrupt("Simulated user cancellation of LLM predictions")

        # Produce predictions as list of lists of class names (multi-label style)
        preds = []
        for i in range(len(test_df)):
            # alternate between single-class and multi-class predictions
            if i % 3 == 0:
                preds.append([self.classes_[0]])
            elif i % 3 == 1:
                preds.append([self.classes_[1]])
            else:
                preds.append(self.classes_)

        return SimpleNamespace(predictions=preds, metadata={})


def read_ndjson_predictions(ndjson_path, num_rows):
    """Read predictions written by generate_llm_predictions_incremental and return
    a list of predictions ordered by row index (None for missing)."""
    preds = [None] * num_rows
    if not os.path.exists(ndjson_path):
        return preds

    with open(ndjson_path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                idx = obj.get('row_index')
                if idx is not None and 0 <= idx < num_rows:
                    preds[idx] = obj.get('prediction')
            except Exception:
                continue
    return preds


def main():
    # Create a small toy dataset
    n = 20
    df = pd.DataFrame({'text': [f"Example text {i}" for i in range(n)],
                       'label_a': [1 if i % 2 == 0 else 0 for i in range(n)],
                       'label_b': [0 if i % 2 == 0 else 1 for i in range(n)]})

    # Prepare ensemble config and instantiate FusionEnsemble
    params = {
        'fusion_hidden_dims': [16, 8],
        'classification_type': 'multi_label',
        'output_dir': 'outputs/smoke_llm',
        'experiment_name': 'smoke_llm'
    }
    cfg = SimpleEnsembleConfig(parameters=params)
    fusion = FusionEnsemble(cfg, output_dir='outputs/smoke_llm', experiment_name='smoke_llm')

    # Attach mock ML and mock LLM (LLM will be set to interrupt on 2nd call)
    mock_ml = MockML()
    mock_llm = MockLLM(interrupt_at_call=2)
    fusion.add_ml_model(mock_ml)
    fusion.add_llm_model(mock_llm)

    # Set required internal state so predict() can run without full training
    fusion.classes_ = mock_ml.classes_
    fusion.num_labels = len(fusion.classes_)
    fusion.fusion_wrapper = fusion.fusion_wrapper or None
    # Create a proper FusionWrapper so _predict_with_fusion works
    from textclassify.ensemble.fusion import FusionWrapper
    fusion.fusion_wrapper = FusionWrapper(ml_model=mock_ml, num_labels=fusion.num_labels, hidden_dims=[16, 8])
    fusion.ml_model = mock_ml
    fusion.llm_model = mock_llm
    fusion.is_trained = True

    # Ensure cache dir exists
    cache_base = 'cache/smoke_test/test'
    cache_dir = os.path.dirname(cache_base)
    os.makedirs(cache_dir, exist_ok=True)
    ndjson_path = f"{cache_base}.ndjson"

    # 1) Run incremental prediction and simulate cancellation
    print("\n--- Starting incremental LLM prediction (will simulate cancellation) ---")
    try:
        fusion.generate_llm_predictions_incremental(df, cache_base, batch_size=5)
    except KeyboardInterrupt as e:
        print(f"Caught interruption: {e}")

    # Inspect partial cache
    partial_preds = read_ndjson_predictions(ndjson_path, n)
    num_written = sum(1 for p in partial_preds if p is not None)
    print(f"Partial predictions written: {num_written}/{n}")

    # 2) Resume by removing interrupt and running again
    print("\n--- Resuming incremental LLM prediction (should skip already-written rows) ---")
    # disable interruption
    mock_llm.interrupt_at_call = None
    fusion.generate_llm_predictions_incremental(df, cache_base, batch_size=5)

    # Verify full predictions present
    final_preds = read_ndjson_predictions(ndjson_path, n)
    num_final = sum(1 for p in final_preds if p is not None)
    print(f"Final predictions written: {num_final}/{n}")

    # 3) Call fusion.predict passing the cached predictions (ordered list)
    # Build ordered list of predictions from ndjson
    ordered_preds = final_preds
    print("\n--- Calling fusion.predict() with provided LLM predictions ---")
    result = fusion.predict(test_df=df, true_labels=None, test_llm_predictions=ordered_preds)

    print("\nFusion prediction completed.")
    print(f"Number of fusion predictions: {len(result.predictions)}")


if __name__ == '__main__':
    main()
