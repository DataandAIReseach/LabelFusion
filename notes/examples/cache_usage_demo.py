"""Cache Usage Demo

Lightweight, runnable demo that shows how cached LLM predictions
can be discovered, inspected, and used by the fusion helpers.
This demo uses safe mocks when optional dependencies (OpenAI/DeepSeek)
are not available so it can run in minimal environments.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Minimal Mock LLM and Fusion helpers for demo purposes
class MockLLM:
    def __init__(self):
        self.provider = 'mock'

    def predict(self, train_df=None, test_df=None, **kwargs):
        texts = list(test_df['text']) if test_df is not None else []
        preds = [[random.choice([0,1])] for _ in texts]
        # return object with .predictions and .metadata to match real classifiers
        return type('R', (), {'predictions': preds, 'metadata': {'provider':'mock'}})()


def demo_cache_usage():
    print('Cache usage demo (mock)')
    # Tiny datasets
    train_df = pd.DataFrame({'text':['train a', 'train b'], 'label':[1,0]})
    val_df = pd.DataFrame({'text':['val a','val b'], 'label':[1,0]})
    test_df = pd.DataFrame({'text':['test a','test b'], 'label':[1,0]})

    # Try to import FusionEnsemble and LLMPredictionCache; fall back to mocks
    try:
        from textclassify.ensemble.fusion import FusionEnsemble
        from textclassify.llm.prediction_cache import LLMPredictionCache
        print('Loaded real FusionEnsemble and LLMPredictionCache')
    except Exception:
        FusionEnsemble = None
        LLMPredictionCache = None
        print('Using mock behavior (FusionEnsemble not available)')

    # Use mock LLM to produce predictions and save simple JSON cache files
    llm = MockLLM()
    val_res = llm.predict(train_df=train_df, test_df=val_df)
    test_res = llm.predict(train_df=train_df, test_df=test_df)

    cache_dir = project_root / 'cache' / 'demo_llm_cache'
    os.makedirs(cache_dir, exist_ok=True)
    val_file = cache_dir / 'validation_predictions.json'
    test_file = cache_dir / 'test_predictions.json'

    with open(val_file, 'w') as f:
        json.dump({'predictions': val_res.predictions, 'provider': llm.provider}, f)
    with open(test_file, 'w') as f:
        json.dump({'predictions': test_res.predictions, 'provider': llm.provider}, f)

    print('Wrote demo cache files:')
    print(' ', val_file)
    print(' ', test_file)

    # If FusionEnsemble is available we could show how to load these files;
    # otherwise, just print discovery info.
    if LLMPredictionCache is not None:
        cache = LLMPredictionCache(cache_dir=str(cache_dir), verbose=False)
        print('Cache stats:', cache.get_cache_stats())
    else:
        print('Cache discovery (mock):', [str(p) for p in cache_dir.glob('*.json')])


if __name__ == '__main__':
    demo_cache_usage()
