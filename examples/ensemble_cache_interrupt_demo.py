"""Ensemble Cache Interrupt Demo

Creates small val/test sets and demonstrates saving LLM predictions to
cache using Fusion utilities when available. Falls back to safe mocks.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

class MockLLM:
    def __init__(self):
        self.provider = 'mock'

    def predict(self, train_df=None, test_df=None, **kwargs):
        texts = list(test_df['text']) if test_df is not None else []
        preds = [[random.choice([0,1])] for _ in texts]
        return type('R', (), {'predictions': preds, 'metadata': {}})()

class MockML:
    def predict(self, df):
        return [0 for _ in range(len(df))]


def main():
    print('Ensemble cache interrupt demo (mock)')
    df_val = pd.DataFrame({'text':[f'val {i}' for i in range(5)]})
    df_test = pd.DataFrame({'text':[f'test {i}' for i in range(5)]})

    try:
        from textclassify.ensemble.fusion import FusionEnsemble
        fusion_available = True
        print('FusionEnsemble available')
    except Exception:
        fusion_available = False
        print('FusionEnsemble not available; using mocks')

    llm = MockLLM()
    val_res = llm.predict(train_df=None, test_df=df_val)
    test_res = llm.predict(train_df=None, test_df=df_test)

    cache_dir = project_root / 'cache' / 'mock_ensemble'
    os.makedirs(cache_dir / 'val', exist_ok=True)
    os.makedirs(cache_dir / 'test', exist_ok=True)

    # Save simple JSON caches
    import json
    with open(cache_dir / 'val' / 'preds.json', 'w') as f:
        json.dump({'predictions': val_res.predictions, 'provider': 'mock'}, f)
    with open(cache_dir / 'test' / 'preds.json', 'w') as f:
        json.dump({'predictions': test_res.predictions, 'provider': 'mock'}, f)

    print('Saved mock cache files under', cache_dir)

if __name__ == '__main__':
    main()
