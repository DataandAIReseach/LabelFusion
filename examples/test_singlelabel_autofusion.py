"""Runnable demo for single-label AutoFusion (safe fallback)

Creates a tiny single-label dataset and demonstrates AutoFusion flow. Falls back
to MockAutoFusion if the real class is not importable.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

class MockAutoFusion:
    def __init__(self, config):
        self.config = config
        self.label_columns = config.get('label_columns', [])
        self.multi_label = False

    def fit(self, df):
        print('MockAutoFusion.fit() called with', len(df), 'rows')

    def predict(self, df):
        preds = [[random.choice([0,1]) for _ in self.label_columns] for _ in range(len(df))]
        return type('R', (), {'predictions': preds, 'metadata': {}})()


def main():
    print('Single-label AutoFusion demo')

    df = pd.DataFrame({'text':[f'sample {i}' for i in range(12)], 'label':[random.choice([0,1]) for _ in range(12)]})
    train_df = df.sample(n=8, random_state=42).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    config = {'label_columns': ['label'], 'multi_label': False}

    try:
        from textclassify.ensemble.auto_fusion import AutoFusionClassifier
        print('Using real AutoFusionClassifier')
        clf = AutoFusionClassifier(config=config)
    except Exception:
        print('AutoFusionClassifier not available; using MockAutoFusion')
        clf = MockAutoFusion(config)

    clf.fit(train_df)
    res = clf.predict(test_df)
    print('Predictions sample:', res.predictions[:5])

if __name__ == '__main__':
    main()
