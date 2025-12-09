"""Runnable demo for single-label ML-only classifier (safe fallback)

This demo trains a tiny RoBERTa-based ML classifier if available, otherwise
it uses a MockML to simulate training and prediction.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

class MockML:
    def __init__(self):
        self.model_name = 'mock'
        self.label_columns = ['label']

    def fit(self, df):
        print('MockML.fit() called with', len(df), 'rows')

    def predict(self, df):
        return type('R', (), {'predictions': [[random.choice([0,1])] for _ in range(len(df))], 'metadata': {}})()


def main():
    print('Single-label ML demo')
    # tiny dataset
    df = pd.DataFrame({'text':[f'sample {i}' for i in range(30)], 'label':[random.choice([0,1]) for _ in range(30)]})
    train_df = df.sample(n=20, random_state=42).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    try:
        from textclassify.ml.roberta_classifier import RoBERTaClassifier
        from textclassify.core.types import ModelConfig, ModelType
        print('Using real RoBERTaClassifier')
        cfg = ModelConfig(model_name='roberta-base', model_type=ModelType.TRADITIONAL_ML, parameters={})
        clf = RoBERTaClassifier(config=cfg, text_column='text', label_columns=['label'], multi_label=False, auto_save_results=False)
    except Exception:
        print('RoBERTaClassifier not available; using MockML')
        clf = MockML()

    clf.fit(train_df)
    res = clf.predict(test_df)
    print('Sample predictions:', res.predictions[:5])

if __name__ == '__main__':
    main()
