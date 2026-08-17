"""Low data regime evaluation with baseline methods.

Evaluates baseline multilabel text classification methods:
1. Frequency-based random sampling baseline
2. TF-IDF + Logistic Regression (One-vs-Rest) baseline

Environment variables:
    - DATA_DIR: path to Reuters data (default /scratch/users/u19147/LabelFusion/data/reuters)
    - OUTPUT_DIR: path to write outputs (default outputs/baselines)
"""


import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report
)

# ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_datasets(data_dir: str = "/scratch/users/u19147/LabelFusion/data/reuters"):
    """Load train, validation, and test datasets."""
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    df_val = pd.read_csv(os.path.join(data_dir, "val.csv")).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv")).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    return df_train, df_val, df_test


def detect_label_columns(df: pd.DataFrame, text_column: str = 'text') -> List[str]:
    """Detect binary label columns in the dataframe."""
    candidate = [c for c in df.columns if c != text_column and not c.lower().startswith('id')]
    labels = []
    for c in candidate:
        ser = df[c]
        try:
            if pd.api.types.is_numeric_dtype(ser):
                uniques = set(pd.Series(ser).dropna().unique())
                if uniques.issubset({0, 1, 0.0, 1.0}):
                    labels.append(c)
        except Exception:
            continue
    if not labels:
        labels = [c for c in df.columns if c != text_column]
    return labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute multilabel classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
    }


class FrequencyRandomBaseline:
    """Frequency-based random sampling baseline.
    
    Predicts each label with probability equal to its empirical frequency
    in the training data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_frequencies_ = None
        
    def fit(self, y_train: np.ndarray) -> 'FrequencyRandomBaseline':
        """Compute label frequencies from training data.
        
        Args:
            y_train: Binary label matrix (n_samples × n_labels)
        """
        self.label_frequencies_ = y_train.mean(axis=0)
        return self
    
    def predict(self, n_samples: int) -> np.ndarray:
        """Generate random predictions based on label frequencies.
        
        Args:
            n_samples: Number of samples to predict
            
        Returns:
            Binary predictions (n_samples × n_labels)
        """
        if self.label_frequencies_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        np.random.seed(self.random_state)
        random_vals = np.random.rand(n_samples, len(self.label_frequencies_))
        predictions = (random_vals < self.label_frequencies_).astype(int)
        return predictions


class TfidfLogisticRegressionBaseline:
    """TF-IDF + Logistic Regression (One-vs-Rest) baseline.
    
    Standard classical baseline for multilabel text classification.
    """
    
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """Initialize TF-IDF + Logistic Regression baseline.
        
        Args:
            ngram_range: Range of n-grams for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            max_iter: Maximum iterations for logistic regression
            random_state: Random state for reproducibility
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.vectorizer_ = None
        self.classifier_ = None
        
    def fit(self, X_train: List[str], y_train: np.ndarray) -> 'TfidfLogisticRegressionBaseline':
        """Fit TF-IDF vectorizer and logistic regression classifiers.
        
        Args:
            X_train: List of text documents
            y_train: Binary label matrix (n_samples × n_labels)
        """
        # Fit TF-IDF vectorizer
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            stop_words='english'
        )
        X_train_tfidf = self.vectorizer_.fit_transform(X_train)
        
        # Train One-vs-Rest logistic regression
        self.classifier_ = OneVsRestClassifier(
            LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='lbfgs'
            )
        )
        self.classifier_.fit(X_train_tfidf, y_train)
        
        return self
    
    def predict(self, X_test: List[str]) -> np.ndarray:
        """Predict labels for test documents.
        
        Args:
            X_test: List of text documents
            
        Returns:
            Binary predictions (n_samples × n_labels)
        """
        if self.vectorizer_ is None or self.classifier_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_tfidf = self.vectorizer_.transform(X_test)
        predictions = self.classifier_.predict(X_test_tfidf)
        return predictions
    
    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """Predict label probabilities for test documents.
        
        Args:
            X_test: List of text documents
            
        Returns:
            Label probabilities (n_samples × n_labels)
        """
        if self.vectorizer_ is None or self.classifier_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_tfidf = self.vectorizer_.transform(X_test)
        probabilities = self.classifier_.predict_proba(X_test_tfidf)
        return probabilities


def evaluate_baselines(data_dir: str, output_dir: str):
    """Evaluate baseline methods on Reuters multilabel classification.
    
    Args:
        data_dir: Directory containing Reuters train/val/test CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Baseline Evaluation: Low Data Regime")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    df_train, df_val, df_test = load_datasets(data_dir)
    
    # Fuse train and val
    df_train_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    print(f"Train: {len(df_train)} samples")
    print(f"Val: {len(df_val)} samples")
    print(f"Train+Val (fused): {len(df_train_full)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Detect label columns
    text_column = 'text'
    label_columns = detect_label_columns(df_train, text_column)
    print(f"\nDetected {len(label_columns)} labels: {label_columns}")
    
    # Prepare data
    X_train_full = df_train_full[text_column].tolist()
    y_train_full = df_train_full[label_columns].values
    
    X_test = df_test[text_column].tolist()
    y_test = df_test[label_columns].values
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # =========================================================================
    # Baseline 1: Frequency-based Random Sampling
    # =========================================================================
    print("\n" + "=" * 80)
    print("Baseline 1: Frequency-based Random Sampling")
    print("=" * 80)
    
    freq_baseline = FrequencyRandomBaseline(random_state=42)
    freq_baseline.fit(y_train_full)
    
    print("\nLabel frequencies:")
    for i, label in enumerate(label_columns):
        print(f"  {label}: {freq_baseline.label_frequencies_[i]:.4f}")
    
    y_pred_freq = freq_baseline.predict(len(y_test))
    metrics_freq = compute_metrics(y_test, y_pred_freq)
    
    print("\nTest Metrics (Frequency Random Baseline):")
    print(f"  Accuracy: {metrics_freq['accuracy']:.4f}")
    print(f"  F1 (micro): {metrics_freq['f1_micro']:.4f}")
    print(f"  F1 (macro): {metrics_freq['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics_freq['f1_weighted']:.4f}")
    print(f"  Precision (micro): {metrics_freq['precision_micro']:.4f}")
    print(f"  Precision (macro): {metrics_freq['precision_macro']:.4f}")
    print(f"  Recall (micro): {metrics_freq['recall_micro']:.4f}")
    print(f"  Recall (macro): {metrics_freq['recall_macro']:.4f}")
    print(f"  Hamming Loss: {metrics_freq['hamming_loss']:.4f}")
    
    results.append({
        'method': 'Frequency Random Sampling',
        'train_samples': len(df_train_full),
        **metrics_freq
    })
    
    # =========================================================================
    # Baseline 2: TF-IDF + Logistic Regression (One-vs-Rest)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Baseline 2: TF-IDF + Logistic Regression (One-vs-Rest)")
    print("=" * 80)
    
    tfidf_lr_baseline = TfidfLogisticRegressionBaseline(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_iter=1000,
        random_state=42
    )
    
    print("\nFitting TF-IDF + Logistic Regression...")
    tfidf_lr_baseline.fit(X_train_full, y_train_full)
    
    print(f"Vocabulary size: {len(tfidf_lr_baseline.vectorizer_.vocabulary_)}")
    
    print("\nPredicting on test set...")
    y_pred_tfidf_lr = tfidf_lr_baseline.predict(X_test)
    metrics_tfidf_lr = compute_metrics(y_test, y_pred_tfidf_lr)
    
    print("\nTest Metrics (TF-IDF + Logistic Regression):")
    print(f"  Accuracy: {metrics_tfidf_lr['accuracy']:.4f}")
    print(f"  F1 (micro): {metrics_tfidf_lr['f1_micro']:.4f}")
    print(f"  F1 (macro): {metrics_tfidf_lr['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics_tfidf_lr['f1_weighted']:.4f}")
    print(f"  Precision (micro): {metrics_tfidf_lr['precision_micro']:.4f}")
    print(f"  Precision (macro): {metrics_tfidf_lr['precision_macro']:.4f}")
    print(f"  Recall (micro): {metrics_tfidf_lr['recall_micro']:.4f}")
    print(f"  Recall (macro): {metrics_tfidf_lr['recall_macro']:.4f}")
    print(f"  Hamming Loss: {metrics_tfidf_lr['hamming_loss']:.4f}")
    
    results.append({
        'method': 'TF-IDF + Logistic Regression (OvR)',
        'train_samples': len(df_train_full),
        **metrics_tfidf_lr
    })
    
    # =========================================================================
    # Print Results Summary
    # =========================================================================
    results_df = pd.DataFrame(results)
    
    # Print full comparison table
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Full Comparison Table")
    print("=" * 80)
    print("\n" + results_df.to_string(index=False))
    
    # Print key metrics summary (for easy reference)
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY")
    print("=" * 80)
    summary_df = results_df[['method', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']]
    summary_df.columns = ['Method', 'Accuracy', 'F1-Score (macro)', 'Precision (macro)', 'Recall (macro)']
    print("\n" + summary_df.to_string(index=False))
    print("\n")
    
    return results_df


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    output_dir = os.getenv('OUTPUT_DIR', 'outputs/baselines')
    
    evaluate_baselines(data_dir=data_dir, output_dir=output_dir)
