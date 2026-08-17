"""Zero-shot LLM evaluation on Reuters dataset.

Evaluates gpt-5-nano in zero-shot mode (no training examples in prompt) on
validation and test sets for multilabel text classification.

Environment variables:
    - DATA_DIR: path to Reuters data (default /scratch/users/u19147/LabelFusion/data/reuters)
    - OUTPUT_DIR: path to write outputs (default outputs/llm_zero_shot)
    - CACHE_DIR: path to cache LLM predictions (default cache)
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ensure project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig, ModelType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)


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


def compute_hash(df: pd.DataFrame, text_column: str = 'text') -> str:
    """Compute hash of dataset for caching."""
    text_concat = ''.join(df[text_column].astype(str).tolist())
    return hashlib.md5(text_concat.encode()).hexdigest()[:8]


def save_predictions(predictions_df: pd.DataFrame, output_dir: str, 
                     dataset_name: str, dataset_hash: str, mode: str = 'zero_shot'):
    """Save predictions to JSON and CSV files.
    
    Args:
        predictions_df: DataFrame with predictions
        output_dir: Directory to save files
        dataset_name: 'val' or 'test'
        dataset_hash: Hash identifier for the dataset
        mode: 'zero_shot' or 'one_shot'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    date = datetime.now().strftime('%Y%m%d')
    base_name = f"{dataset_name}_{date}_{dataset_hash}_{mode}"
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    predictions_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    predictions_dict = predictions_df.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump(predictions_dict, f, indent=2)
    print(f"  Saved JSON: {json_path}")


def create_llm_model(text_column: str, label_columns: List[str], 
                     output_dir: str, experiment_name: str, 
                     cache_dir: str = 'cache'):
    """Create zero-shot LLM model (gpt-5-nano).
    
    Args:
        text_column: Name of text column
        label_columns: List of label column names
        output_dir: Output directory
        experiment_name: Experiment name
        cache_dir: Cache directory for LLM predictions
    """
    llm_config = ModelConfig(
        model_name='gpt-5-nano',
        model_type=ModelType.LLM,
        parameters={
            'model': 'gpt-5-nano',
            'temperature': 0.1,
            'max_completion_tokens': 150,
            'top_p': 1.0
        }
    )
    
    return OpenAIClassifier(
        config=llm_config,
        text_column=text_column,
        label_columns=label_columns,
        multi_label=True,
        few_shot_mode=0,  # Zero-shot: no training examples in prompt
        auto_use_cache=True,
        cache_dir=cache_dir,
        auto_save_results=True,
        output_dir=output_dir,
        experiment_name=f"{experiment_name}_llm_zero_shot"
    )


def evaluate_llm_zeroshot(data_dir: str, output_dir: str, cache_dir: str = 'cache', 
                          eval_mode: str = 'both'):
    """Evaluate LLM in zero-shot mode on validation and test sets.
    
    Args:
        data_dir: Directory containing Reuters train/val/test CSV files
        output_dir: Directory to save results
        cache_dir: Directory to cache LLM predictions
        eval_mode: Which dataset to evaluate - 'val', 'test', or 'both' (default: 'both')
    """
    print("=" * 80)
    print("Zero-Shot LLM Evaluation (gpt-5-nano)")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    df_train, df_val, df_test = load_datasets(data_dir)
    
    print(f"Train: {len(df_train)} samples (not used for zero-shot)")
    print(f"Val: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Detect label columns
    text_column = 'text'
    label_columns = detect_label_columns(df_train, text_column)
    print(f"\nDetected {len(label_columns)} labels: {label_columns}")
    
    # Compute dataset hashes
    val_hash = compute_hash(df_val, text_column)
    test_hash = compute_hash(df_test, text_column)
    print(f"\nValidation set hash: {val_hash}")
    print(f"Test set hash: {test_hash}")
    
    # Create LLM model
    os.makedirs(output_dir, exist_ok=True)
    experiment_name = f"reuters_llm_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 80)
    print("Creating Zero-Shot LLM Model")
    print("=" * 80)
    print(f"Model: gpt-5-nano")
    print(f"Mode: Zero-shot (few_shot_mode=0)")
    print(f"Cache directory: {cache_dir}")
    
    llm_model = create_llm_model(
        text_column=text_column,
        label_columns=label_columns,
        output_dir=output_dir,
        experiment_name=experiment_name,
        cache_dir=cache_dir
    )
    
    results = []
    
    # =========================================================================
    # Validation Set Predictions
    # =========================================================================
    if eval_mode in ['val', 'both']:
        print("\n" + "=" * 80)
        print("VALIDATION SET - Zero-Shot Prediction")
        print("=" * 80)
        
        print(f"\nPredicting on {len(df_val)} validation samples...")
        llm_model.set_mode('val')
        val_result = llm_model.predict(test_df=df_val)
        
        # Convert predictions from list of label names to binary matrix
        y_val_true = df_val[label_columns].values
        y_val_pred = np.zeros((len(val_result.predictions), len(label_columns)), dtype=int)
        for i, pred_labels in enumerate(val_result.predictions):
            if isinstance(pred_labels, list):  # Multi-label
                for label in pred_labels:
                    if label in label_columns:
                        j = label_columns.index(label)
                        y_val_pred[i, j] = 1
            else:  # Single label
                if pred_labels in label_columns:
                    j = label_columns.index(pred_labels)
                    y_val_pred[i, j] = 1
        
        # Compute metrics
        val_metrics = compute_metrics(y_val_true, y_val_pred)
        
        # Create predictions DataFrame for saving
        val_predictions_df = df_val[[text_column]].copy()
        for i, label in enumerate(label_columns):
            val_predictions_df[label] = y_val_pred[:, i]
        
        print("\nValidation Metrics (Zero-Shot LLM):")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  F1 (micro): {val_metrics['f1_micro']:.4f}")
        print(f"  F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"  Precision (micro): {val_metrics['precision_micro']:.4f}")
        print(f"  Precision (macro): {val_metrics['precision_macro']:.4f}")
        print(f"  Recall (micro): {val_metrics['recall_micro']:.4f}")
        print(f"  Recall (macro): {val_metrics['recall_macro']:.4f}")
        print(f"  Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        
        # Save validation predictions
        print("\nSaving validation predictions...")
        save_predictions(val_predictions_df, output_dir, 'val', val_hash, 'zero_shot')
        
        results.append({
            'dataset': 'validation',
            'samples': len(df_val),
            **val_metrics
        })
    
    # =========================================================================
    # Test Set Predictions
    # =========================================================================
    if eval_mode in ['test', 'both']:
        print("\n" + "=" * 80)
        print("TEST SET - Zero-Shot Prediction")
        print("=" * 80)
        
        print(f"\nPredicting on {len(df_test)} test samples...")
        llm_model.set_mode('test')
        test_result = llm_model.predict(test_df=df_test)
        
        # Convert predictions from list of label names to binary matrix
        y_test_true = df_test[label_columns].values
        y_test_pred = np.zeros((len(test_result.predictions), len(label_columns)), dtype=int)
        for i, pred_labels in enumerate(test_result.predictions):
            if isinstance(pred_labels, list):  # Multi-label
                for label in pred_labels:
                    if label in label_columns:
                        j = label_columns.index(label)
                        y_test_pred[i, j] = 1
            else:  # Single label
                if pred_labels in label_columns:
                    j = label_columns.index(pred_labels)
                    y_test_pred[i, j] = 1
        
        # Compute metrics
        test_metrics = compute_metrics(y_test_true, y_test_pred)
        
        # Create predictions DataFrame for saving
        test_predictions_df = df_test[[text_column]].copy()
        for i, label in enumerate(label_columns):
            test_predictions_df[label] = y_test_pred[:, i]
        
        print("\nTest Metrics (Zero-Shot LLM):")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 (micro): {test_metrics['f1_micro']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  Precision (micro): {test_metrics['precision_micro']:.4f}")
        print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
        print(f"  Recall (micro): {test_metrics['recall_micro']:.4f}")
        print(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
        print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
        
        # Save test predictions
        print("\nSaving test predictions...")
        save_predictions(test_predictions_df, output_dir, 'test', test_hash, 'zero_shot')
        
        results.append({
            'dataset': 'test',
            'samples': len(df_test),
            **test_metrics
        })
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Zero-Shot LLM Evaluation")
    print("=" * 80)
    print("\n" + results_df.to_string(index=False))
    
    # Print key metrics summary
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY")
    print("=" * 80)
    summary_df = results_df[['dataset', 'samples', 'accuracy', 'f1_macro', 
                             'precision_macro', 'recall_macro']]
    summary_df.columns = ['Dataset', 'Samples', 'Accuracy', 'F1-Score (macro)', 
                          'Precision (macro)', 'Recall (macro)']
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Prediction files saved (with date):")
    print("=" * 80)
    print(f"  val_<date>_{val_hash}_zero_shot.csv")
    print(f"  val_<date>_{val_hash}_zero_shot.json")
    print(f"  test_<date>_{test_hash}_zero_shot.csv")
    print(f"  test_<date>_{test_hash}_zero_shot.json")
    print(f"\nLocation: {output_dir}/")
    print("\n")
    
    return results_df


if __name__ == '__main__':
    data_dir = os.getenv('DATA_DIR', '/scratch/users/u19147/LabelFusion/data/reuters')
    output_dir = os.getenv('OUTPUT_DIR', 'outputs/llm_zero_shot')
    cache_dir = os.getenv('CACHE_DIR', 'cache')
    eval_mode = os.getenv('EVAL_MODE', 'both')  # 'val', 'test', or 'both'
    
    evaluate_llm_zeroshot(data_dir=data_dir, output_dir=output_dir, 
                         cache_dir=cache_dir, eval_mode=eval_mode)
