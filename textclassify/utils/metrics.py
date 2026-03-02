"""Evaluation metrics for text classification."""

import numpy as np
from typing import Dict, List, Union, Optional, Any
from collections import defaultdict

from ..core.types import ClassificationResult, ClassificationType


class ClassificationMetrics:
    """Comprehensive metrics for text classification evaluation."""
    
    def __init__(self, classification_type: ClassificationType):
        """Initialize metrics calculator.
        
        Args:
            classification_type: Type of classification task
        """
        self.classification_type = classification_type
    
    def calculate_metrics(
        self,
        y_true: List[Union[str, List[str]]],
        y_pred: List[Union[str, List[str]]],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: List of class names (optional)
            
        Returns:
            Dictionary containing various metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if self.classification_type == ClassificationType.MULTI_CLASS:
            return self._calculate_multiclass_metrics(y_true, y_pred, classes)
        else:
            return self._calculate_multilabel_metrics(y_true, y_pred, classes)
    
    def _calculate_multiclass_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for multi-class classification."""
        
        # Get unique classes
        if classes is None:
            classes = sorted(list(set(y_true + y_pred)))
        
        # Calculate basic metrics
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-class metrics
        class_metrics = {}
        confusion_matrix = self._build_confusion_matrix(y_true, y_pred, classes)
        
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        
        for i, class_name in enumerate(classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(len(classes)) if j != i)
            fn = sum(confusion_matrix[i][j] for j in range(len(classes)) if j != i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': sum(1 for label in y_true if label == class_name)
            }
            
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
        
        # Macro averages
        num_classes = len(classes)
        macro_precision /= num_classes
        macro_recall /= num_classes
        macro_f1 /= num_classes
        
        # Weighted averages
        total_support = sum(class_metrics[cls]['support'] for cls in classes)
        weighted_precision = sum(
            class_metrics[cls]['precision'] * class_metrics[cls]['support']
            for cls in classes
        ) / total_support if total_support > 0 else 0.0
        
        weighted_recall = sum(
            class_metrics[cls]['recall'] * class_metrics[cls]['support']
            for cls in classes
        ) / total_support if total_support > 0 else 0.0
        
        weighted_f1 = sum(
            class_metrics[cls]['f1_score'] * class_metrics[cls]['support']
            for cls in classes
        ) / total_support if total_support > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix,
            'classes': classes,
            'total_samples': total
        }
    
    def _calculate_multilabel_metrics(
        self,
        y_true: List[List[str]],
        y_pred: List[List[str]],
        classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for multi-label classification."""
        
        # Get unique classes
        if classes is None:
            all_labels = set()
            for labels in y_true + y_pred:
                all_labels.update(labels)
            classes = sorted(list(all_labels))
        
        # Convert to binary format
        y_true_binary = self._to_binary_matrix(y_true, classes)
        y_pred_binary = self._to_binary_matrix(y_pred, classes)
        
        # Calculate sample-wise metrics
        exact_match = sum(
            1 for true_labels, pred_labels in zip(y_true, y_pred)
            if set(true_labels) == set(pred_labels)
        ) / len(y_true)
        
        # Hamming loss (fraction of wrong labels)
        hamming_loss = np.mean(y_true_binary != y_pred_binary)
        
        # Per-class metrics
        class_metrics = {}
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        
        for i, class_name in enumerate(classes):
            tp = np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 1))
            fp = np.sum((y_true_binary[:, i] == 0) & (y_pred_binary[:, i] == 1))
            fn = np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(y_true_binary[:, i])
            }
            
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
        
        # Macro averages
        num_classes = len(classes)
        macro_precision /= num_classes
        macro_recall /= num_classes
        macro_f1 /= num_classes
        
        # Micro averages (aggregate across all classes)
        total_tp = sum(
            np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 1))
            for i in range(len(classes))
        )
        total_fp = sum(
            np.sum((y_true_binary[:, i] == 0) & (y_pred_binary[:, i] == 1))
            for i in range(len(classes))
        )
        total_fn = sum(
            np.sum((y_true_binary[:, i] == 1) & (y_pred_binary[:, i] == 0))
            for i in range(len(classes))
        )
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        return {
            'exact_match_ratio': exact_match,
            'hamming_loss': hamming_loss,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'per_class_metrics': class_metrics,
            'classes': classes,
            'total_samples': len(y_true)
        }
    
    def _build_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        classes: List[str]
    ) -> List[List[int]]:
        """Build confusion matrix for multi-class classification."""
        
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        matrix = [[0] * len(classes) for _ in range(len(classes))]
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in class_to_idx and pred_label in class_to_idx:
                true_idx = class_to_idx[true_label]
                pred_idx = class_to_idx[pred_label]
                matrix[true_idx][pred_idx] += 1
        
        return matrix
    
    def _to_binary_matrix(
        self,
        labels: List[List[str]],
        classes: List[str]
    ) -> np.ndarray:
        """Convert multi-label format to binary matrix."""
        
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        matrix = np.zeros((len(labels), len(classes)), dtype=int)
        
        for i, label_list in enumerate(labels):
            for label in label_list:
                if label in class_to_idx:
                    matrix[i, class_to_idx[label]] = 1
        
        return matrix


def evaluate_predictions(
    result: ClassificationResult,
    y_true: List[Union[str, List[str]]],
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluate classification predictions.
    
    Args:
        result: Classification result from a model
        y_true: True labels
        classes: List of class names (optional)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if result.classification_type is None:
        raise ValueError("Classification type must be specified in result")
    
    metrics_calculator = ClassificationMetrics(result.classification_type)
    metrics = metrics_calculator.calculate_metrics(y_true, result.predictions, classes)
    
    # Add model information
    metrics['model_info'] = {
        'model_name': result.model_name,
        'model_type': result.model_type.value if result.model_type else None,
        'processing_time': result.processing_time
    }
    
    return metrics


def compare_models(
    results: List[ClassificationResult],
    y_true: List[Union[str, List[str]]],
    model_names: Optional[List[str]] = None,
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare multiple model results.
    
    Args:
        results: List of classification results
        y_true: True labels
        model_names: Optional list of model names
        classes: List of class names (optional)
        
    Returns:
        Dictionary containing comparison metrics
    """
    if not results:
        raise ValueError("At least one result is required")
    
    if model_names and len(model_names) != len(results):
        raise ValueError("Number of model names must match number of results")
    
    comparison = {
        'models': {},
        'summary': {}
    }
    
    # Evaluate each model
    for i, result in enumerate(results):
        model_name = model_names[i] if model_names else f"model_{i+1}"
        metrics = evaluate_predictions(result, y_true, classes)
        comparison['models'][model_name] = metrics
    
    # Create summary comparison
    if results[0].classification_type == ClassificationType.MULTI_CLASS:
        key_metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    else:
        key_metrics = ['exact_match_ratio', 'macro_f1', 'micro_f1']
    
    for metric in key_metrics:
        comparison['summary'][metric] = {
            model_name: comparison['models'][model_name][metric]
            for model_name in comparison['models']
        }
    
    # Find best model for each metric
    comparison['best_models'] = {}
    for metric in key_metrics:
        best_model = max(
            comparison['summary'][metric],
            key=comparison['summary'][metric].get
        )
        comparison['best_models'][metric] = best_model
    
    return comparison

