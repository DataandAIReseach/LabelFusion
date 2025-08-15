"""Base class for traditional machine learning classifiers."""

import pickle
import os
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType, TrainingData
from ..core.exceptions import ModelTrainingError, PredictionError, ValidationError


class BaseMLClassifier(BaseClassifier):
    """Base class for traditional machine learning text classifiers."""
    
    def __init__(self, config):
        """Initialize the ML classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.config.model_type = ModelType.TRADITIONAL_ML
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.model_path = None
        
    def save_model(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Raises:
            ModelTrainingError: If model is not trained or save fails
        """
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before saving", self.config.model_name)
        
        try:
            model_data = {
                'model': self.model,
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'classes_': self.classes_,
                'classification_type': self.classification_type,
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.model_path = path
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}", self.config.model_name)
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Raises:
            ModelTrainingError: If model loading fails
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.tokenizer = model_data['tokenizer']
            self.label_encoder = model_data['label_encoder']
            self.classes_ = model_data['classes_']
            self.classification_type = model_data['classification_type']
            self.is_trained = model_data['is_trained']
            self.model_path = path
            
            # Update config if needed
            if 'config' in model_data:
                saved_config = model_data['config']
                # Merge saved config with current config, prioritizing current config
                for key, value in saved_config.parameters.items():
                    if key not in self.config.parameters:
                        self.config.parameters[key] = value
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}", self.config.model_name)
    
    def _prepare_labels(self, labels: Union[List[str], List[List[str]]], classification_type: ClassificationType):
        """Prepare labels for training.
        
        Args:
            labels: Raw labels
            classification_type: Type of classification
            
        Returns:
            Processed labels suitable for training
        """
        if classification_type == ClassificationType.MULTI_CLASS:
            # For multi-class, labels are already in the right format
            return labels
        else:
            # For multi-label, we need to handle the format differently
            # This will be implemented by specific classifiers
            return labels
    
    def _validate_training_data(self, training_data: TrainingData) -> None:
        """Validate training data.
        
        Args:
            training_data: Training data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not training_data.texts:
            raise ValidationError("Training texts cannot be empty")
        
        if not training_data.labels:
            raise ValidationError("Training labels cannot be empty")
        
        if len(training_data.texts) != len(training_data.labels):
            raise ValidationError("Number of texts and labels must match")
        
        # Check for empty texts
        if any(not text.strip() for text in training_data.texts):
            raise ValidationError("Training texts cannot be empty or whitespace only")
        
        # Validate label format based on classification type
        if training_data.classification_type == ClassificationType.MULTI_CLASS:
            # Multi-class now uses binary/one-hot encoding format
            if not all(isinstance(label, list) for label in training_data.labels):
                raise ValidationError("Multi-class labels must be lists of integers (one-hot encoded)")
            
            # Check that each label has exactly one 1 (one-hot encoding)
            for i, label in enumerate(training_data.labels):
                if not all(isinstance(x, int) and x in [0, 1] for x in label):
                    raise ValidationError(f"Multi-class labels must contain only 0s and 1s. Error at index {i}")
                if sum(label) != 1:
                    raise ValidationError(f"Multi-class labels must have exactly one 1 (one-hot encoding). Error at index {i}")
        else:
            # Multi-label uses binary encoding format
            if not all(isinstance(label, list) for label in training_data.labels):
                raise ValidationError("Multi-label labels must be lists of integers (binary encoded)")
            
            # Check that each label contains only 0s and 1s
            for i, label in enumerate(training_data.labels):
                if not all(isinstance(x, int) and x in [0, 1] for x in label):
                    raise ValidationError(f"Multi-label labels must contain only 0s and 1s. Error at index {i}")
            
            # Allow empty label lists for multi-label (all zeros)
            # This is valid in multi-label classification
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get ML model information."""
        info = super().model_info
        info.update({
            "model_path": self.model_path,
            "has_tokenizer": self.tokenizer is not None,
            "has_label_encoder": self.label_encoder is not None
        })
        return info

    def _create_result(
        self,
        predictions: List[List[int]],
        probabilities: Optional[List[Dict[str, float]]] = None,
        confidence_scores: Optional[List[float]] = None,
        true_labels: Optional[List[List[int]]] = None
    ) -> ClassificationResult:
        """Create ClassificationResult with metrics calculation if true labels provided."""
        
        # Calculate metrics if true labels are provided
        metrics = None
        if true_labels is not None:
            metrics = self._calculate_metrics(predictions, true_labels)
        
        # Create base result using inherited method from core base class
        result = super()._create_result(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            true_labels=true_labels
        )
        
        # Add metrics to metadata if calculated
        if metrics:
            if result.metadata is None:
                result.metadata = {}
            result.metadata['metrics'] = metrics
        
        return result

    def _calculate_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for binary vector predictions."""
        if self.classification_type == ClassificationType.MULTI_LABEL:
            return self._calculate_multi_label_metrics(predictions, true_labels)
        return self._calculate_single_label_metrics(predictions, true_labels)

    def _calculate_single_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for single-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
        # Convert binary vectors to class indices for sklearn compatibility
        pred_classes = [pred.index(1) if 1 in pred else 0 for pred in predictions]
        true_classes = [true.index(1) if 1 in true else 0 for true in true_labels]
        
        # Calculate basic accuracy
        correct = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == true)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # For single-label classification with more than 2 classes, use macro averaging
        num_classes = len(self.classes_) if self.classes_ else max(max(pred_classes, default=0), max(true_classes, default=0)) + 1
        
        if num_classes <= 2:
            # Binary classification metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                precision = precision_score(true_classes, pred_classes, average='binary', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='binary', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='binary', zero_division=0)
                
                # For AUC, we need probability scores, but we only have binary predictions
                # Use the prediction confidence as a proxy (1.0 for predicted class, 0.0 for others)
                try:
                    auc = roc_auc_score(true_classes, pred_classes)
                except ValueError:
                    # If all predictions are the same class, AUC is undefined
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        else:
            # Multi-class classification metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                precision = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
                
                # For multi-class AUC, convert to one-hot and use ovr strategy
                try:
                    from sklearn.preprocessing import label_binarize
                    true_binary = label_binarize(true_classes, classes=list(range(num_classes)))
                    pred_binary = label_binarize(pred_classes, classes=list(range(num_classes)))
                    auc = roc_auc_score(true_binary, pred_binary, average='macro', multi_class='ovr')
                except (ValueError, ImportError):
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _calculate_metrics_manual(self, pred_classes: List[int], true_classes: List[int], num_classes: int) -> tuple:
        """Manual calculation of metrics when sklearn is not available."""
        # Calculate per-class metrics
        class_metrics = []
        
        for class_idx in range(num_classes):
            # True positives, false positives, false negatives for this class
            tp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true == class_idx)
            fp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true != class_idx)
            fn = sum(1 for pred, true in zip(pred_classes, true_classes) if pred != class_idx and true == class_idx)
            
            # Calculate precision, recall, f1 for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics.append((precision, recall, f1))
        
        # Macro average
        if class_metrics:
            avg_precision = sum(m[0] for m in class_metrics) / len(class_metrics)
            avg_recall = sum(m[1] for m in class_metrics) / len(class_metrics)
            avg_f1 = sum(m[2] for m in class_metrics) / len(class_metrics)
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
        
        # Simple AUC approximation (not perfect but better than nothing)
        auc = 0.5  # Default for when we can't calculate properly
        
        return avg_precision, avg_recall, avg_f1, auc

    def _calculate_multi_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for multi-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}
        
        # Sample-wise metrics
        sample_precisions = []
        sample_recalls = []
        sample_f1s = []
        exact_matches = 0
        hamming_distance = 0
        total_predictions = 0
        
        for pred, true in zip(predictions, true_labels):
            pred_set = set(i for i, val in enumerate(pred) if val == 1)
            true_set = set(i for i, val in enumerate(true) if val == 1)
            
            # Sample-wise precision, recall, F1
            if pred_set:
                precision = len(pred_set & true_set) / len(pred_set)
            else:
                precision = 1.0 if not true_set else 0.0
            
            if true_set:
                recall = len(pred_set & true_set) / len(true_set)
            else:
                recall = 1.0 if not pred_set else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            sample_precisions.append(precision)
            sample_recalls.append(recall)
            sample_f1s.append(f1)
            
            # Exact match (subset accuracy)
            if pred_set == true_set:
                exact_matches += 1
            
            # Hamming loss components
            for i in range(len(pred)):
                total_predictions += 1
                if pred[i] != true[i]:
                    hamming_distance += 1
        
        # Calculate averages
        avg_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
        avg_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
        avg_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
        
        # Subset accuracy (exact match ratio)
        subset_accuracy = exact_matches / len(predictions) if predictions else 0.0
        
        # Hamming loss
        hamming_loss = hamming_distance / total_predictions if total_predictions > 0 else 0.0
        
        # Label-wise metrics (micro-averaged)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Flatten for micro-averaging
            y_true_flat = [label for true in true_labels for label in true]
            y_pred_flat = [label for pred in predictions for label in pred]
            
            micro_precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            
        except ImportError:
            # Fallback when sklearn is not available
            micro_precision = avg_precision
            micro_recall = avg_recall
            micro_f1 = avg_f1
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming_loss,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1
        }

