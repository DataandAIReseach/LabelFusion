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

