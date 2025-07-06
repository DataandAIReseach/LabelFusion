"""Base classifier interface and abstract classes."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .types import ClassificationResult, ClassificationType, ModelConfig, ModelType, TrainingData
from .exceptions import ValidationError, PredictionError


class BaseClassifier(ABC):
    """Abstract base class for all text classifiers."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the classifier with configuration.
        
        Args:
            config: Model configuration containing parameters and settings
        """
        self.config = config
        self.is_trained = False
        self.classes_ = None
        self.classification_type = None
        
    @abstractmethod
    def fit(self, training_data: TrainingData) -> None:
        """Train the classifier on the provided data.
        
        Args:
            training_data: Training data containing texts and labels
            
        Raises:
            ModelTrainingError: If training fails
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> ClassificationResult:
        """Predict labels for the given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult containing predictions and metadata
            
        Raises:
            PredictionError: If prediction fails
            ValidationError: If input validation fails
        """
        pass
    
    @abstractmethod
    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Predict class probabilities for the given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult containing predictions, probabilities and metadata
            
        Raises:
            PredictionError: If prediction fails
            ValidationError: If input validation fails
        """
        pass
    
    def validate_input(self, texts: List[str]) -> None:
        """Validate input texts.
        
        Args:
            texts: List of texts to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not texts:
            raise ValidationError("Input texts cannot be empty")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValidationError("All inputs must be strings")
        
        if not all(text.strip() for text in texts):
            raise ValidationError("Input texts cannot be empty or whitespace only")
    
    def _create_result(
        self,
        predictions: List[Union[str, List[str]]],
        probabilities: Optional[List[Dict[str, float]]] = None,
        confidence_scores: Optional[List[float]] = None,
        processing_time: Optional[float] = None,
        **metadata
    ) -> ClassificationResult:
        """Create a ClassificationResult with standard metadata.
        
        Args:
            predictions: Predicted labels
            probabilities: Class probabilities (optional)
            confidence_scores: Confidence scores (optional)
            processing_time: Time taken for processing (optional)
            **metadata: Additional metadata
            
        Returns:
            ClassificationResult with populated metadata
        """
        return ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            model_name=self.config.model_name,
            model_type=self.config.model_type,
            classification_type=self.classification_type,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _time_operation(self, operation_func, *args, **kwargs):
        """Time an operation and return result with timing.
        
        Args:
            operation_func: Function to time
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (result, processing_time)
        """
        start_time = time.time()
        try:
            result = operation_func(*args, **kwargs)
            processing_time = time.time() - start_time
            return result, processing_time
        except Exception as e:
            processing_time = time.time() - start_time
            raise PredictionError(
                f"Operation failed after {processing_time:.2f}s: {str(e)}",
                model_name=self.config.model_name
            )
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type.value,
            "is_trained": self.is_trained,
            "classification_type": self.classification_type.value if self.classification_type else None,
            "classes": self.classes_,
            "config": self.config.parameters
        }


class AsyncBaseClassifier(BaseClassifier):
    """Base class for asynchronous classifiers (primarily for LLM-based models)."""
    
    @abstractmethod
    async def predict_async(self, texts: List[str]) -> ClassificationResult:
        """Asynchronously predict labels for the given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult containing predictions and metadata
            
        Raises:
            PredictionError: If prediction fails
            ValidationError: If input validation fails
        """
        pass
    
    @abstractmethod
    async def predict_proba_async(self, texts: List[str]) -> ClassificationResult:
        """Asynchronously predict class probabilities for the given texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult containing predictions, probabilities and metadata
            
        Raises:
            PredictionError: If prediction fails
            ValidationError: If input validation fails
        """
        pass

