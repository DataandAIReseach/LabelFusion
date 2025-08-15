"""Core types and data structures for text classification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class ClassificationType(Enum):
    """Type of classification task."""
    SINGLE_LABEL = "single_label"  # Single label per text (mutually exclusive)
    MULTI_CLASS = "multi_class"  # Single label per text (mutually exclusive) - alias for SINGLE_LABEL
    MULTI_LABEL = "multi_label"  # Multiple labels per text (non-exclusive)


class ModelType(Enum):
    """Type of model used for classification."""
    LLM = "llm"
    TRADITIONAL_ML = "traditional_ml"
    ENSEMBLE = "ensemble"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"


@dataclass
class ClassificationResult:
    """Result of a classification operation."""
    
    # Core prediction results
    predictions: List[Union[str, List[str]]]  # Single label or list of labels per text
    probabilities: Optional[List[Dict[str, float]]] = None  # Class probabilities per text
    confidence_scores: Optional[List[float]] = None  # Overall confidence per text
    
    # Metadata
    model_name: Optional[str] = None
    model_type: Optional[ModelType] = None
    classification_type: Optional[ClassificationType] = None
    processing_time: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the result after initialization."""
        if not self.predictions:
            raise ValueError("Predictions cannot be empty")
        
        # Validate probabilities if provided
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.predictions):
                raise ValueError("Probabilities length must match predictions length")
        
        # Validate confidence scores if provided
        if self.confidence_scores is not None:
            if len(self.confidence_scores) != len(self.predictions):
                raise ValueError("Confidence scores length must match predictions length")


@dataclass
class TrainingData:
    """Training data for classification models."""
    
    texts: List[str]
    labels: Union[List[str], List[List[str]]]  # Single or multiple labels per text
    classification_type: ClassificationType
    
    # Optional metadata
    text_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate training data after initialization."""
        if len(self.texts) != len(self.labels):
            raise ValueError("Texts and labels must have the same length")
        
        if not self.texts:
            raise ValueError("Training data cannot be empty")
        
        # Validate label format based on classification type
        if self.classification_type == ClassificationType.MULTI_CLASS:
            # Check if labels are string format or binary encoded format
            first_label = self.labels[0]
            
            if isinstance(first_label, str):
                # String format: all labels should be strings
                if not all(isinstance(label, str) for label in self.labels):
                    raise ValueError("Multi-class labels must all be strings when using string format")
            elif isinstance(first_label, (list, tuple)) or (NUMPY_AVAILABLE and isinstance(first_label, np.ndarray)):
                # Binary encoded format: validate one-hot encoding
                self._validate_binary_encoded_labels(is_multi_class=True)
            else:
                raise ValueError("Multi-class labels must be strings or binary encoded vectors")
                
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            # Check if labels are list of strings format or binary encoded format
            first_label = self.labels[0]
            
            if isinstance(first_label, list) and all(isinstance(item, str) for item in first_label):
                # List of strings format: all labels should be lists of strings
                if not all(isinstance(label, list) and all(isinstance(item, str) for item in label) for label in self.labels):
                    raise ValueError("Multi-label labels must all be lists of strings when using string format")
            elif isinstance(first_label, (list, tuple)) or (NUMPY_AVAILABLE and isinstance(first_label, np.ndarray)):
                # Binary encoded format: validate multi-hot encoding
                self._validate_binary_encoded_labels(is_multi_class=False)
            else:
                raise ValueError("Multi-label labels must be lists of strings or binary encoded vectors")
    
    def _validate_binary_encoded_labels(self, is_multi_class: bool):
        """Validate binary encoded labels for consistency."""
        if not self.labels:
            return
            
        # Get the expected length from the first label
        first_label = self.labels[0]
        if NUMPY_AVAILABLE and isinstance(first_label, np.ndarray):
            expected_length = len(first_label)
        else:
            expected_length = len(first_label)
        
        for i, label in enumerate(self.labels):
            # Check if it's a valid vector type
            if not (isinstance(label, (list, tuple)) or (NUMPY_AVAILABLE and isinstance(label, np.ndarray))):
                raise ValueError(f"Label at index {i} must be a list, tuple, or numpy array for binary encoding")
            
            # Check vector length consistency
            if len(label) != expected_length:
                raise ValueError(f"Label at index {i} has length {len(label)}, expected {expected_length}")
            
            # Check if all values are integers (0 or 1)
            for j, value in enumerate(label):
                if not isinstance(value, (int, float)) or value not in [0, 1]:
                    raise ValueError(f"Label at index {i}, position {j} must be 0 or 1, got {value}")
            
            # Multi-class specific validation: exactly one value should be 1
            if is_multi_class:
                ones_count = sum(1 for x in label if x == 1)
                if ones_count != 1:
                    raise ValueError(f"Multi-class label at index {i} must have exactly one value equal to 1, found {ones_count}")
            
            # Multi-label specific validation: at least one value should be 1 (optional, can be all zeros)
            # No additional validation needed for multi-label as multiple 1s are allowed


@dataclass
class ModelConfig:
    """Configuration for a classification model."""
    
    model_name: str
    model_type: ModelType
    
    # Model-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # API configuration (for LLM models)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Training configuration (for ML models)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Caching and performance
    enable_caching: bool = True
    batch_size: int = 32
    max_retries: int = 3
    timeout: float = 30.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    ensemble_method: str  # "voting", "weighted", "routing", "fusion"
    models: Optional[List] = None  # List of model instances or configs
    
    # General parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Method-specific parameters  
    weights: Optional[List[float]] = None  # For weighted ensemble
    routing_rules: Optional[Dict[str, str]] = None  # For class routing
    
    # Ensemble behavior
    require_all_models: bool = False  # Whether all models must succeed
    fallback_model: Optional[str] = None  # Fallback if primary models fail

