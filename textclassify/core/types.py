"""Core types and data structures for text classification."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ClassificationType(Enum):
    """Type of classification task."""
    MULTI_CLASS = "multi_class"  # Single label per text (mutually exclusive)
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
            if not all(isinstance(label, str) for label in self.labels):
                raise ValueError("Multi-class labels must be strings")
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            if not all(isinstance(label, list) for label in self.labels):
                raise ValueError("Multi-label labels must be lists of strings")


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
    
    models: List[ModelConfig]
    ensemble_method: str  # "voting", "weighted", "routing"
    
    # Method-specific parameters
    weights: Optional[List[float]] = None  # For weighted ensemble
    routing_rules: Optional[Dict[str, str]] = None  # For class routing
    
    # Ensemble behavior
    require_all_models: bool = False  # Whether all models must succeed
    fallback_model: Optional[str] = None  # Fallback if primary models fail

