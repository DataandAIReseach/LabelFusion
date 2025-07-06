"""Core module containing base classes, types, and exceptions."""

from .base import BaseClassifier
from .types import ClassificationResult, ClassificationType, ModelType
from .exceptions import TextClassifyError, ModelNotFoundError, ConfigurationError

__all__ = [
    "BaseClassifier",
    "ClassificationResult", 
    "ClassificationType",
    "ModelType",
    "TextClassifyError",
    "ModelNotFoundError",
    "ConfigurationError",
]

