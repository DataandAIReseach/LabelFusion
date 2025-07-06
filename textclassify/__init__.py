"""
TextClassify: A comprehensive text classification package supporting LLMs and traditional ML models.

This package provides multi-class and multi-label text classification capabilities using:
- LLM providers: OpenAI, Claude, Gemini, DeepSeek
- Traditional ML models: RoBERTa (optional)
- Ensemble methods for optimized performance
"""

__version__ = "0.1.0"
__author__ = "TextClassify Team"
__email__ = "contact@textclassify.com"

# Core imports (always available)
from .core.base import BaseClassifier
from .core.types import (
    ClassificationResult, ClassificationType, ModelType, LLMProvider,
    TrainingData, ModelConfig, EnsembleConfig
)
from .core.exceptions import TextClassifyError, ModelNotFoundError, ConfigurationError, APIError, EnsembleError

# LLM Classifiers (require API keys but no additional dependencies)
from .llm.openai_classifier import OpenAIClassifier
from .llm.claude_classifier import ClaudeClassifier
from .llm.gemini_classifier import GeminiClassifier
from .llm.deepseek_classifier import DeepSeekClassifier

# Traditional ML Classifiers (optional - require transformers/torch)
try:
    from .ml.roberta_classifier import RoBERTaClassifier
    _HAS_ML = True
except ImportError:
    _HAS_ML = False
    RoBERTaClassifier = None

# Ensemble Methods
from .ensemble.voting import VotingEnsemble
from .ensemble.weighted import WeightedEnsemble
from .ensemble.routing import ClassRoutingEnsemble

# Configuration
from .config.settings import Config
from .config.api_keys import APIKeyManager

# Base exports
__all__ = [
    # Core
    "BaseClassifier",
    "ClassificationResult",
    "ClassificationType",
    "ModelType", 
    "LLMProvider",
    "TrainingData",
    "ModelConfig",
    "EnsembleConfig",
    "TextClassifyError",
    "ModelNotFoundError",
    "ConfigurationError",
    "APIError",
    "EnsembleError",
    
    # LLM Classifiers
    "OpenAIClassifier",
    "ClaudeClassifier", 
    "GeminiClassifier",
    "DeepSeekClassifier",
    
    # Ensemble Methods
    "VotingEnsemble",
    "WeightedEnsemble",
    "ClassRoutingEnsemble",
    
    # Configuration
    "Config",
    "APIKeyManager",
    
    # Version
    "__version__",
]

# Add ML classifiers to exports if available
if _HAS_ML:
    __all__.append("RoBERTaClassifier")


def get_available_features():
    """Get information about available features based on installed dependencies.
    
    Returns:
        dict: Dictionary with feature availability information
    """
    features = {
        "llm_classifiers": True,
        "ensemble_methods": True,
        "configuration": True,
        "traditional_ml": _HAS_ML,
    }
    
    # Check for optional dependencies
    try:
        import yaml
        features["yaml_config"] = True
    except ImportError:
        features["yaml_config"] = False
    
    return features


def check_dependencies():
    """Check and report on package dependencies.
    
    Prints information about available and missing dependencies.
    """
    print("TextClassify Dependency Check")
    print("=" * 30)
    
    # Core dependencies
    core_deps = ["aiohttp", "requests", "numpy", "pandas"]
    print("\nCore Dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (missing)")
    
    # Optional dependencies
    optional_deps = {
        "torch": "Required for RoBERTa classifier",
        "transformers": "Required for RoBERTa classifier", 
        "scikit-learn": "Required for ML utilities",
        "yaml": "Required for YAML configuration files"
    }
    
    print("\nOptional Dependencies:")
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"  ✓ {dep} - {description}")
        except ImportError:
            print(f"  ✗ {dep} - {description}")
    
    # Feature availability
    features = get_available_features()
    print("\nAvailable Features:")
    for feature, available in features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    
    if not features["traditional_ml"]:
        print("\nTo enable traditional ML features, install:")
        print("  pip install transformers torch scikit-learn")
    
    if not features["yaml_config"]:
        print("\nTo enable YAML configuration support, install:")
        print("  pip install pyyaml")

