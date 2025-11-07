"""Utility functions and helpers."""

from .logging import setup_logging, get_logger
from .metrics import ClassificationMetrics, evaluate_predictions
from .data import DataLoader, split_data, balance_data
from .results_manager import ResultsManager, ModelResultsManager, ExperimentMetadata

# Fusion utilities (optional import due to matplotlib dependency)
try:
    from .fusion_utils import (
        FusionUtils, prepare_fusion_data, validate_fusion_data, 
        load_fusion_model, evaluate_fusion_model
    )
    FUSION_UTILS_AVAILABLE = True
except ImportError:
    FUSION_UTILS_AVAILABLE = False

__all__ = [
    "setup_logging",
    "get_logger",
    "ClassificationMetrics",
    "evaluate_predictions",
    "DataLoader",
    "split_data",
    "balance_data",
    "ResultsManager",
    "ModelResultsManager",
    "ExperimentMetadata",
]

if FUSION_UTILS_AVAILABLE:
    __all__.extend([
        "FusionUtils",
        "prepare_fusion_data",
        "validate_fusion_data", 
        "load_fusion_model",
        "evaluate_fusion_model"
    ])

