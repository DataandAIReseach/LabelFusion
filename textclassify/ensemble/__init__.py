"""Ensemble methods for combining multiple classifiers.

FusionEnsemble now supports automatic cache management:

    from textclassify.ensemble import FusionEnsemble
    from textclassify.config.settings import Config
    
    # Enable auto-cache for LLM predictions
    fusion = FusionEnsemble(
        config,
        auto_use_cache=True,  # ✨ Automatically reuse cached LLM predictions
        cache_dir="cache"
    )
    
    # LLM predictions will be automatically cached and reused!
    fusion.fit(train_df, val_df)

See docs/AUTO_CACHE_FEATURE.md for details.
"""

from .base import BaseEnsemble
from .voting import VotingEnsemble
from .weighted import WeightedEnsemble
from .routing import ClassRoutingEnsemble
from .fusion import FusionEnsemble, FusionMLP, FusionWrapper
from .auto_fusion import AutoFusionClassifier

__all__ = [
    "BaseEnsemble",
    "VotingEnsemble",
    "WeightedEnsemble",
    "ClassRoutingEnsemble",
    "FusionEnsemble",
    "FusionMLP",
    "FusionWrapper",
    "AutoFusionClassifier",
]

