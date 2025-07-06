"""Ensemble methods for combining multiple classifiers."""

from .base import BaseEnsemble
from .voting import VotingEnsemble
from .weighted import WeightedEnsemble
from .routing import ClassRoutingEnsemble

__all__ = [
    "BaseEnsemble",
    "VotingEnsemble",
    "WeightedEnsemble",
    "ClassRoutingEnsemble",
]

