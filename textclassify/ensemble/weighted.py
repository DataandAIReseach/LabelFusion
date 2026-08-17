"""Weighted ensemble for combining multiple classifiers."""

from typing import Dict, List, Union, Optional
import numpy as np

from ..core.types import ClassificationResult, ClassificationType
from ..core.exceptions import EnsembleError
from .base import BaseEnsemble


class WeightedEnsemble(BaseEnsemble):
    """Ensemble that combines predictions using weighted averages."""
    
    def __init__(
        self, 
        ensemble_config,
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True
    ):
        """Initialize weighted ensemble.
        
        Args:
            ensemble_config: Configuration for the ensemble
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
        """
        super().__init__(
            ensemble_config, 
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results
        )
        
        # Set weights
        if ensemble_config.weights:
            if len(ensemble_config.weights) != len(ensemble_config.models):
                raise EnsembleError(
                    "Number of weights must match number of models",
                    ensemble_config.ensemble_method
                )
            self.weights = np.array(ensemble_config.weights)
        else:
            # Equal weights by default
            self.weights = np.ones(len(ensemble_config.models))
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
    
    def add_model(self, model, name=None, weight=None):
        """Add a model to the ensemble with optional weight.
        
        Args:
            model: Classifier to add
            name: Optional name for the model
            weight: Optional weight for the model
        """
        super().add_model(model, name)
        
        if weight is not None:
            # Update weights array
            if len(self.weights) == len(self.models) - 1:
                # First time adding weight, extend array
                self.weights = np.append(self.weights, weight)
            else:
                # Update existing weight
                self.weights[-1] = weight
            
            # Renormalize weights
            self.weights = self.weights / np.sum(self.weights)
    
    def _combine_predictions(self, model_results: List[ClassificationResult], texts: List[str]) -> List[Union[str, List[str]]]:
        """Combine predictions using weights.
        
        Args:
            model_results: Results from all models
            texts: Original texts
            
        Returns:
            Combined predictions
        """
        if not model_results:
            return []
        
        num_texts = len(texts)
        combined_predictions = []
        
        for text_idx in range(num_texts):
            if self.classification_type == ClassificationType.MULTI_CLASS:
                # For multi-class without probabilities, use weighted voting
                class_votes = {}
                total_weight = 0
                
                for model_idx, result in enumerate(model_results):
                    if text_idx < len(result.predictions) and result.predictions[text_idx]:
                        prediction = result.predictions[text_idx]
                        weight = self.weights[model_idx] if model_idx < len(self.weights) else 1.0
                        
                        if prediction not in class_votes:
                            class_votes[prediction] = 0
                        class_votes[prediction] += weight
                        total_weight += weight
                
                if class_votes:
                    # Choose class with highest weighted vote
                    best_class = max(class_votes, key=class_votes.get)
                    combined_predictions.append(best_class)
                else:
                    # Fallback
                    combined_predictions.append(self.classes_[0] if self.classes_ else "unknown")
            
            else:
                # Multi-label classification
                label_scores = {}
                total_weight = 0
                
                for model_idx, result in enumerate(model_results):
                    if text_idx < len(result.predictions) and result.predictions[text_idx]:
                        labels = result.predictions[text_idx]
                        weight = self.weights[model_idx] if model_idx < len(self.weights) else 1.0
                        
                        if isinstance(labels, list):
                            for label in labels:
                                if label not in label_scores:
                                    label_scores[label] = 0
                                label_scores[label] += weight
                        
                        total_weight += weight
                
                if label_scores and total_weight > 0:
                    # Normalize scores and apply threshold
                    threshold = 0.5  # Default threshold
                    final_labels = []
                    
                    for label, score in label_scores.items():
                        normalized_score = score / total_weight
                        if normalized_score >= threshold:
                            final_labels.append(label)
                    
                    combined_predictions.append(final_labels)
                else:
                    combined_predictions.append([])
        
        return combined_predictions
    
    def _combine_predictions_with_probabilities(
        self, 
        model_results: List[ClassificationResult], 
        texts: List[str]
    ) -> tuple:
        """Combine predictions and probabilities using weights.
        
        Args:
            model_results: Results from all models
            texts: Original texts
            
        Returns:
            Tuple of (predictions, probabilities, confidence_scores)
        """
        if not model_results:
            return [], [], []
        
        num_texts = len(texts)
        combined_predictions = []
        combined_probabilities = []
        combined_confidence = []
        
        for text_idx in range(num_texts):
            if self.classification_type == ClassificationType.MULTI_CLASS:
                # Weighted average of probabilities
                weighted_probabilities = {}
                total_weight = 0
                
                for model_idx, result in enumerate(model_results):
                    if (text_idx < len(result.predictions) and 
                        result.probabilities and 
                        text_idx < len(result.probabilities)):
                        
                        prob_dict = result.probabilities[text_idx]
                        weight = self.weights[model_idx] if model_idx < len(self.weights) else 1.0
                        
                        for class_name, prob in prob_dict.items():
                            if class_name not in weighted_probabilities:
                                weighted_probabilities[class_name] = 0
                            weighted_probabilities[class_name] += prob * weight
                        
                        total_weight += weight
                
                if weighted_probabilities and total_weight > 0:
                    # Normalize probabilities
                    for class_name in weighted_probabilities:
                        weighted_probabilities[class_name] /= total_weight
                    
                    # Ensure all classes are represented
                    for class_name in self.classes_:
                        if class_name not in weighted_probabilities:
                            weighted_probabilities[class_name] = 0.0
                    
                    # Get prediction with highest probability
                    best_class = max(weighted_probabilities, key=weighted_probabilities.get)
                    confidence = weighted_probabilities[best_class]
                    
                    combined_predictions.append(best_class)
                    combined_probabilities.append(weighted_probabilities)
                    combined_confidence.append(confidence)
                
                else:
                    # Fallback to hard voting
                    hard_predictions = self._combine_predictions(model_results, [texts[text_idx]])
                    prediction = hard_predictions[0] if hard_predictions else (self.classes_[0] if self.classes_ else "unknown")
                    
                    # Create uniform probabilities
                    uniform_prob = 1.0 / len(self.classes_) if self.classes_ else 0.0
                    probabilities = {cls: uniform_prob for cls in self.classes_}
                    
                    combined_predictions.append(prediction)
                    combined_probabilities.append(probabilities)
                    combined_confidence.append(uniform_prob)
            
            else:
                # Multi-label classification
                weighted_probabilities = {}
                total_weight = 0
                
                for model_idx, result in enumerate(model_results):
                    if (text_idx < len(result.predictions) and 
                        result.probabilities and 
                        text_idx < len(result.probabilities)):
                        
                        prob_dict = result.probabilities[text_idx]
                        weight = self.weights[model_idx] if model_idx < len(self.weights) else 1.0
                        
                        for class_name, prob in prob_dict.items():
                            if class_name not in weighted_probabilities:
                                weighted_probabilities[class_name] = 0
                            weighted_probabilities[class_name] += prob * weight
                        
                        total_weight += weight
                
                if weighted_probabilities and total_weight > 0:
                    # Normalize probabilities
                    for class_name in weighted_probabilities:
                        weighted_probabilities[class_name] /= total_weight
                    
                    # Ensure all classes are represented
                    for class_name in self.classes_:
                        if class_name not in weighted_probabilities:
                            weighted_probabilities[class_name] = 0.0
                    
                    # Apply threshold to determine final labels
                    threshold = 0.5  # Default threshold
                    final_labels = [cls for cls, prob in weighted_probabilities.items() if prob >= threshold]
                    confidence = max(weighted_probabilities.values()) if weighted_probabilities else 0.0
                    
                    combined_predictions.append(final_labels)
                    combined_probabilities.append(weighted_probabilities)
                    combined_confidence.append(confidence)
                
                else:
                    # Fallback to hard voting
                    hard_predictions = self._combine_predictions(model_results, [texts[text_idx]])
                    prediction = hard_predictions[0] if hard_predictions else []
                    
                    # Create uniform probabilities
                    uniform_prob = 0.5
                    probabilities = {cls: uniform_prob for cls in self.classes_}
                    
                    combined_predictions.append(prediction)
                    combined_probabilities.append(probabilities)
                    combined_confidence.append(uniform_prob)
        
        return combined_predictions, combined_probabilities, combined_confidence
    
    def update_weights(self, new_weights: List[float]) -> None:
        """Update model weights.
        
        Args:
            new_weights: New weights for the models
        """
        if len(new_weights) != len(self.models):
            raise EnsembleError(
                "Number of weights must match number of models",
                self.ensemble_config.ensemble_method
            )
        
        self.weights = np.array(new_weights)
        self.weights = self.weights / np.sum(self.weights)  # Normalize
    
    @property
    def model_info(self) -> Dict:
        """Get weighted ensemble information."""
        info = super().model_info
        info.update({
            "weights": self.weights.tolist(),
            "normalized_weights": True
        })
        return info

