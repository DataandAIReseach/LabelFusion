"""Class routing ensemble for using different models for different classes."""

from typing import Dict, List, Union, Optional
import numpy as np

from ..core.types import ClassificationResult, ClassificationType
from ..core.exceptions import EnsembleError
from .base import BaseEnsemble


class ClassRoutingEnsemble(BaseEnsemble):
    """Ensemble that routes different classes to different models."""
    
    def __init__(self, ensemble_config):
        """Initialize class routing ensemble.
        
        Args:
            ensemble_config: Configuration for the ensemble
        """
        super().__init__(ensemble_config)
        
        # Set up routing rules
        self.routing_rules = ensemble_config.routing_rules or {}
        self.default_model_idx = 0  # Default to first model
        self.class_to_model = {}  # Maps class names to model indices
        
        # Fallback strategy
        self.fallback_strategy = "voting"  # "voting", "weighted", "first"
    
    def fit(self, training_data):
        """Fit all models and set up routing rules.
        
        Args:
            training_data: Training data for all models
        """
        super().fit(training_data)
        
        # Set up class-to-model mapping
        self._setup_routing()
    
    def _setup_routing(self):
        """Set up routing rules mapping classes to models."""
        # Clear existing mapping
        self.class_to_model = {}
        
        # Apply explicit routing rules
        for class_name, model_name in self.routing_rules.items():
            if model_name in self.model_names:
                model_idx = self.model_names.index(model_name)
                self.class_to_model[class_name] = model_idx
            else:
                print(f"Warning: Model '{model_name}' not found for class '{class_name}'")
        
        # For classes without explicit rules, use default model
        for class_name in self.classes_:
            if class_name not in self.class_to_model:
                self.class_to_model[class_name] = self.default_model_idx
    
    def add_routing_rule(self, class_name: str, model_name: str):
        """Add a routing rule for a specific class.
        
        Args:
            class_name: Name of the class
            model_name: Name of the model to use for this class
        """
        if model_name not in self.model_names:
            raise EnsembleError(f"Model '{model_name}' not found in ensemble", self.ensemble_config.ensemble_method)
        
        model_idx = self.model_names.index(model_name)
        self.class_to_model[class_name] = model_idx
        self.routing_rules[class_name] = model_name
    
    def _combine_predictions(self, model_results: List[ClassificationResult], texts: List[str]) -> List[Union[str, List[str]]]:
        """Combine predictions using class routing.
        
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
                # For multi-class, we need to determine which model to trust
                # Get predictions from all models first
                text_predictions = []
                for result in model_results:
                    if text_idx < len(result.predictions) and result.predictions[text_idx]:
                        text_predictions.append(result.predictions[text_idx])
                
                if text_predictions:
                    # Use routing rules to select the best prediction
                    best_prediction = self._route_multiclass_prediction(text_predictions)
                    combined_predictions.append(best_prediction)
                else:
                    combined_predictions.append(self.classes_[0] if self.classes_ else "unknown")
            
            else:
                # Multi-label classification
                final_labels = []
                
                # For each possible class, check if any model assigned to it predicts it
                for class_name in self.classes_:
                    model_idx = self.class_to_model.get(class_name, self.default_model_idx)
                    
                    if model_idx < len(model_results):
                        result = model_results[model_idx]
                        if text_idx < len(result.predictions) and result.predictions[text_idx]:
                            predicted_labels = result.predictions[text_idx]
                            if isinstance(predicted_labels, list) and class_name in predicted_labels:
                                final_labels.append(class_name)
                
                combined_predictions.append(final_labels)
        
        return combined_predictions
    
    def _combine_predictions_with_probabilities(
        self, 
        model_results: List[ClassificationResult], 
        texts: List[str]
    ) -> tuple:
        """Combine predictions and probabilities using class routing.
        
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
                # Route each class to its designated model
                routed_probabilities = {}
                max_confidence = 0.0
                best_prediction = None
                
                for class_name in self.classes_:
                    model_idx = self.class_to_model.get(class_name, self.default_model_idx)
                    
                    if model_idx < len(model_results):
                        result = model_results[model_idx]
                        if (text_idx < len(result.predictions) and 
                            result.probabilities and 
                            text_idx < len(result.probabilities)):
                            
                            prob_dict = result.probabilities[text_idx]
                            if class_name in prob_dict:
                                prob = prob_dict[class_name]
                                routed_probabilities[class_name] = prob
                                
                                if prob > max_confidence:
                                    max_confidence = prob
                                    best_prediction = class_name
                            else:
                                routed_probabilities[class_name] = 0.0
                        else:
                            routed_probabilities[class_name] = 0.0
                    else:
                        routed_probabilities[class_name] = 0.0
                
                # Normalize probabilities to sum to 1
                total_prob = sum(routed_probabilities.values())
                if total_prob > 0:
                    for class_name in routed_probabilities:
                        routed_probabilities[class_name] /= total_prob
                else:
                    # Uniform distribution as fallback
                    uniform_prob = 1.0 / len(self.classes_) if self.classes_ else 0.0
                    routed_probabilities = {cls: uniform_prob for cls in self.classes_}
                    best_prediction = self.classes_[0] if self.classes_ else "unknown"
                    max_confidence = uniform_prob
                
                if best_prediction is None:
                    best_prediction = max(routed_probabilities, key=routed_probabilities.get)
                    max_confidence = routed_probabilities[best_prediction]
                
                combined_predictions.append(best_prediction)
                combined_probabilities.append(routed_probabilities)
                combined_confidence.append(max_confidence)
            
            else:
                # Multi-label classification
                routed_probabilities = {}
                final_labels = []
                max_confidence = 0.0
                
                for class_name in self.classes_:
                    model_idx = self.class_to_model.get(class_name, self.default_model_idx)
                    
                    if model_idx < len(model_results):
                        result = model_results[model_idx]
                        if (text_idx < len(result.predictions) and 
                            result.probabilities and 
                            text_idx < len(result.probabilities)):
                            
                            prob_dict = result.probabilities[text_idx]
                            if class_name in prob_dict:
                                prob = prob_dict[class_name]
                                routed_probabilities[class_name] = prob
                                
                                # Apply threshold
                                threshold = 0.5
                                if prob >= threshold:
                                    final_labels.append(class_name)
                                
                                max_confidence = max(max_confidence, prob)
                            else:
                                routed_probabilities[class_name] = 0.0
                        else:
                            routed_probabilities[class_name] = 0.0
                    else:
                        routed_probabilities[class_name] = 0.0
                
                combined_predictions.append(final_labels)
                combined_probabilities.append(routed_probabilities)
                combined_confidence.append(max_confidence)
        
        return combined_predictions, combined_probabilities, combined_confidence
    
    def _route_multiclass_prediction(self, predictions: List[str]) -> str:
        """Route multi-class prediction based on class-specific models.
        
        Args:
            predictions: List of predictions from all models
            
        Returns:
            Best prediction based on routing rules
        """
        # Count votes for each prediction, weighted by routing preference
        prediction_scores = {}
        
        for i, prediction in enumerate(predictions):
            if prediction in self.class_to_model:
                # This prediction comes from the model assigned to this class
                preferred_model_idx = self.class_to_model[prediction]
                if i == preferred_model_idx:
                    # This prediction comes from its preferred model
                    score = 2.0  # Higher weight for preferred model
                else:
                    score = 1.0  # Normal weight for non-preferred model
            else:
                score = 1.0  # Normal weight for unknown class
            
            if prediction not in prediction_scores:
                prediction_scores[prediction] = 0
            prediction_scores[prediction] += score
        
        if prediction_scores:
            return max(prediction_scores, key=prediction_scores.get)
        else:
            return self.classes_[0] if self.classes_ else "unknown"
    
    def get_model_for_class(self, class_name: str) -> Optional[int]:
        """Get the model index assigned to a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Model index or None if class not found
        """
        return self.class_to_model.get(class_name)
    
    def get_routing_summary(self) -> Dict[str, str]:
        """Get a summary of routing rules.
        
        Returns:
            Dictionary mapping class names to model names
        """
        summary = {}
        for class_name, model_idx in self.class_to_model.items():
            if model_idx < len(self.model_names):
                summary[class_name] = self.model_names[model_idx]
            else:
                summary[class_name] = f"model_{model_idx}"
        return summary
    
    @property
    def model_info(self) -> Dict:
        """Get class routing ensemble information."""
        info = super().model_info
        info.update({
            "routing_rules": self.routing_rules,
            "class_to_model": self.class_to_model,
            "default_model_idx": self.default_model_idx,
            "fallback_strategy": self.fallback_strategy,
            "routing_summary": self.get_routing_summary()
        })
        return info

