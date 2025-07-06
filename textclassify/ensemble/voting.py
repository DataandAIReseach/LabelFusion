"""Voting ensemble for combining multiple classifiers."""

from collections import Counter
from typing import Dict, List, Union
import numpy as np

from ..core.types import ClassificationResult, ClassificationType
from .base import BaseEnsemble


class VotingEnsemble(BaseEnsemble):
    """Ensemble that combines predictions through voting."""
    
    def __init__(self, ensemble_config):
        """Initialize voting ensemble.
        
        Args:
            ensemble_config: Configuration for the ensemble
        """
        super().__init__(ensemble_config)
        self.voting_strategy = ensemble_config.ensemble_method
        
        # Validate voting strategy
        valid_strategies = ['hard', 'soft', 'majority', 'plurality']
        if self.voting_strategy not in valid_strategies:
            self.voting_strategy = 'majority'  # Default strategy
    
    def _combine_predictions(self, model_results: List[ClassificationResult], texts: List[str]) -> List[Union[str, List[str]]]:
        """Combine predictions using voting.
        
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
                # Collect predictions for this text from all models
                text_predictions = []
                for result in model_results:
                    if text_idx < len(result.predictions) and result.predictions[text_idx]:
                        text_predictions.append(result.predictions[text_idx])
                
                # Vote for the best prediction
                if text_predictions:
                    if self.voting_strategy in ['majority', 'plurality']:
                        # Use most common prediction
                        vote_counts = Counter(text_predictions)
                        most_common = vote_counts.most_common(1)[0]
                        
                        if self.voting_strategy == 'majority' and most_common[1] <= len(text_predictions) // 2:
                            # No majority, use first prediction as fallback
                            combined_prediction = text_predictions[0]
                        else:
                            combined_prediction = most_common[0]
                    else:
                        # Default to first prediction
                        combined_prediction = text_predictions[0]
                else:
                    # No valid predictions, use first class as fallback
                    combined_prediction = self.classes_[0] if self.classes_ else "unknown"
                
                combined_predictions.append(combined_prediction)
            
            else:
                # Multi-label classification
                # Collect all predicted labels from all models
                all_labels = set()
                label_votes = Counter()
                
                for result in model_results:
                    if text_idx < len(result.predictions) and result.predictions[text_idx]:
                        labels = result.predictions[text_idx]
                        if isinstance(labels, list):
                            for label in labels:
                                label_votes[label] += 1
                                all_labels.add(label)
                
                # Determine final labels based on voting strategy
                if self.voting_strategy == 'majority':
                    # Require majority vote for each label
                    threshold = len(model_results) // 2 + 1
                    final_labels = [label for label, count in label_votes.items() if count >= threshold]
                elif self.voting_strategy == 'plurality':
                    # Use any label that got at least one vote
                    final_labels = list(label_votes.keys())
                else:
                    # Default: use labels that got more than half the votes
                    threshold = max(1, len(model_results) // 2)
                    final_labels = [label for label, count in label_votes.items() if count > threshold]
                
                combined_predictions.append(final_labels)
        
        return combined_predictions
    
    def _combine_predictions_with_probabilities(
        self, 
        model_results: List[ClassificationResult], 
        texts: List[str]
    ) -> tuple:
        """Combine predictions and probabilities using voting.
        
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
                # Collect probabilities for this text from all models
                text_probabilities = []
                valid_models = 0
                
                for result in model_results:
                    if (text_idx < len(result.predictions) and 
                        result.probabilities and 
                        text_idx < len(result.probabilities)):
                        text_probabilities.append(result.probabilities[text_idx])
                        valid_models += 1
                
                if text_probabilities and valid_models > 0:
                    # Average probabilities across models
                    avg_probabilities = {}
                    for class_name in self.classes_:
                        class_probs = []
                        for prob_dict in text_probabilities:
                            if class_name in prob_dict:
                                class_probs.append(prob_dict[class_name])
                        
                        if class_probs:
                            avg_probabilities[class_name] = np.mean(class_probs)
                        else:
                            avg_probabilities[class_name] = 0.0
                    
                    # Get prediction with highest average probability
                    best_class = max(avg_probabilities, key=avg_probabilities.get)
                    confidence = avg_probabilities[best_class]
                    
                    combined_predictions.append(best_class)
                    combined_probabilities.append(avg_probabilities)
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
                # Collect probabilities for each class
                class_probabilities = {cls: [] for cls in self.classes_}
                valid_models = 0
                
                for result in model_results:
                    if (text_idx < len(result.predictions) and 
                        result.probabilities and 
                        text_idx < len(result.probabilities)):
                        prob_dict = result.probabilities[text_idx]
                        for class_name in self.classes_:
                            if class_name in prob_dict:
                                class_probabilities[class_name].append(prob_dict[class_name])
                        valid_models += 1
                
                if valid_models > 0:
                    # Average probabilities for each class
                    avg_probabilities = {}
                    for class_name in self.classes_:
                        if class_probabilities[class_name]:
                            avg_probabilities[class_name] = np.mean(class_probabilities[class_name])
                        else:
                            avg_probabilities[class_name] = 0.0
                    
                    # Apply threshold to determine final labels
                    threshold = 0.5  # Default threshold
                    if hasattr(self.ensemble_config, 'threshold'):
                        threshold = self.ensemble_config.threshold
                    
                    final_labels = [cls for cls, prob in avg_probabilities.items() if prob >= threshold]
                    confidence = max(avg_probabilities.values()) if avg_probabilities else 0.0
                    
                    combined_predictions.append(final_labels)
                    combined_probabilities.append(avg_probabilities)
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

