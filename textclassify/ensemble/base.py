"""Base class for ensemble methods."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseClassifier, AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType, TrainingData, EnsembleConfig
from ..core.exceptions import EnsembleError, PredictionError, ValidationError
from ..utils.results_manager import ResultsManager, ModelResultsManager


class BaseEnsemble(BaseClassifier):
    """Base class for ensemble methods."""
    
    def __init__(
        self, 
        ensemble_config: EnsembleConfig,
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True
    ):
        """Initialize ensemble with configuration.
        
        Args:
            ensemble_config: Configuration for the ensemble
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
        """
        # Create a dummy config for the base class
        from ..core.types import ModelConfig
        dummy_config = ModelConfig(
            model_name=f"ensemble_{ensemble_config.ensemble_method}",
            model_type=ModelType.ENSEMBLE
        )
        
        super().__init__(dummy_config)
        self.ensemble_config = ensemble_config
        self.models = []
        self.model_names = []
        
        # Setup results management
        self.results_manager = None
        self.model_results_manager = None
        
        if auto_save_results:
            if not experiment_name:
                experiment_name = f"{ensemble_config.ensemble_method}_ensemble"
            
            self.results_manager = ResultsManager(
                base_output_dir=output_dir,
                experiment_name=experiment_name
            )
            self.model_results_manager = ModelResultsManager(
                self.results_manager,
                f"{ensemble_config.ensemble_method}_ensemble_{self.results_manager.experiment_id}"
            )
        
        # Validate ensemble configuration
        if not ensemble_config.models:
            raise EnsembleError("Ensemble must have at least one model", ensemble_config.ensemble_method)
    
    def add_model(self, model: BaseClassifier, name: Optional[str] = None) -> None:
        """Add a model to the ensemble.
        
        Args:
            model: Classifier to add to the ensemble
            name: Optional name for the model
        """
        if not isinstance(model, BaseClassifier):
            raise EnsembleError("Model must be a BaseClassifier instance", self.ensemble_config.ensemble_method)
        
        self.models.append(model)
        model_name = name or f"model_{len(self.models)}"
        self.model_names.append(model_name)
    
    def fit(self, training_data: TrainingData) -> None:
        """Fit all models in the ensemble.
        
        Args:
            training_data: Training data for all models
        """
        if not self.models:
            raise EnsembleError("No models added to ensemble", self.ensemble_config.ensemble_method)
        
        self.classification_type = training_data.classification_type
        
        # Collect all unique classes from all models after training
        all_classes = set()
        
        for i, model in enumerate(self.models):
            try:
                model.fit(training_data)
                if model.classes_:
                    all_classes.update(model.classes_)
            except Exception as e:
                if self.ensemble_config.require_all_models:
                    raise EnsembleError(
                        f"Failed to train model {self.model_names[i]}: {str(e)}",
                        self.ensemble_config.ensemble_method
                    )
                else:
                    print(f"Warning: Failed to train model {self.model_names[i]}: {str(e)}")
        
        self.classes_ = list(all_classes)
        self.is_trained = True
    
    def predict(self, texts: List[str]) -> ClassificationResult:
        """Predict using ensemble method.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with ensemble predictions
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Ensemble must be trained before prediction", self.config.model_name)
        
        # Get predictions from all models
        model_results = self._get_model_predictions(texts, with_probabilities=False)
        
        # Combine predictions using ensemble method
        ensemble_predictions = self._combine_predictions(model_results, texts)
        
        return self._create_result(predictions=ensemble_predictions, texts=texts)
    
    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Predict probabilities using ensemble method.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            ClassificationResult with ensemble predictions and probabilities
        """
        self.validate_input(texts)
        
        if not self.is_trained:
            raise PredictionError("Ensemble must be trained before prediction", self.config.model_name)
        
        # Get predictions with probabilities from all models
        model_results = self._get_model_predictions(texts, with_probabilities=True)
        
        # Combine predictions and probabilities using ensemble method
        ensemble_predictions, ensemble_probabilities, ensemble_confidence = self._combine_predictions_with_probabilities(
            model_results, texts
        )
        
        return self._create_result(
            predictions=ensemble_predictions,
            probabilities=ensemble_probabilities,
            confidence_scores=ensemble_confidence,
            texts=texts
        )
    
    def _get_model_predictions(self, texts: List[str], with_probabilities: bool = False) -> List[ClassificationResult]:
        """Get predictions from all models.
        
        Args:
            texts: List of texts to classify
            with_probabilities: Whether to get probabilities
            
        Returns:
            List of ClassificationResult from each model
        """
        model_results = []
        
        for i, model in enumerate(self.models):
            try:
                if with_probabilities:
                    result = model.predict_proba(texts)
                else:
                    result = model.predict(texts)
                model_results.append(result)
            except Exception as e:
                if self.ensemble_config.require_all_models:
                    raise EnsembleError(
                        f"Failed to get predictions from model {self.model_names[i]}: {str(e)}",
                        self.ensemble_config.ensemble_method
                    )
                else:
                    print(f"Warning: Failed to get predictions from model {self.model_names[i]}: {str(e)}")
                    # Add empty result as placeholder
                    empty_predictions = [""] * len(texts) if self.classification_type == ClassificationType.MULTI_CLASS else [[] for _ in texts]
                    empty_result = ClassificationResult(predictions=empty_predictions)
                    model_results.append(empty_result)
        
        return model_results
    
    @abstractmethod
    def _combine_predictions(self, model_results: List[ClassificationResult], texts: List[str]) -> List[Union[str, List[str]]]:
        """Combine predictions from multiple models.
        
        Args:
            model_results: Results from all models
            texts: Original texts (for context)
            
        Returns:
            Combined predictions
        """
        pass
    
    @abstractmethod
    def _combine_predictions_with_probabilities(
        self, 
        model_results: List[ClassificationResult], 
        texts: List[str]
    ) -> tuple:
        """Combine predictions and probabilities from multiple models.
        
        Args:
            model_results: Results from all models
            texts: Original texts (for context)
            
        Returns:
            Tuple of (predictions, probabilities, confidence_scores)
        """
        pass
    
    def _handle_async_models(self, texts: List[str], with_probabilities: bool = False) -> List[ClassificationResult]:
        """Handle asynchronous models in the ensemble.
        
        Args:
            texts: List of texts to classify
            with_probabilities: Whether to get probabilities
            
        Returns:
            List of ClassificationResult from each model
        """
        async def get_async_predictions():
            tasks = []
            
            for model in self.models:
                if isinstance(model, AsyncBaseClassifier):
                    if with_probabilities:
                        task = model.predict_proba_async(texts)
                    else:
                        task = model.predict_async(texts)
                else:
                    # For synchronous models, wrap in async
                    if with_probabilities:
                        task = asyncio.create_task(asyncio.to_thread(model.predict_proba, texts))
                    else:
                        task = asyncio.create_task(asyncio.to_thread(model.predict, texts))
                
                tasks.append(task)
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run async predictions
        results = asyncio.run(get_async_predictions())
        
        # Handle exceptions
        model_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if self.ensemble_config.require_all_models:
                    raise EnsembleError(
                        f"Failed to get predictions from model {self.model_names[i]}: {str(result)}",
                        self.ensemble_config.ensemble_method
                    )
                else:
                    print(f"Warning: Failed to get predictions from model {self.model_names[i]}: {str(result)}")
                    # Add empty result as placeholder
                    empty_predictions = [""] * len(texts) if self.classification_type == ClassificationType.MULTI_CLASS else [[] for _ in texts]
                    empty_result = ClassificationResult(predictions=empty_predictions)
                    model_results.append(empty_result)
            else:
                model_results.append(result)
        
        return model_results
    
    def _create_result(
        self,
        predictions: List[Union[str, List[str]]],
        probabilities: Optional[List[Dict[str, float]]] = None,
        confidence_scores: Optional[List[float]] = None,
        texts: Optional[List[str]] = None,
        true_labels: Optional[List[List[int]]] = None
    ) -> ClassificationResult:
        """Create a ClassificationResult object with ensemble predictions and save results.
        
        Args:
            predictions: List of predictions
            probabilities: Optional list of probability dictionaries
            confidence_scores: Optional list of confidence scores
            texts: Optional list of input texts
            true_labels: Optional list of true labels for evaluation
            
        Returns:
            ClassificationResult object
        """
        # Calculate metrics if true labels are provided
        metrics = None
        if true_labels is not None:
            # Convert string predictions to binary vectors for metric calculation
            binary_predictions = []
            for pred in predictions:
                if isinstance(pred, str):
                    # Single-label: create binary vector
                    binary_vec = [0] * len(self.classes_)
                    if pred in self.classes_:
                        idx = self.classes_.index(pred)
                        binary_vec[idx] = 1
                    binary_predictions.append(binary_vec)
                else:
                    # Multi-label: create binary vector
                    binary_vec = [0] * len(self.classes_)
                    for label in pred:
                        if label in self.classes_:
                            idx = self.classes_.index(label)
                            binary_vec[idx] = 1
                    binary_predictions.append(binary_vec)
            
            metrics = self._calculate_metrics(binary_predictions, true_labels)
        
        # Create metadata with metrics and ensemble info
        metadata = {
            "metrics": metrics or {},
            "ensemble_method": self.ensemble_config.ensemble_method,
            "num_models": len(self.models),
            "model_names": self.model_names
        }
        
        result = ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence_scores=confidence_scores,
            model_name=f"{self.ensemble_config.ensemble_method}_ensemble",
            model_type=ModelType.ENSEMBLE,
            classification_type=self.classification_type,
            metadata=metadata
        )
        
        # Save results using ResultsManager
        if self.results_manager and texts is not None:
            try:
                # Determine dataset type
                dataset_type = "test" if true_labels is not None else "prediction"
                
                # Create DataFrame for saving
                import pandas as pd
                df = pd.DataFrame({
                    'text': texts,
                    'prediction': predictions
                })
                if true_labels is not None:
                    df['true_labels'] = [','.join(map(str, labels)) for labels in true_labels]
                
                # Save predictions
                saved_files = self.results_manager.save_predictions(
                    result, dataset_type, df
                )
                
                # Save metrics if available
                if metrics:
                    metrics_file = self.results_manager.save_metrics(
                        metrics, dataset_type, f"{self.ensemble_config.ensemble_method}_ensemble"
                    )
                    saved_files["metrics"] = metrics_file
                
                # Save ensemble configuration
                ensemble_config = {
                    'ensemble_method': self.ensemble_config.ensemble_method,
                    'num_models': len(self.models),
                    'model_names': self.model_names,
                    'require_all_models': self.ensemble_config.require_all_models,
                    'classification_type': self.classification_type.value if self.classification_type else 'unknown'
                }
                
                config_file = self.results_manager.save_model_config(
                    ensemble_config, f"{self.ensemble_config.ensemble_method}_ensemble"
                )
                saved_files["config"] = config_file
                
                # Save experiment summary
                experiment_summary = {
                    'model_name': f"{self.ensemble_config.ensemble_method}_ensemble",
                    'model_type': 'ensemble',
                    'classification_type': self.classification_type.value if self.classification_type else 'unknown',
                    'num_predictions': len(predictions),
                    'ensemble_method': self.ensemble_config.ensemble_method,
                    'num_models': len(self.models),
                    'accuracy': metrics.get('accuracy', 0.0) if metrics else 0.0,
                    'completed': True
                }
                
                if self.results_manager:
                    self.results_manager.save_experiment_summary(experiment_summary)
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
            except Exception as e:
                print(f"Warning: Could not save ensemble results: {e}")
        
        return result
    
    def _calculate_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for ensemble predictions.
        
        Args:
            predictions: Binary prediction vectors
            true_labels: Binary true label vectors
            
        Returns:
            Dictionary of metrics
        """
        if self.classification_type == ClassificationType.MULTI_CLASS:
            return self._calculate_single_label_metrics(predictions, true_labels)
        else:
            return self._calculate_multi_label_metrics(predictions, true_labels)
    
    def _calculate_single_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for single-label classification."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        import numpy as np
        
        # Convert binary vectors to class indices
        pred_classes = [np.argmax(pred) if sum(pred) > 0 else 0 for pred in predictions]
        true_classes = [np.argmax(true) if sum(true) > 0 else 0 for true in true_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def _calculate_multi_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for multi-label classification."""
        from sklearn.metrics import hamming_loss, jaccard_score
        import numpy as np
        
        pred_array = np.array(predictions)
        true_array = np.array(true_labels)
        
        # Calculate metrics
        hamming = hamming_loss(true_array, pred_array)
        jaccard = jaccard_score(true_array, pred_array, average='weighted', zero_division=0)
        
        # Calculate accuracy (exact match ratio)
        exact_match = np.all(pred_array == true_array, axis=1).mean()
        
        return {
            'hamming_loss': float(hamming),
            'jaccard_score': float(jaccard),
            'exact_match_ratio': float(exact_match),
            'accuracy': float(exact_match)  # For consistency
        }

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        info = super().model_info
        info.update({
            "ensemble_method": self.ensemble_config.ensemble_method,
            "num_models": len(self.models),
            "model_names": self.model_names,
            "require_all_models": self.ensemble_config.require_all_models,
            "fallback_model": self.ensemble_config.fallback_model
        })
        
        # Add individual model info
        model_info = []
        for i, model in enumerate(self.models):
            model_info.append({
                "name": self.model_names[i],
                "type": model.config.model_type.value,
                "is_trained": model.is_trained
            })
        info["models"] = model_info
        
        return info

