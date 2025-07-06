"""Base class for ensemble methods."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..core.base import BaseClassifier, AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType, TrainingData, EnsembleConfig
from ..core.exceptions import EnsembleError, PredictionError, ValidationError


class BaseEnsemble(BaseClassifier):
    """Base class for ensemble methods."""
    
    def __init__(self, ensemble_config: EnsembleConfig):
        """Initialize ensemble with configuration.
        
        Args:
            ensemble_config: Configuration for the ensemble
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
        
        return self._create_result(predictions=ensemble_predictions)
    
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
            confidence_scores=ensemble_confidence
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

