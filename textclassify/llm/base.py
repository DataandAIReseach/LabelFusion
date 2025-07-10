"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import PredictionError, ValidationError, APIError
from ..prompt_engineer import PromptEngineer

class BaseLLMClassifier(AsyncBaseClassifier):
    """Base class for all LLM-based text classifiers."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.prompt_engineer = PromptEngineer()
        self.classes_: List[str] = []
        self._setup_config()

    def _setup_config(self) -> None:
        """Initialize configuration parameters."""
        self.batch_size = self.config.parameters.get('batch_size', 32)
        self.threshold = self.config.parameters.get('threshold', 0.5)

    def predict(self, texts: List[str]) -> ClassificationResult:
        """Synchronous wrapper for predictions."""
        return asyncio.run(self.predict_async(texts))

    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Synchronous wrapper for probability predictions."""
        return asyncio.run(self.predict_proba_async(texts))

    async def predict_async(
        self,
        texts: List[str],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        text_column: str = "text",
        label_column: str = "label",
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None,
        few_shot_mode: str = "few_shot"
    ) -> ClassificationResult:
        """Asynchronously predict labels for texts with optional train/test data."""
        try:
            self._validate_prediction_inputs(texts)
            self._setup_prompt_configuration(context, label_definitions, few_shot_mode)
            
            if train_df is not None:
                self._setup_few_shot_learning(train_df, text_column, label_column)
            
            predictions = await self._generate_predictions(texts)
            metrics = await self._evaluate_test_data(test_df, text_column, label_column) if test_df is not None else None
            
            return self._create_result(predictions=predictions, metrics=metrics)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.model_name)

    def _validate_prediction_inputs(self, texts: List[str]) -> None:
        """Validate prediction inputs."""
        if not texts:
            raise ValidationError("No texts provided for prediction")
        if not all(isinstance(text, str) for text in texts):
            raise ValidationError("All inputs must be strings")

    def _setup_prompt_configuration(
        self,
        context: Optional[str],
        label_definitions: Optional[Dict[str, str]],
        few_shot_mode: str
    ) -> None:
        """Configure the prompt engineer."""
        if context is not None:
            self.prompt_engineer.context = context
        if label_definitions is not None:
            self.prompt_engineer.label_definitions = label_definitions
        self.prompt_engineer.set_few_shot_mode(few_shot_mode)

    def _setup_few_shot_learning(
        self,
        train_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> None:
        """Configure few-shot learning with training data."""
        train_texts = train_df[text_column].tolist()
        train_labels = train_df[label_column].tolist()
        
        label_type = self._infer_label_type(train_labels)
        self.classes_ = self._extract_unique_classes(train_labels)
        
        self.prompt_engineer.set_labels(self.classes_, label_type=label_type)
        self.prompt_engineer.set_examples(train_texts, train_labels)

    def _infer_label_type(self, labels: List[Any]) -> str:
        """Infer the type of labels (single or multiple)."""
        if all(isinstance(label, str) for label in labels):
            return "single"
        elif all(isinstance(label, list) for label in labels):
            return "multiple"
        raise ValidationError("Inconsistent label format")

    def _extract_unique_classes(self, labels: List[Any]) -> List[str]:
        """Extract unique class labels."""
        if all(isinstance(label, str) for label in labels):
            return list(set(labels))
        all_labels = []
        for label_list in labels:
            all_labels.extend(label_list)
        return list(set(all_labels))

    async def _generate_predictions(self, texts: List[str]) -> List[Union[str, List[str]]]:
        """Generate predictions in batches."""
        predictions = []
        for batch in self._get_batches(texts):
            batch_predictions = await self._process_batch(batch)
            predictions.extend(batch_predictions)
        return predictions

    def _get_batches(self, texts: List[str]) -> Iterator[List[str]]:
        """Yield batches of texts."""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]

    async def _process_batch(self, texts: List[str]) -> List[Union[str, List[str]]]:
        """Process a batch of texts for prediction."""
        prompts = [self.prompt_engineer.build_prompt(text) for text in texts]
        responses = await asyncio.gather(
            *[self._call_llm(prompt) for prompt in prompts],
            return_exceptions=True
        )
        return [
            self._parse_prediction_response(r) if not isinstance(r, Exception)
            else self._handle_error(r)
            for r in responses
        ]

    def _parse_prediction_response(self, response: str) -> Union[str, List[str]]:
        """Parse the LLM response for predictions."""
        response = response.strip()
        if self.prompt_engineer.label_type == "single":
            return self._parse_single_label(response)
        return self._parse_multiple_labels(response)

    def _parse_single_label(self, response: str) -> str:
        """Parse response for single-label classification."""
        for class_name in self.classes_:
            if class_name.lower() in response.lower():
                return class_name
        return self.classes_[0] if self.classes_ else "unknown"

    def _parse_multiple_labels(self, response: str) -> List[str]:
        """Parse response for multi-label classification."""
        if response.upper() == "NONE":
            return []
        predicted_classes = []
        for part in response.split(','):
            part = part.strip()
            for class_name in self.classes_:
                if class_name.lower() in part.lower():
                    predicted_classes.append(class_name)
                    break
        return predicted_classes

    async def _evaluate_test_data(
        self,
        test_df: pd.DataFrame,
        text_column: str,
        label_column: str
    ) -> Optional[Dict[str, float]]:
        """Evaluate model performance on test data."""
        if test_df is None:
            return None
        test_texts = test_df[text_column].tolist()
        test_labels = test_df[label_column].tolist()
        
        test_predictions = await self._generate_predictions(test_texts)
        return self._calculate_metrics(test_predictions, test_labels)

    def _calculate_metrics(
        self,
        predictions: List[Union[str, List[str]]],
        true_labels: List[Union[str, List[str]]]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if all(isinstance(label, str) for label in true_labels):
            return self._calculate_single_label_metrics(predictions, true_labels)
        return self._calculate_multi_label_metrics(predictions, true_labels)

    def _calculate_single_label_metrics(
        self,
        predictions: List[str],
        true_labels: List[str]
    ) -> Dict[str, float]:
        """Calculate metrics for single-label classification."""
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        return {'accuracy': correct / len(true_labels)}

    def _calculate_multi_label_metrics(
        self,
        predictions: List[List[str]],
        true_labels: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate metrics for multi-label classification."""
        metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for pred, true in zip(predictions, true_labels):
            pred_set = set(pred)
            true_set = set(true)
            
            if not (pred_set or true_set):
                continue
                
            precision = len(pred_set & true_set) / len(pred_set) if pred_set else 0
            recall = len(pred_set & true_set) / len(true_set) if true_set else 0
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            if precision + recall > 0:
                metrics['f1'].append(2 * precision * recall / (precision + recall))
        
        return {
            k: sum(v) / len(v) if v else 0.0 
            for k, v in metrics.items()
        }

    def _handle_error(self, error: Exception) -> Union[str, List[str]]:
        """Handle prediction errors."""
        raise PredictionError(f"LLM call failed: {str(error)}", self.config.model_name)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt."""
        raise NotImplementedError("Subclasses must implement _call_llm method")