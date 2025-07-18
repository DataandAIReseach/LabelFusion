"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import PredictionError, ValidationError, APIError
from ..prompt_engineer.base import PromptEngineer

class BaseLLMClassifier(AsyncBaseClassifier):
    """Base class for all LLM-based text classifiers."""
    
    def __init__(
        self, 
        config,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot"
    ):
        """Initialize the LLM classifier.
        
        Args:
            config: Configuration object
            multi_label: Whether this is a multi-label classifier (default: False)
            few_shot_mode: Mode for few-shot learning (default: "few_shot")
        """
        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.multi_label = multi_label
        self.few_shot_mode = few_shot_mode
        
        # Initialize prompt engineer with configuration
        self.prompt_engineer = PromptEngineer(
            multi_label=self.multi_label,
            few_shot_mode=self.few_shot_mode
        )
        self.classes_: List[str] = []
        self._setup_config()

    def _setup_config(self) -> None:
        """Initialize configuration parameters."""
        self.batch_size = self.config.parameters.get('batch_size', 32)
        self.threshold = self.config.parameters.get('threshold', 0.5)

    def predict(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_columns: List[str] = None
    ) -> ClassificationResult:
        """Synchronous wrapper for predictions."""
        return asyncio.run(self.predict_async(
            df=df,
            text_column=text_column,
            label_columns=label_columns
        ))

    def predict_proba(self, texts: List[str]) -> ClassificationResult:
        """Synchronous wrapper for probability predictions."""
        return asyncio.run(self.predict_proba_async(texts))

    async def predict_async(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        text_column: str = "text",
        label_columns: List[str] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Asynchronously predict labels for texts."""
        try:
            if label_columns is None:
                label_columns = getattr(self.config, 'label_columns', ["label"])
                
            self._validate_prediction_inputs(
                df=df,
                text_column=text_column,
                label_columns=label_columns
            )
            
            self._setup_prompt_configuration(context, label_definitions, few_shot_mode)
            
            if train_df is not None:
                self._validate_prediction_inputs(
                    df=train_df,
                    text_column=text_column,
                    label_columns=label_columns,
                    multi_label=multi_label
                )
                
            predictions = await self._generate_predictions(df, text_column)
            metrics = await self._evaluate_test_data(
                test_df=test_df,
                text_column=text_column,
                label_columns=label_columns
            ) if test_df is not None else None
            
            return self._create_result(predictions=predictions, metrics=metrics)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.model_name)

    def _validate_prediction_inputs(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        label_columns: List[str],
        multi_label: bool
    ) -> None:
        """Validate prediction inputs.
        
        Args:
            df: DataFrame to validate
            text_column: Name of the column containing text data
            label_columns: Names of the columns containing labels
            multi_label: Whether this is a multi-label classification task
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValidationError("DataFrame is empty")
            
        # Validate text column
        if text_column not in df.columns:
            raise ValidationError(f"Text column '{text_column}' not found in DataFrame")
        if not df[text_column].dtype == 'object':
            raise ValidationError(f"Text column '{text_column}' must be of type string/object")
        if not df[text_column].apply(lambda x: isinstance(x, str)).all():
            raise ValidationError(f"All entries in text column '{text_column}' must be strings")
            
        # Validate label columns
        for label_col in label_columns:
            if label_col not in df.columns:
                raise ValidationError(f"Label column '{label_col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[label_col]):
                raise ValidationError(f"Label column '{label_col}' must contain only numbers")
            if not df[label_col].isin([0, 1]).all():
                raise ValidationError(f"Label column '{label_col}' must contain only binary values (0 or 1)")
        
        # Get sum of 1s for each row across label columns
        label_sums = df[label_columns].sum(axis=1)
        
        if not multi_label:
            # For single-label: check if exactly one 1 per row
            invalid_rows = label_sums != 1
            if invalid_rows.any():
                problematic_rows = df.index[invalid_rows].tolist()
                raise ValidationError(
                    f"Single-label classification requires exactly one 1 per row. "
                    f"Problematic rows: {problematic_rows}"
                )
        else:
            # For multi-label: check if number of 1s per row <= number of labels
            invalid_rows = label_sums > len(label_columns)
            if invalid_rows.any():
                problematic_rows = df.index[invalid_rows].tolist()
                raise ValidationError(
                    f"Multi-label classification allows at most {len(label_columns)} labels per row. "
                    f"Problematic rows: {problematic_rows}"
                )

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
        # Set multi_label based on the classifier's label type
        label_type = getattr(self.config, 'label_type', 'single')
        self.prompt_engineer.multi_label = (label_type == "multiple")


    async def _generate_predictions(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> List[Union[str, List[str]]]:
        """Generate predictions in batches."""
        predictions = []
        for batch_df in self._get_batches(df, text_column):
            batch_predictions = await self._process_batch(batch_df, text_column)
            predictions.extend(batch_predictions)
        return predictions

    def _get_batches(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> Iterator[pd.DataFrame]:
        """Yield batches of DataFrame."""
        for i in range(0, len(df), self.batch_size):
            yield df.iloc[i:i + self.batch_size]

    async def _process_batch(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> List[Union[str, List[str]]]:
        """Process a batch of texts for prediction."""
        texts = batch_df[text_column].tolist()
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
        label_columns: List[str]
    ) -> Optional[Dict[str, float]]:
        """Evaluate model performance on test data."""
        if test_df is None:
            return None
            
        predictions = await self._generate_predictions(test_df, text_column)
        true_labels = test_df[label_columns].values.tolist()
        return self._calculate_metrics(predictions, true_labels)

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

    async def predict_async(self, texts: List[str]) -> List[str]:
        """Predict classifications for multiple texts."""
        # Get prompts from PromptEngineer
        prompts = self.prompt_engineer.create_prompts(
            texts=texts,
            is_multi_label=self.is_multi_label
        )
        
        # Make API calls
        responses = await asyncio.gather(
            *[self._call_llm(prompt.render()) for prompt in prompts],
            return_exceptions=True
        )
        
        # Parse responses
        return [
            self._parse_prediction_response(r) if not isinstance(r, Exception)
            else self._handle_error(r)
            for r in responses
        ]