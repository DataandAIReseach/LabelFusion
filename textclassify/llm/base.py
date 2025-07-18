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
                
            # Validate input DataFrame
            self._validate_prediction_inputs(df, text_column, label_columns)
            
            self._setup_prompt_configuration(context, label_definitions)
            
            # Engineer prompts and add to DataFrame
            df_with_prompts = await self._engineer_prompts_for_data(
                df=df,
                text_column=text_column,
                label_columns=label_columns
            )
            
            # Generate predictions using engineered prompts
            predictions = await self._generate_predictions(df_with_prompts, text_column)
            
            # Evaluate if test data provided
            metrics = None
            if test_df is not None:
                test_df_with_prompts = await self._engineer_prompts_for_data(
                    df=test_df,
                    text_column=text_column,
                    label_columns=label_columns
                )
                metrics = await self._evaluate_test_data(
                    test_df=test_df_with_prompts,
                    text_column=text_column,
                    label_columns=label_columns
                )
            
            return self._create_result(predictions=predictions, metrics=metrics)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.model_name)

    def _validate_prediction_inputs(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        label_columns: List[str]
    ) -> None:
        """Validate prediction inputs."""
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
        
        if not self.multi_label:  # Use class attribute
            invalid_rows = label_sums != 1
            if invalid_rows.any():
                problematic_rows = df.index[invalid_rows].tolist()
                raise ValidationError(
                    f"Single-label classification requires exactly one 1 per row. "
                    f"Problematic rows: {problematic_rows}"
                )
        else:
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
        label_definitions: Optional[Dict[str, str]]
    ) -> None:
        """Configure the prompt engineer.
    
        Args:
        context: Optional context string to use
        label_definitions: Optional dict mapping labels to definitions
        """
        if context is not None:
            self.prompt_engineer.context = context
        if label_definitions is not None:
            self.prompt_engineer.label_definitions = label_definitions
        # few_shot_mode and multi_label are already set in constructor
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

    async def _engineer_prompts_for_data(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_columns: List[str]
    ) -> pd.DataFrame:
        """Engineer prompts for all texts in DataFrame using PromptEngineer.
    
        Args:
            df: DataFrame containing texts to classify
            text_column: Name of column containing text data
            label_columns: Names of columns containing labels
        
        Returns:
            DataFrame with engineered prompts added as new column
        """
        # Create copy to avoid modifying original
        df_with_prompts = df.copy()
    
        # Engineer prompts using PromptEngineer
        engineered_prompts = await self.prompt_engineer.engineer_prompts(
            data=df,
            sample_size=self.batch_size
        )
    
        # Convert prompts to strings and add as new column
        df_with_prompts['engineered_prompt'] = [
            prompt.render() for prompt in engineered_prompts
        ]
    
        return df_with_prompts

    def fill_procedure_prompt_creator_prompt(
        self,
        train_df: pd.DataFrame,
        sample_size: int = 20,
        custom_prompt: Optional[str] = None,
        custom_role_prompt: Optional[str] = None,
        include_role: bool = True,
        context_content: Optional[str] = None
    ) -> str:
        """Fill a procedure prompt creator prompt using sampled data.
    
        Args:
            train_df: DataFrame containing training examples
            sample_size: Number of examples to use for procedure creation
            custom_prompt: Optional custom prompt template
            custom_role_prompt: Optional custom role prompt
            include_role: Whether to include the role prompt
            context_content: Optional context to include in the prompt
    
        Returns:
            str: Procedure prompt creator content
    
        Raises:
            ValueError: If train_df is None
        """
        if train_df is None:
            raise ValueError("No data available for procedure prompt creation")
    
        # Sample data
        sampled_data = train_df.sample(n=min(sample_size, len(train_df)))
        examples = []
        for _, row in sampled_data.iterrows():
            text = row[self.text_column]
            labels = {col: row[col] for col in self.label_columns}
            examples.append({'text': text, 'labels': labels})
    
        # Format prompt with examples
        formatted_examples = "\n".join([
            f"Text: {ex['text']}\nLabels: {ex['labels']}"
            for ex in examples
        ])
    
        # Select prompt template
        if custom_prompt:
            prompt_template = custom_prompt
            # Only apply context formatting if template expects it
            if context_content and '{context}' in custom_prompt:
                prompt_template = prompt_template.format(context=context_content)
        else:
            prompt_template = PromptWarehouse.procedure_prompt_creator_prompt
    
        # Format with available data
        try:
            prompt_text = prompt_template.format(
                data=formatted_examples,
                features=", ".join(self.label_columns)
            )
        except KeyError as e:
            raise ValueError(f"Prompt template contains unknown placeholder: {e}")
    
        # Add role prompt if requested
        if include_role and custom_role_prompt:
            return f"{custom_role_prompt}\n\n{prompt_text}"
        elif include_role and self.role_prompt:
            return f"{self.role_prompt}\n\n{prompt_text}"
        else:
            return prompt_text