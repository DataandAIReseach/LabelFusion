"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd
from tqdm import tqdm

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import PredictionError, ValidationError, APIError
from ..prompt_engineer.base import PromptEngineer
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager

class BaseLLMClassifier(AsyncBaseClassifier):
    """Base class for all LLM-based text classifiers."""
    
    def __init__(
        self, 
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot",
        verbose: bool = True
    ):
        """Initialize the LLM classifier.
        
        Args:
            config: Configuration object
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier (default: False)
            few_shot_mode: Mode for few-shot learning (default: "few_shot")
            verbose: Whether to show detailed progress (default: True)
        """
        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.multi_label = multi_label
        self.few_shot_mode = few_shot_mode
        self.verbose = verbose
        
        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set column specifications
        self.text_column = text_column
        self.label_columns = label_columns if label_columns else []
        self.classes_ = self.label_columns.copy()  # For compatibility with sklearn-style APIs
        
        if self.verbose:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            self.logger.info(f"Text column: {self.text_column}")
            self.logger.info(f"Label columns: {self.label_columns}")
            self.logger.info(f"Multi-label: {self.multi_label}")
            self.logger.info(f"Few-shot mode: {self.few_shot_mode}")
        
        # Initialize prompt engineer with configuration
        self.prompt_engineer = PromptEngineer(
            text_column=self.text_column,
            label_columns=self.label_columns,
            multi_label=self.multi_label,
            few_shot_mode=self.few_shot_mode,
            model_name=self.config.parameters["model"]  # Pass model from config.parameters
        )
        
        # Initialize LLM generator
        key_manager = APIKeyManager()
        api_key = key_manager.get_key("openai")  # Default to openai for now
        if not api_key:
            raise ValueError("No API key found for openai")
            
        self.llm_generator = create_llm_generator(
            provider="openai",
            model_name=self.config.parameters["model"],
            api_key=api_key
        )
        
        if self.verbose:
            self.logger.info(f"PromptEngineer initialized with model: {self.config.parameters['model']}")
            self.logger.info(f"LLM generator initialized with provider: openai")
        
        self._setup_config()

    def _setup_config(self) -> None:
        """Initialize configuration parameters."""
        self.batch_size = self.config.parameters.get('batch_size', 32)
        self.threshold = self.config.parameters.get('threshold', 0.5)
        
        if self.verbose:
            self.logger.info(f"Configuration setup - Batch size: {self.batch_size}, Threshold: {self.threshold}")

    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Synchronous wrapper for predictions."""
        return asyncio.run(self.predict_async(
            train_df=train_df,
            test_df=test_df,
            context=context,
            label_definitions=label_definitions
        ))

    async def predict_async(
        self,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Asynchronously predict labels for texts with detailed progress tracking."""
        
        start_time = time.time()
        
        try:
            if self.verbose:
                self.logger.info("="*80)
                self.logger.info("STARTING PREDICTION PROCESS")
                self.logger.info("="*80)
                self.logger.info(f"Test data shape: {test_df.shape}")
                if train_df is not None:
                    self.logger.info(f"Training data shape: {train_df.shape}")
                else:
                    self.logger.info("No training data provided")
            
            # Step 1: Data preparation and validation
            if self.verbose:
                self.logger.info("\nSTEP 1: Data Preparation and Validation")
                print("Validating input data...")
            
            # Stack DataFrames vertically if training data is provided
            df = pd.concat([train_df, test_df], axis=0) if train_df is not None else test_df
            
            if self.verbose:
                self.logger.info(f"Combined dataset shape: {df.shape}")
            
            # Validate input DataFrame
            self._validate_prediction_inputs(df, self.text_column, self.label_columns)
            
            if self.verbose:
                self.logger.info("Data validation completed successfully")
                print("Data validation passed")
            
            # Step 2: Prompt configuration setup
            if self.verbose:
                self.logger.info("\nSTEP 2: Prompt Configuration Setup")
                print("Setting up prompt configuration...")
            
            self._setup_prompt_configuration(context, label_definitions)
            
            if self.verbose:
                self.logger.info("Prompt configuration completed")
                print("Prompt configuration ready")
            
            # Step 3: Prompt engineering
            if self.verbose:
                self.logger.info("\nSTEP 3: Engineering Prompts")
                print("Engineering prompts for classification...")
            
            df_with_prompts = await self._engineer_prompts_for_data(
                train_df=train_df,
                test_df=test_df,
                text_column=self.text_column,
                label_columns=self.label_columns
            )
            
            if self.verbose:
                self.logger.info(f"Prompts engineered for {len(df_with_prompts)} samples")
                print(f"{len(df_with_prompts)} prompts ready for processing")
            
            # Step 4: Generate predictions
            if self.verbose:
                self.logger.info("\nSTEP 4: Generating Predictions")
                print("Generating predictions using LLM...")
            
            all_predictions = await self._generate_predictions(df_with_prompts, self.text_column)
            
            if self.verbose:
                self.logger.info(f"Generated {len(all_predictions)} predictions")
                print(f"{len(all_predictions)} predictions generated")
            
            # Step 5: Process results
            if self.verbose:
                self.logger.info("\nSTEP 5: Processing Results")
                print("Processing and organizing results...")
            
            # Split predictions back into train and test sets
            if train_df is not None:
                train_size = len(train_df)
                predictions = all_predictions[train_size:]  # Get only test predictions
                if self.verbose:
                    self.logger.info(f"Extracted {len(predictions)} test predictions from {len(all_predictions)} total")
            else:
                predictions = all_predictions
            
            # Step 6: Calculate metrics
            if self.verbose:
                self.logger.info("\nSTEP 6: Calculating Metrics")
                print("Calculating performance metrics...")
            
            metrics = await self._evaluate_test_data(
                test_df=test_df,
                text_column=self.text_column,
                label_columns=self.label_columns
            ) if test_df is not None else None
            
            if self.verbose and metrics:
                self.logger.info("Metrics calculated successfully")
                for metric_name, metric_value in metrics.items():
                    self.logger.info(f"  {metric_name}: {metric_value:.4f}")
                    print(f"{metric_name}: {metric_value:.4f}")
            
            # Final results
            end_time = time.time()
            total_time = end_time - start_time
            
            if self.verbose:
                self.logger.info("\nPREDICTION PROCESS COMPLETED")
                self.logger.info(f"Total processing time: {total_time:.2f} seconds")
                self.logger.info(f"Average time per sample: {total_time/len(test_df):.3f} seconds")
                self.logger.info("="*80)
                
                print(f"\nProcess completed in {total_time:.2f} seconds")
                print(f"Average: {total_time/len(test_df):.3f} seconds per sample")
            
            return self._create_result(predictions=predictions, metrics=metrics)
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"PREDICTION FAILED: {str(e)}")
                print(f"Error: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.parameters.get("model", "unknown"))

    def _validate_prediction_inputs(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        label_columns: List[str]
    ) -> None:
        """Validate prediction inputs with detailed logging."""
        
        if self.verbose:
            self.logger.info("Validating input data structure...")
            print("Validating DataFrame structure...")
        
        # Validate DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        if self.verbose:
            self.logger.info(f"DataFrame validation passed - Shape: {df.shape}")
            
        # Validate text column
        if self.verbose:
            self.logger.info(f"Validating text column: '{text_column}'")
            
        if text_column not in df.columns:
            raise ValidationError(f"Text column '{text_column}' not found in DataFrame")
        if not df[text_column].dtype == 'object':
            raise ValidationError(f"Text column '{text_column}' must be of type string/object")
        if not df[text_column].apply(lambda x: isinstance(x, str)).all():
            raise ValidationError(f"All entries in text column '{text_column}' must be strings")
        
        if self.verbose:
            self.logger.info(f"Text column validation passed")
            
        # Validate label columns
        if self.verbose:
            self.logger.info(f"Validating {len(label_columns)} label columns...")
            
        for i, label_col in enumerate(label_columns):
            if self.verbose:
                self.logger.debug(f"Validating label column {i+1}/{len(label_columns)}: '{label_col}'")
                
            if label_col not in df.columns:
                raise ValidationError(f"Label column '{label_col}' not found in DataFrame")
            if not pd.api.types.is_bool_dtype(df[label_col]) and not pd.api.types.is_numeric_dtype(df[label_col]):
                raise ValidationError(f"Label column '{label_col}' must contain only boolean or numeric values")
            if not df[label_col].isin([0, 1, True, False]).all():
                raise ValidationError(f"Label column '{label_col}' must contain only binary values (0, 1, True, False)")
        
        if self.verbose:
            self.logger.info(f"All {len(label_columns)} label columns validated")
        
        # Validate label consistency
        if self.verbose:
            self.logger.info("Validating label consistency for classification type...")
            
        label_sums = df[label_columns].sum(axis=1)
        
        if not self.multi_label:
            invalid_rows = label_sums != 1
            if invalid_rows.any():
                problematic_rows = df.index[invalid_rows].tolist()
                raise ValidationError(
                    f"Single-label classification requires exactly one 1/True per row. "
                    f"Problematic rows: {problematic_rows}"
                )
            if self.verbose:
                self.logger.info("Single-label consistency validated")
        else:
            invalid_rows = label_sums > len(label_columns)
            if invalid_rows.any():
                problematic_rows = df.index[invalid_rows].tolist()
                raise ValidationError(
                    f"Multi-label classification allows at most {len(label_columns)} labels per row. "
                    f"Problematic rows: {problematic_rows}"
                )
            if self.verbose:
                self.logger.info("Multi-label consistency validated")

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
        """Generate predictions in batches with progress tracking."""
        predictions = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        if self.verbose:
            self.logger.info(f"Processing {len(df)} samples in {total_batches} batches of size {self.batch_size}")
            print(f"Processing in {total_batches} batches...")
        
        # Use tqdm for progress bar if verbose mode is enabled
        batch_iterator = range(0, len(df), self.batch_size)
        if self.verbose:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", total=total_batches)
        
        for i, batch_start in enumerate(batch_iterator):
            batch_df = df.iloc[batch_start:batch_start + self.batch_size]
            
            if self.verbose and not isinstance(batch_iterator, tqdm):
                self.logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch_df)} samples)")
                print(f"Batch {i+1}/{total_batches} processing...")
            
            batch_predictions = await self._process_batch(batch_df, text_column)
            predictions.extend(batch_predictions)
            
            if self.verbose and not isinstance(batch_iterator, tqdm):
                self.logger.info(f"Batch {i+1} completed ({len(batch_predictions)} predictions)")
        
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
        """Process a batch of texts for prediction with detailed logging."""
        if self.verbose:
            self.logger.debug(f"Processing batch with {len(batch_df)} texts")
        
        texts = batch_df[text_column].tolist()
        
        # Check if prompts are already engineered
        if 'engineered_prompt' in batch_df.columns:
            prompts = batch_df['engineered_prompt'].tolist()
        else:
            prompts = [self.prompt_engineer.build_prompt(text) for text in texts]
        
        if self.verbose:
            self.logger.debug(f"Generated {len(prompts)} prompts for LLM calls")

        responses = await asyncio.gather(
            *[self._call_llm(prompt) for prompt in prompts],
            return_exceptions=True
        )
        
        successful_responses = sum(1 for r in responses if not isinstance(r, Exception))
        failed_responses = len(responses) - successful_responses
        
        if self.verbose and failed_responses > 0:
            self.logger.warning(f"WARNING: {failed_responses} out of {len(responses)} LLM calls failed")
        
        return [
            self._parse_prediction_response(r) if not isinstance(r, Exception)
            else self._handle_error(r)
            for r in responses
        ]

    def _parse_prediction_response(self, response: str) -> Union[str, List[str]]:
        """Parse the LLM response for predictions."""
        if not response:
            if self.verbose:
                self.logger.warning("Received empty response from LLM")
            # Return default values for empty responses
            if self.multi_label:
                return []
            else:
                return self.classes_[0] if self.classes_ else "unknown"
        
        response = response.strip()
        if not response:
            if self.verbose:
                self.logger.warning("Received whitespace-only response from LLM")
            # Return default values for empty responses
            if self.multi_label:
                return []
            else:
                return self.classes_[0] if self.classes_ else "unknown"
        
        if self.multi_label:
            return self._parse_multiple_labels(response)
        else:
            return self._parse_single_label(response)

    def _parse_single_label(self, response: str) -> str:
        """Parse response for single-label classification."""
        if not response:
            if self.verbose:
                self.logger.warning("Empty response in single-label parsing")
            return self.classes_[0] if self.classes_ else "unknown"
        
        response_lower = response.lower()
        
        # Try to find exact matches first
        for class_name in self.classes_:
            if class_name.lower() == response_lower:
                return class_name
        
        # Try to find partial matches
        for class_name in self.classes_:
            if class_name.lower() in response_lower:
                return class_name
        
        # Check for common response patterns
        if any(word in response_lower for word in ['unknown', 'unclear', 'cannot', "can't", 'unable']):
            if self.verbose:
                self.logger.warning(f"LLM indicated uncertainty in response: {response}")
        
        # Default fallback
        if self.verbose:
            self.logger.warning(f"No class found in response: '{response}', using default: {self.classes_[0] if self.classes_ else 'unknown'}")
        
        return self.classes_[0] if self.classes_ else "unknown"

    def _parse_multiple_labels(self, response: str) -> List[str]:
        """Parse response for multi-label classification."""
        if not response:
            if self.verbose:
                self.logger.warning("Empty response in multi-label parsing")
            return []
        
        response_upper = response.upper()
        
        # Handle explicit "NONE" responses
        if response_upper in ["NONE", "EMPTY", "NULL", "NO LABELS", "NOTHING"]:
            return []
        
        # Handle uncertainty responses
        if any(word in response_upper for word in ['UNKNOWN', 'UNCLEAR', 'CANNOT', "CAN'T", 'UNABLE']):
            if self.verbose:
                self.logger.warning(f"LLM indicated uncertainty in response: {response}")
            return []
        
        predicted_classes = []
        
        # Split by common separators
        parts = []
        for separator in [',', ';', '|', '\n', 'and']:
            if separator in response:
                parts = response.split(separator)
                break
        
        if not parts:
            parts = [response]  # Treat as single item
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Try exact matches first
            for class_name in self.classes_:
                if class_name.lower() == part.lower():
                    if class_name not in predicted_classes:
                        predicted_classes.append(class_name)
                    break
            else:
                # Try partial matches
                for class_name in self.classes_:
                    if class_name.lower() in part.lower():
                        if class_name not in predicted_classes:
                            predicted_classes.append(class_name)
                        break
        
        if not predicted_classes and self.verbose:
            self.logger.warning(f"No classes found in multi-label response: '{response}'")
        
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
        """Handle prediction errors by returning default values instead of raising exceptions."""
        if self.verbose:
            self.logger.error(f"LLM call failed: {str(error)}")
        
        # Return default values instead of raising exceptions to allow processing to continue
        if self.multi_label:
            return []
        else:
            return self.classes_[0] if self.classes_ else "unknown"

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with the given prompt."""
        raise NotImplementedError("Subclasses must implement _call_llm method")

    async def _engineer_prompts_for_data(
        self,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        text_column: Optional[str] = None,
        label_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Engineer prompts for all texts in DataFrame with progress tracking."""
        
        if self.verbose:
            self.logger.info("Starting prompt engineering process...")
            print("Starting prompt engineering...")
        
        # Use instance variables if not provided
        text_column = text_column or self.text_column
        label_columns = label_columns or self.label_columns
        
        # Create copy of test data to avoid modifying original
        df_with_prompts = test_df.copy()

        if self.verbose:
            self.logger.info(f"Engineering prompts for {len(test_df)} test samples")
            if train_df is not None:
                self.logger.info(f"Using {len(train_df)} training samples for few-shot learning")

        # Engineer prompts using PromptEngineer
        engineered_prompts = await self.prompt_engineer.engineer_prompts(
            test_df=test_df,
            train_df=train_df,
            sample_size=self.batch_size
        )

        if self.verbose:
            self.logger.info(f"Successfully engineered {len(engineered_prompts)} prompts")
            print(f"{len(engineered_prompts)} prompts engineered")

        # Convert prompts to strings and add as new column
        if self.verbose:
            print("Rendering prompts...")
            
        rendered_prompts = []
        for i, prompt in enumerate(engineered_prompts):
            if self.verbose and i % 10 == 0:  # Log every 10th prompt
                self.logger.debug(f"Rendering prompt {i+1}/{len(engineered_prompts)}")
            rendered_prompts.append(prompt.render())
        
        df_with_prompts['engineered_prompt'] = rendered_prompts

        if self.verbose:
            self.logger.info("All prompts rendered and added to DataFrame")
            print("Prompt engineering completed")

        return df_with_prompts

    def _create_result(
        self,
        predictions: List[Union[str, List[str]]],
        metrics: Optional[Dict[str, float]] = None
    ) -> ClassificationResult:
        """Create a ClassificationResult object."""
        return ClassificationResult(
            predictions=predictions,
            metrics=metrics or {},
            model_name=self.config.parameters.get("model", "unknown"),
            classification_type=ClassificationType.MULTI_LABEL if self.multi_label else ClassificationType.SINGLE_LABEL
        )
