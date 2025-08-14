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
        verbose: bool = True,
        provider: Optional[str] = None
    ):
        """Initialize the LLM classifier.
        
        Args:
            config: Configuration object
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier (default: False)
            few_shot_mode: Mode for few-shot learning (default: "few_shot")
            verbose: Whether to show detailed progress (default: True)
            provider: LLM provider to use ('openai', 'gemini', 'deepseek', etc.)
        """
        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.multi_label = multi_label
        self.few_shot_mode = few_shot_mode
        self.verbose = verbose
        
        # Set provider - use parameter if provided, otherwise get from config, default to openai
        self.provider = provider or getattr(self.config, 'provider', 'openai')
        # Also set it on config for consistency
        self.config.provider = self.provider
        
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
            model_name=self.config.parameters["model"],  # Pass model from config.parameters
            provider=self.provider  # Pass provider for correct instance handling
        )
        
        # Initialize LLM generator
        key_manager = APIKeyManager()
        
        # Get appropriate API key based on provider
        if self.provider == 'gemini':
            api_key = key_manager.get_key("gemini") or key_manager.get_key("google")
            if not api_key:
                raise ValueError("No API key found for gemini. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        elif self.provider == 'deepseek':
            api_key = key_manager.get_key("deepseek")
            if not api_key:
                raise ValueError("No API key found for deepseek")
        else:  # default to openai
            api_key = key_manager.get_key("openai")
            if not api_key:
                raise ValueError("No API key found for openai")
            
        self.llm_generator = create_llm_generator(
            provider=self.provider,
            model_name=self.config.parameters["model"],
            api_key=api_key
        )
        
        if self.verbose:
            self.logger.info(f"PromptEngineer initialized with model: {self.config.parameters['model']}")
            self.logger.info(f"LLM generator initialized with provider: {self.provider}")
            self.logger.info(f"Using API key for: {self.provider}")
        
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
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Synchronous wrapper for predictions."""
        # Handle texts parameter by converting to DataFrame
        if texts is not None and test_df is None:
            test_df = pd.DataFrame({'text': texts})
        
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
            
            # Process DataFrames through data preparation pipeline
            prepared_train_df, prepared_test_df = self._prepare_dataframe_for_prediction(
                train_df, test_df, self.text_column, self.label_columns
            )
            
            if self.verbose:
                self.logger.info("Data preparation and validation completed successfully")
                print("Data preparation and validation passed")
            
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
                train_df=prepared_train_df,
                test_df=prepared_test_df,
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
            
            predictions = await self._generate_predictions(df_with_prompts, self.text_column)
            
            if self.verbose:
                self.logger.info(f"Generated {len(predictions)} predictions")
                print(f"{len(predictions)} predictions generated")
            
            # Step 5: Process results
            if self.verbose:
                self.logger.info("\nSTEP 5: Processing Results")
                print("Processing and organizing results...")
            
            # Step 6: Calculate metrics
            if self.verbose:
                self.logger.info("\nSTEP 6: Calculating Metrics")
                print("Calculating performance metrics...")
            
            metrics = self._evaluate_test_data(
                test_df=prepared_test_df,
                predictions=predictions,
                label_columns=self.label_columns
            ) if prepared_test_df is not None else None
            
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
                self.logger.info(f"Average time per sample: {total_time/len(prepared_test_df):.3f} seconds")
                self.logger.info("="*80)
                
                print(f"\nProcess completed in {total_time:.2f} seconds")
                print(f"Average: {total_time/len(prepared_test_df):.3f} seconds per sample")
            
            return self._create_result(predictions=predictions, metrics=metrics)
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"PREDICTION FAILED: {str(e)}")
                print(f"Error: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.parameters.get("model", "unknown"))
    
    def _prepare_dataframe_for_prediction(self, train_df: Optional[pd.DataFrame], 
                                        test_df: pd.DataFrame, text_column: str, 
                                        label_columns: List[str]) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Prepare DataFrames for prediction through a complete data pipeline.
        
        This method consolidates all data preparation steps for both train and test data:
        1. Format conversion (list labels to columns)
        2. Column renaming (class_X to actual names)
        3. Validation
        
        Args:
            train_df: Optional training DataFrame
            test_df: Test DataFrame
            text_column: Name of the text column
            label_columns: Expected label column names
            
        Returns:
            Tuple of (prepared_train_df, prepared_test_df)
            
        Raises:
            ValidationError: If data doesn't meet requirements
        """
        if self.verbose:
            self.logger.info("Starting DataFrame preparation pipeline...")
        
        prepared_train_df = None
        
        # Process training DataFrame if provided
        if train_df is not None:
            if self.verbose:
                self.logger.info(f"Processing training data: {train_df.shape}")
            
            # Step 1: Auto-convert DataFrame format if needed
            train_converted = self._auto_convert_dataframe_format(train_df, text_column, label_columns)
            
            # Step 2: Rename columns if needed after conversion
            train_renamed = self._rename_converted_columns(train_converted, label_columns)
            
            # Step 3: Clean up - remove the original 'label' column if it exists
            if 'label' in train_renamed.columns:
                train_renamed = train_renamed.drop('label', axis=1)
                if self.verbose:
                    self.logger.info("Removed original 'label' column from training data after conversion")
            
            # Step 4: Validate the training data
            self._validate_prediction_inputs(train_renamed, text_column, label_columns)
            prepared_train_df = train_renamed
        
        # Process test DataFrame
        if self.verbose:
            self.logger.info(f"Processing test data: {test_df.shape}")
        
        # Step 1: Auto-convert DataFrame format if needed
        test_converted = self._auto_convert_dataframe_format(test_df, text_column, label_columns)
        
        # Step 2: Rename columns if needed after conversion
        test_renamed = self._rename_converted_columns(test_converted, label_columns)
        
        # Step 3: Clean up - remove the original 'label' column if it exists
        if 'label' in test_renamed.columns:
            test_renamed = test_renamed.drop('label', axis=1)
            if self.verbose:
                self.logger.info("Removed original 'label' column from test data after conversion")
        
        # Step 4: Validate the test data
        self._validate_prediction_inputs(test_renamed, text_column, label_columns)
        prepared_test_df = test_renamed
        
        if self.verbose:
            self.logger.info("DataFrame preparation pipeline completed successfully")
            if prepared_train_df is not None:
                self.logger.info(f"Prepared training data: {prepared_train_df.shape}")
            self.logger.info(f"Prepared test data: {prepared_test_df.shape}")
        
        return prepared_train_df, prepared_test_df

    def convert_list_labels_to_columns(self, df: pd.DataFrame, label_column: str = 'label') -> pd.DataFrame:
        """Convert list-based labels to separate binary columns.
        
        Args:
            df: DataFrame with list-based labels
            label_column: Name of the column containing list labels
            
        Returns:
            DataFrame with separate binary columns for each class
        """
        if label_column not in df.columns:
            return df
        
        # Check if labels are already in column format
        sample_label = df[label_column].iloc[0] if len(df) > 0 else None
        if not isinstance(sample_label, list):
            return df  # Already in column format or not list-based
        
        # Create binary columns for each class
        df_converted = df.copy()
        
        # Get all unique classes from the list labels
        all_classes = set()
        for label_list in df[label_column]:
            if isinstance(label_list, list):
                for idx, value in enumerate(label_list):
                    if value == 1:  # Binary encoding: 1 means class is active
                        all_classes.add(f"class_{idx}")
        
        # Create binary columns
        for class_name in sorted(all_classes):
            class_idx = int(class_name.split('_')[1])
            df_converted[class_name] = df[label_column].apply(
                lambda x: x[class_idx] if isinstance(x, list) and len(x) > class_idx else 0
            )
        
        return df_converted

    def _auto_convert_dataframe_format(self, df: pd.DataFrame, text_column: str, 
                                     label_columns: List[str]) -> pd.DataFrame:
        """Automatically detect and convert DataFrame format if needed."""
        # Check if we have a 'label' column with list-based labels
        if 'label' in df.columns and len(df) > 0:
            sample_label = df['label'].iloc[0]
            if isinstance(sample_label, list):
                print("Detected list-based labels, converting to column format...")
                return self.convert_list_labels_to_columns(df, 'label')
        
        return df

    def _rename_converted_columns(self, df: pd.DataFrame, label_columns: List[str]) -> pd.DataFrame:
        """Rename converted class_X columns to actual label column names if needed.
        
        Args:
            df: DataFrame potentially containing class_X columns
            label_columns: Expected label column names
            
        Returns:
            DataFrame with properly named columns
        """
        # Handle column renaming if we have class_X columns but need specific label names
        class_columns = [col for col in df.columns if col.startswith('class_')]
        if class_columns and label_columns and len(class_columns) == len(label_columns):
            # Check if we need to rename class_X columns to actual label names
            missing_labels = [col for col in label_columns if col not in df.columns]
            if missing_labels and len(missing_labels) == len(label_columns):
                if self.verbose:
                    self.logger.info(f"Renaming class columns to label names: {class_columns} -> {label_columns}")
                    print("Renaming converted columns to match expected labels...")
                
                # Sort class columns to ensure proper mapping
                sorted_class_cols = sorted(class_columns, key=lambda x: int(x.split('_')[1]))
                rename_mapping = {class_col: label_col for class_col, label_col in zip(sorted_class_cols, label_columns)}
                df = df.rename(columns=rename_mapping)
                
                if self.verbose:
                    self.logger.info("Column renaming completed")
        
        return df

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
            if not pd.api.types.is_numeric_dtype(df[label_col]):
                raise ValidationError(f"Label column '{label_col}' must contain only integer values")
            if not df[label_col].isin([0, 1]).all():
                raise ValidationError(f"Label column '{label_col}' must contain only binary integer values (0, 1)")
        
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
                    f"Single-label classification requires exactly one 1 per row. "
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
    ) -> List[List[int]]:
        """Generate predictions in batches with progress tracking.
        
        Returns:
            List[List[int]]: List of binary vectors representing predictions
        """
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
    ) -> List[List[int]]:
        """Process a batch of texts for prediction with detailed logging.
        
        Returns:
            List[List[int]]: List of binary vectors representing predictions
        """
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

    def _parse_prediction_response(self, response: str) -> List[int]:
        """Parse the LLM response for predictions.
        
        Returns:
            List[int]: Binary vector representation of predictions
        """
        if not response:
            if self.verbose:
                self.logger.warning("Received empty response from LLM")
            # Return default values for empty responses
            if self.multi_label:
                return [0] * len(self.classes_) if self.classes_ else [0]
            else:
                return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]
        
        response = response.strip()
        if not response:
            if self.verbose:
                self.logger.warning("Received whitespace-only response from LLM")
            # Return default values for empty responses
            if self.multi_label:
                return [0] * len(self.classes_) if self.classes_ else [0]
            else:
                return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]
        
        if self.multi_label:
            return self._parse_multiple_labels(response)
        else:
            return self._parse_single_label(response)

    def _parse_single_label(self, response: str) -> List[int]:
        """Parse response for single-label classification.
        
        Returns:
            List[int]: Binary vector with exactly one 1 and rest 0s
        """
        if not response:
            if self.verbose:
                self.logger.warning("Empty response in single-label parsing")
            # Return first class as default (1 at index 0, rest 0)
            return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]
        
        response = response.strip()
        
        # Check if response is in binary format like '1 | 0 | 0 | 0'
        if '|' in response:
            binary_parts = [part.strip() for part in response.split('|')]
            # Check if this looks like a binary format (all parts are '0' or '1')
            if (len(binary_parts) == len(self.classes_) and 
                all(part in ['0', '1'] for part in binary_parts)):
                # Convert to integer list
                binary_vector = [int(part) for part in binary_parts]
                # Ensure exactly one '1' for single-label
                if sum(binary_vector) == 1:
                    return binary_vector
                elif sum(binary_vector) == 0:
                    if self.verbose:
                        self.logger.warning(f"No '1' found in binary response: '{response}', using default")
                    return [1] + [0] * (len(self.classes_) - 1)
                else:
                    if self.verbose:
                        self.logger.warning(f"Multiple '1's found in single-label response: '{response}', using first")
                    # Keep only the first '1'
                    first_one_idx = binary_vector.index(1)
                    result = [0] * len(binary_vector)
                    result[first_one_idx] = 1
                    return result
            elif self.verbose and len(binary_parts) == len(self.classes_):
                self.logger.warning(f"Response length matches classes but contains non-binary values: {binary_parts}")
            elif self.verbose:
                self.logger.warning(f"Response length ({len(binary_parts)}) doesn't match classes count ({len(self.classes_)})")
        
        # Fallback: try to match class names and convert to binary vector
        response_lower = response.lower()
        
        # Try to find exact matches first
        for i, class_name in enumerate(self.classes_):
            if class_name.lower() == response_lower:
                result = [0] * len(self.classes_)
                result[i] = 1
                return result
        
        # Try to find partial matches
        for i, class_name in enumerate(self.classes_):
            if class_name.lower() in response_lower:
                result = [0] * len(self.classes_)
                result[i] = 1
                return result
        
        # Check for common response patterns
        if any(word in response_lower for word in ['unknown', 'unclear', 'cannot', "can't", 'unable']):
            if self.verbose:
                self.logger.warning(f"LLM indicated uncertainty in response: {response}")
        
        # Default fallback
        if self.verbose:
            self.logger.warning(f"No class found in response: '{response}', using default (first class)")
        
        return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]

    def _parse_multiple_labels(self, response: str) -> List[int]:
        """Parse response for multi-label classification.
        
        Returns:
            List[int]: Binary vector where 1s indicate predicted classes
        """
        if not response:
            if self.verbose:
                self.logger.warning("Empty response in multi-label parsing")
            return [0] * len(self.classes_) if self.classes_ else [0]
        
        response = response.strip()
        response_upper = response.upper()
        
        # Handle explicit "NONE" responses
        if response_upper in ["NONE", "EMPTY", "NULL", "NO LABELS", "NOTHING"]:
            return [0] * len(self.classes_)
        
        # Handle uncertainty responses
        if any(word in response_upper for word in ['UNKNOWN', 'UNCLEAR', 'CANNOT', "CAN'T", 'UNABLE']):
            if self.verbose:
                self.logger.warning(f"LLM indicated uncertainty in response: {response}")
            return [0] * len(self.classes_)
        
        # Check if response is in binary format like '1 | 0 | 1 | 0'
        if '|' in response:
            binary_parts = [part.strip() for part in response.split('|')]
            # Check if this looks like a binary format (all parts are '0' or '1')
            if (len(binary_parts) == len(self.classes_) and 
                all(part in ['0', '1'] for part in binary_parts)):
                # Convert to integer list and return
                return [int(part) for part in binary_parts]
            else:
                if self.verbose and len(binary_parts) == len(self.classes_):
                    self.logger.warning(f"Binary response length matches but contains non-binary values: {binary_parts}")
                elif self.verbose:
                    self.logger.warning(f"Binary response length ({len(binary_parts)}) doesn't match classes count ({len(self.classes_)})")
        
        # Fallback: parse text-based responses and convert to binary vector
        predicted_classes = []
        
        # Split by common separators for text-based responses (excluding '|' since we handled it above)
        parts = []
        for separator in [',', ';', '\n', 'and']:
            if separator in response:
                parts = response.split(separator)
                break
        
        # If no separators found but contains '|', treat as text with '|' separator
        if not parts and '|' in response:
            parts = response.split('|')
        
        if not parts:
            parts = [response]  # Treat as single item
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Try exact matches first
            for i, class_name in enumerate(self.classes_):
                if class_name.lower() == part.lower():
                    if i not in predicted_classes:
                        predicted_classes.append(i)
                    break
            else:
                # Try partial matches
                for i, class_name in enumerate(self.classes_):
                    if class_name.lower() in part.lower():
                        if i not in predicted_classes:
                            predicted_classes.append(i)
                        break
        
        # Convert predicted class indices to binary vector
        binary_vector = [0] * len(self.classes_)
        for idx in predicted_classes:
            binary_vector[idx] = 1
        
        if not predicted_classes and self.verbose:
            self.logger.warning(f"No classes found in multi-label response: '{response}'")
        
        return binary_vector

    def _evaluate_test_data(
        self,
        test_df: pd.DataFrame,
        predictions: List[List[int]],
        label_columns: List[str]
    ) -> Optional[Dict[str, float]]:
        """Evaluate model performance on test data using provided predictions."""
        if test_df is None or not predictions:
            return None
            
        # Convert true labels to binary vector format (same as predictions)
        true_labels = test_df[label_columns].values.astype(int).tolist()
        return self._calculate_metrics(predictions, true_labels)

    def _calculate_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for binary vector predictions."""
        if not self.multi_label:
            return self._calculate_single_label_metrics(predictions, true_labels)
        return self._calculate_multi_label_metrics(predictions, true_labels)

    def _calculate_single_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for single-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
        # Convert binary vectors to class indices for sklearn compatibility
        pred_classes = [pred.index(1) if 1 in pred else 0 for pred in predictions]
        true_classes = [true.index(1) if 1 in true else 0 for true in true_labels]
        
        # Calculate basic accuracy
        correct = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == true)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # For single-label classification with more than 2 classes, use macro averaging
        num_classes = len(self.classes_) if self.classes_ else max(max(pred_classes, default=0), max(true_classes, default=0)) + 1
        
        if num_classes <= 2:
            # Binary classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            try:
                precision = precision_score(true_classes, pred_classes, average='binary', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='binary', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='binary', zero_division=0)
                
                # For AUC, we need probability scores, but we only have binary predictions
                # Use the prediction confidence as a proxy (1.0 for predicted class, 0.0 for others)
                try:
                    auc = roc_auc_score(true_classes, pred_classes)
                except ValueError:
                    # If all predictions are the same class, AUC is undefined
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        else:
            # Multi-class classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            try:
                precision = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
                recall = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
                f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
                
                # For multi-class AUC, convert to one-hot and use ovr strategy
                try:
                    from sklearn.preprocessing import label_binarize
                    true_binary = label_binarize(true_classes, classes=list(range(num_classes)))
                    pred_binary = label_binarize(pred_classes, classes=list(range(num_classes)))
                    auc = roc_auc_score(true_binary, pred_binary, average='macro', multi_class='ovr')
                except (ValueError, ImportError):
                    auc = 0.5
            except ImportError:
                # Fallback if sklearn is not available
                precision, recall, f1, auc = self._calculate_metrics_manual(pred_classes, true_classes, num_classes)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _calculate_metrics_manual(self, pred_classes: List[int], true_classes: List[int], num_classes: int) -> tuple:
        """Manual calculation of metrics when sklearn is not available."""
        # Calculate per-class metrics
        class_metrics = []
        
        for class_idx in range(num_classes):
            # True positives, false positives, false negatives for this class
            tp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true == class_idx)
            fp = sum(1 for pred, true in zip(pred_classes, true_classes) if pred == class_idx and true != class_idx)
            fn = sum(1 for pred, true in zip(pred_classes, true_classes) if pred != class_idx and true == class_idx)
            
            # Calculate precision, recall, f1 for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics.append((precision, recall, f1))
        
        # Macro average
        if class_metrics:
            avg_precision = sum(m[0] for m in class_metrics) / len(class_metrics)
            avg_recall = sum(m[1] for m in class_metrics) / len(class_metrics)
            avg_f1 = sum(m[2] for m in class_metrics) / len(class_metrics)
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
        
        # Simple AUC approximation (not perfect but better than nothing)
        auc = 0.5  # Default for when we can't calculate properly
        
        return avg_precision, avg_recall, avg_f1, auc

    def _calculate_multi_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate metrics for multi-label classification using binary vectors."""
        if not predictions or not true_labels:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}
        
        # Sample-wise metrics
        sample_precisions = []
        sample_recalls = []
        sample_f1s = []
        exact_matches = 0
        hamming_distance = 0
        total_predictions = 0
        
        for pred, true in zip(predictions, true_labels):
            pred_set = set(i for i, val in enumerate(pred) if val == 1)
            true_set = set(i for i, val in enumerate(true) if val == 1)
            
            # Sample-wise precision, recall, F1
            if pred_set:
                precision = len(pred_set & true_set) / len(pred_set)
            else:
                precision = 1.0 if not true_set else 0.0
            
            if true_set:
                recall = len(pred_set & true_set) / len(true_set)
            else:
                recall = 1.0 if not pred_set else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            sample_precisions.append(precision)
            sample_recalls.append(recall)
            sample_f1s.append(f1)
            
            # Exact match (subset accuracy)
            if pred_set == true_set:
                exact_matches += 1
            
            # Hamming loss components
            for i in range(len(pred)):
                total_predictions += 1
                if pred[i] != true[i]:
                    hamming_distance += 1
        
        # Calculate averages
        avg_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
        avg_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
        avg_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
        
        # Subset accuracy (exact match ratio)
        subset_accuracy = exact_matches / len(predictions) if predictions else 0.0
        
        # Hamming loss
        hamming_loss = hamming_distance / total_predictions if total_predictions > 0 else 0.0
        
        # Label-wise metrics (micro-averaged)
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Flatten for micro-averaging
            y_true_flat = [label for true in true_labels for label in true]
            y_pred_flat = [label for pred in predictions for label in pred]
            
            micro_precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            micro_f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            
        except ImportError:
            # Fallback when sklearn is not available
            micro_precision = avg_precision
            micro_recall = avg_recall
            micro_f1 = avg_f1
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming_loss,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1
        }

    def _handle_error(self, error: Exception) -> List[int]:
        """Handle prediction errors by returning default binary vectors instead of raising exceptions."""
        if self.verbose:
            self.logger.error(f"LLM call failed: {str(error)}")
        
        # Return default binary vectors instead of raising exceptions to allow processing to continue
        if self.multi_label:
            return [0] * len(self.classes_) if self.classes_ else [0]
        else:
            return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]

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
        predictions: List[List[int]],
        metrics: Optional[Dict[str, float]] = None
    ) -> ClassificationResult:
        """Create a ClassificationResult object with binary vector predictions."""
        # Convert binary vectors to string predictions for compatibility with ClassificationResult
        if self.multi_label:
            # For multi-label: convert binary vectors to lists of active class names
            string_predictions = []
            for pred in predictions:
                active_classes = [self.classes_[i] for i, val in enumerate(pred) if val == 1 and i < len(self.classes_)]
                string_predictions.append(active_classes if active_classes else [])
        else:
            # For single-label: convert binary vectors to single class names
            string_predictions = []
            for pred in predictions:
                if 1 in pred:
                    class_idx = pred.index(1)
                    if class_idx < len(self.classes_):
                        string_predictions.append(self.classes_[class_idx])
                    else:
                        string_predictions.append(f"class_{class_idx}")
                else:
                    string_predictions.append(self.classes_[0] if self.classes_ else "unknown")
        
        # Create metadata with metrics
        metadata = {"metrics": metrics or {}}
        
        return ClassificationResult(
            predictions=string_predictions,
            model_name=self.config.parameters.get("model", "unknown"),
            model_type=ModelType.LLM,
            classification_type=ClassificationType.MULTI_LABEL if self.multi_label else ClassificationType.SINGLE_LABEL,
            metadata=metadata
        )
