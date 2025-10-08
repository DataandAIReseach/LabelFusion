"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
import logging
import time
import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import PredictionError, ValidationError, APIError
from ..prompt_engineer.base import PromptEngineer
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager
from .prediction_cache import LLMPredictionCache

class BaseLLMClassifier(AsyncBaseClassifier):
    """Base class for all LLM-based text classifiers."""
    
    def __init__(
        self, 
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot",
        verbose: bool = True,
        provider: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: str = "cache/llm"
    ):
        """Initialize the LLM classifier.
        
        Args:
            config: Configuration object
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier (default: False)
            few_shot_mode: Mode for few-shot learning (default: "few_shot")
            verbose: Whether to show detailed progress (default: True)
            provider: LLM provider to use ('openai', 'gemini', 'deepseek', etc.)
            enable_cache: Whether to enable prediction caching and persistence (default: True)
            cache_dir: Directory for caching prediction results (default: "llm_cache")
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
        
        # Initialize prediction cache
        self.enable_cache = enable_cache
        self.cache = None
        if self.enable_cache:
            self.cache = LLMPredictionCache(
                cache_dir=cache_dir,
                verbose=self.verbose
            )
            if self.verbose:
                self.logger.info(f"Prediction cache initialized: {self.cache.cache_dir}")
        
        self._setup_config()

    def _setup_config(self) -> None:
        """Initialize configuration parameters."""
        self.batch_size = self.config.parameters.get('batch_size', 64)  # Increased from 32 to better support stratified sampling
        self.threshold = self.config.parameters.get('threshold', 0.5)
        
        if self.verbose:
            self.logger.info(f"Configuration setup - Batch size: {self.batch_size}, Threshold: {self.threshold}")

    def _sample_training_data_stratified(
        self, 
        train_df: pd.DataFrame, 
        sample_size: int
    ) -> pd.DataFrame:
        """Sample training data ensuring all classes are represented.
        
        Args:
            train_df: Training DataFrame with one-hot encoded labels
            sample_size: Target number of samples to select
            
        Returns:
            pd.DataFrame: Stratified sample ensuring class representation
        """
        if self.verbose:
            self.logger.info(f"Performing stratified sampling: {sample_size} samples from {len(train_df)} total")
        
        if self.multi_label:
            # For multi-label, use smart sampling to ensure label diversity
            return self._sample_multilabel_data(train_df, sample_size)
        else:
            # For single-label, use stratified sampling
            return self._sample_single_label_data(train_df, sample_size)

    def _sample_single_label_data(
        self, 
        train_df: pd.DataFrame, 
        sample_size: int
    ) -> pd.DataFrame:
        """Stratified sampling for single-label classification."""
        
        # Convert one-hot encoded labels back to single labels for stratification
        label_df = train_df[self.label_columns]
        train_df_copy = train_df.copy()
        train_df_copy['_temp_label'] = label_df.idxmax(axis=1)
        
        # Calculate samples per class
        unique_classes = train_df_copy['_temp_label'].unique()
        num_classes = len(unique_classes)
        min_per_class = max(2, sample_size // num_classes)  # At least 2 per class
        
        if self.verbose:
            self.logger.info(f"Single-label stratified sampling: {num_classes} classes, {min_per_class} min per class")
        
        sampled_dfs = []
        remaining_samples = sample_size
        
        # First pass: ensure minimum representation per class
        for class_val in unique_classes:
            class_df = train_df_copy[train_df_copy['_temp_label'] == class_val]
            
            # Take minimum of: available samples, desired per class, remaining budget
            n_samples = min(len(class_df), min_per_class, remaining_samples)
            
            if n_samples > 0:
                sampled_class = class_df.sample(n=n_samples, random_state=42)
                sampled_dfs.append(sampled_class)
                remaining_samples -= n_samples
                
                if self.verbose:
                    self.logger.debug(f"Class '{class_val}': sampled {n_samples}/{len(class_df)} examples")
        
        # Second pass: distribute remaining samples proportionally
        if remaining_samples > 0 and sampled_dfs:
            combined_sample = pd.concat(sampled_dfs, ignore_index=True)
            
            for class_val in unique_classes:
                if remaining_samples <= 0:
                    break
                    
                class_df = train_df_copy[train_df_copy['_temp_label'] == class_val]
                already_sampled_indices = combined_sample[combined_sample['_temp_label'] == class_val].index
                available_df = class_df.drop(already_sampled_indices, errors='ignore')
                
                if len(available_df) > 0:
                    additional = min(len(available_df), remaining_samples // num_classes + 1)
                    if additional > 0:
                        additional_sample = available_df.sample(n=additional, random_state=42)
                        sampled_dfs.append(additional_sample)
                        remaining_samples -= len(additional_sample)
        
        # Combine all samples and remove temp column
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            result = result.drop('_temp_label', axis=1)
            # Shuffle the final result
            result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # Fallback to random sampling if stratification fails
            result = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        
        if self.verbose:
            self.logger.info(f"Stratified sampling completed: {len(result)} samples selected")
        
        return result

    def _sample_multilabel_data(
        self, 
        train_df: pd.DataFrame, 
        sample_size: int
    ) -> pd.DataFrame:
        """Smart sampling for multi-label data to ensure label diversity."""
        
        if self.verbose:
            self.logger.info(f"Multi-label diversity sampling: {sample_size} samples from {len(train_df)} total")
        
        # Calculate label frequencies to prioritize rare labels
        label_counts = train_df[self.label_columns].sum()
        
        if self.verbose:
            rare_labels = label_counts.nsmallest(5)
            self.logger.info(f"Rarest labels: {rare_labels.to_dict()}")
        
        # Score each sample based on label rarity
        sample_scores = []
        for idx, row in train_df.iterrows():
            score = 0
            active_labels = [col for col in self.label_columns if row[col] == 1]
            
            if active_labels:
                # Higher score for rarer label combinations
                for label in active_labels:
                    label_freq = label_counts[label]
                    score += 1.0 / (label_freq + 1)  # +1 to avoid division by zero
            else:
                score = 0.1  # Small score for samples with no labels
            
            sample_scores.append((idx, score))
        
        # Sort by score (highest first) and take top samples
        sample_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in sample_scores[:sample_size]]
        
        result = train_df.loc[selected_indices].reset_index(drop=True)
        
        if self.verbose:
            # Check label coverage in selected samples
            selected_label_counts = result[self.label_columns].sum()
            covered_labels = (selected_label_counts > 0).sum()
            self.logger.info(f"Multi-label sampling completed: {covered_labels}/{len(self.label_columns)} labels covered")
        
        return result

    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
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
        
        # Handle texts parameter by converting to DataFrame
        if texts is not None and test_df is None:
            test_df = pd.DataFrame({'text': texts})
        
        return asyncio.run(self.predict_async(
            train_df=train_df,
            test_df=test_df,
            context=context,
            label_definitions=label_definitions
            train_df=train_df,
            test_df=test_df,
            context=context,
            label_definitions=label_definitions
        ))

    async def predict_async(
        self,
        test_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Asynchronously predict labels for texts with detailed progress tracking."""
        
        start_time = time.time()
        
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
                train_df=prepared_train_df,
                test_df=prepared_test_df,
                text_column=self.text_column,
                label_columns=self.label_columns
            )
            
            if self.verbose:
                self.logger.info(f"Prompts engineered for {len(df_with_prompts)} samples")
                print(f"{len(df_with_prompts)} prompts ready for processing")
            
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
            
            # Finalize cache
            if self.enable_cache and self.cache:
                self.cache.save_cache()
                if self.verbose:
                    cache_stats = self.cache.get_cache_stats()
                    self.logger.info(f"💾 Cache saved: {cache_stats['total_predictions']} predictions, "
                                   f"{cache_stats['cache_size_mb']:.2f}MB")
                    print(f"Cache: {cache_stats['total_predictions']} predictions saved to {cache_stats['cache_directory']}")
            
            return self._create_result(predictions=predictions, metrics=metrics)
            
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"PREDICTION FAILED: {str(e)}")
                print(f"Error: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.parameters.get("model", "unknown"))
    
    def _prepare_dataframe_for_prediction(self, train_df: Optional[pd.DataFrame], 
                                        test_df: pd.DataFrame, text_column: str, 
                                        label_columns: List[str]) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Prepare DataFrames for prediction with simple validation.
        
        The user is responsible for ensuring data is in the correct format.
        This method only validates basic structure requirements:
        - One text column exists
        - Label columns exist and contain binary values
        
        Args:
            train_df: Optional training DataFrame
            test_df: Test DataFrame
            text_column: Name of the text column
            label_columns: Expected label column names
            
        Returns:
            Tuple of (prepared_train_df, prepared_test_df)
            
        Raises:
            ValidationError: If basic structure requirements are not met
        """
        if self.verbose:
            self.logger.info("Starting simple DataFrame validation...")
        
        prepared_train_df = None
        
        # Process training DataFrame if provided
        if train_df is not None:
            if self.verbose:
                self.logger.info(f"Validating training data: {train_df.shape}")
            
            self._validate_prediction_inputs(train_df, text_column, label_columns)
            prepared_train_df = train_df
        
        # Process test DataFrame
        if self.verbose:
            self.logger.info(f"Validating test data: {test_df.shape}")
        
        self._validate_prediction_inputs(test_df, text_column, label_columns)
        prepared_test_df = test_df
        
        if self.verbose:
            self.logger.info("DataFrame validation completed successfully")
            if prepared_train_df is not None:
                self.logger.info(f"Training data: {prepared_train_df.shape}")
            self.logger.info(f"Test data: {prepared_test_df.shape}")
        
        return prepared_train_df, prepared_test_df


            if self.verbose:
                self.logger.error(f"PREDICTION FAILED: {str(e)}")
                print(f"Error: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}", self.config.parameters.get("model", "unknown"))
    
    def _prepare_dataframe_for_prediction(self, train_df: Optional[pd.DataFrame], 
                                        test_df: pd.DataFrame, text_column: str, 
                                        label_columns: List[str]) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Prepare DataFrames for prediction with simple validation.
        
        The user is responsible for ensuring data is in the correct format.
        This method only validates basic structure requirements:
        - One text column exists
        - Label columns exist and contain binary values
        
        Args:
            train_df: Optional training DataFrame
            test_df: Test DataFrame
            text_column: Name of the text column
            label_columns: Expected label column names
            
        Returns:
            Tuple of (prepared_train_df, prepared_test_df)
            
        Raises:
            ValidationError: If basic structure requirements are not met
        """
        if self.verbose:
            self.logger.info("Starting simple DataFrame validation...")
        
        prepared_train_df = None
        
        # Process training DataFrame if provided
        if train_df is not None:
            if self.verbose:
                self.logger.info(f"Validating training data: {train_df.shape}")
            
            self._validate_prediction_inputs(train_df, text_column, label_columns)
            prepared_train_df = train_df
        
        # Process test DataFrame
        if self.verbose:
            self.logger.info(f"Validating test data: {test_df.shape}")
        
        self._validate_prediction_inputs(test_df, text_column, label_columns)
        prepared_test_df = test_df
        
        if self.verbose:
            self.logger.info("DataFrame validation completed successfully")
            if prepared_train_df is not None:
                self.logger.info(f"Training data: {prepared_train_df.shape}")
            self.logger.info(f"Test data: {prepared_test_df.shape}")
        
        return prepared_train_df, prepared_test_df



    def _validate_prediction_inputs(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        label_columns: List[str]
    ) -> None:
        """Simple validation of prediction inputs.
        
        User is responsible for correct data format. This only checks:
        - DataFrame is not empty
        - Text column exists and contains strings
        - Label columns exist and contain binary values (0/1)
        """
        
        if self.verbose:
            self.logger.info("Performing basic data validation...")
        
        # Basic DataFrame validation
        """Simple validation of prediction inputs.
        
        User is responsible for correct data format. This only checks:
        - DataFrame is not empty
        - Text column exists and contains strings
        - Label columns exist and contain binary values (0/1)
        """
        
        if self.verbose:
            self.logger.info("Performing basic data validation...")
        
        # Basic DataFrame validation
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValidationError("DataFrame is empty")
            
        # Validate text column exists and contains strings
        # Validate text column exists and contains strings
        if text_column not in df.columns:
            raise ValidationError(f"Text column '{text_column}' not found in DataFrame")
        if not df[text_column].dtype == 'object':
            raise ValidationError(f"Text column '{text_column}' must contain text data")
            raise ValidationError(f"Text column '{text_column}' must contain text data")
            
        # Validate label columns exist and contain binary values
        # Validate label columns exist and contain binary values
        for label_col in label_columns:
            if label_col not in df.columns:
                raise ValidationError(f"Label column '{label_col}' not found in DataFrame")
            if not df[label_col].isin([0, 1]).all():
                raise ValidationError(f"Label column '{label_col}' must contain only binary values (0, 1)")
        
        if self.verbose:
            self.logger.info(f"Basic validation passed - Text column: {text_column}, Label columns: {len(label_columns)}")
                raise ValidationError(f"Label column '{label_col}' must contain only binary values (0, 1)")
        
        if self.verbose:
            self.logger.info(f"Basic validation passed - Text column: {text_column}, Label columns: {len(label_columns)}")

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
        """Generate predictions in batches with enhanced progress tracking.
        
        Returns:
            List[List[int]]: List of binary vectors representing predictions
        """
        predictions = []
        total_samples = len(df)
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        # Initialize progress tracking
        cache_hits = 0
        cache_misses = 0
        successful_predictions = 0
        failed_predictions = 0
        
        if self.verbose:
            self.logger.info(f"🚀 Starting prediction for {total_samples} samples in {total_batches} batches")
            print(f"\n📊 Processing {total_samples} samples in {total_batches} batches of size {self.batch_size}")
        
        # Create simple, visible progress bar
        if self.verbose:
            from tqdm import tqdm
            import sys
            
            # Single progress bar for all predictions (simpler and more reliable)
            pbar = tqdm(
                total=total_samples,
                desc="� LLM Predictions",
                unit="sample",
                ncols=80,
                file=sys.stdout,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        batch_iterator = range(0, len(df), self.batch_size)
        
        try:
            for i, batch_start in enumerate(batch_iterator):
                batch_df = df.iloc[batch_start:batch_start + self.batch_size]
                batch_size_actual = len(batch_df)
                
                if self.verbose:
                    # Update postfix with current batch info
                    pbar.set_postfix({
                        'Batch': f'{i+1}/{total_batches}',
                        'Cache': f'💾{cache_hits}',
                        'Miss': f'🌐{cache_misses}',
                        'Fail': f'❌{failed_predictions}'
                    })
                
                # Process batch and track individual predictions
                batch_predictions, batch_stats = await self._process_batch_with_stats(batch_df, text_column)
                predictions.extend(batch_predictions)
                
                # Update statistics
                cache_hits += batch_stats.get('cache_hits', 0)
                cache_misses += batch_stats.get('cache_misses', 0)
                successful_predictions += batch_stats.get('successful', 0)
                failed_predictions += batch_stats.get('failed', 0)
                
                # Update progress bar by batch size
                if self.verbose:
                    pbar.update(batch_size_actual)
                    # Force refresh
                    pbar.refresh()
                
                # Log batch completion for non-tqdm environments
                if not self.verbose:
                    print(f"Completed batch {i+1}/{total_batches}")
        
        finally:
            # Clean up progress bar
            if self.verbose:
                # Final statistics update
                pbar.set_postfix({
                    'Cache': f'💾{cache_hits}',
                    'API': f'🌐{cache_misses}',
                    'Success': f'✅{successful_predictions}',
                    'Failed': f'❌{failed_predictions}'
                })
                pbar.close()
                
                # Print final summary
                print(f"\n📊 Prediction Summary:")
                print(f"   ✅ Successful: {successful_predictions}/{total_samples}")
                print(f"   ❌ Failed: {failed_predictions}/{total_samples}")
                print(f"   💾 Cache hits: {cache_hits}/{total_samples}")
                print(f"   🌐 API calls: {cache_misses}/{total_samples}")
        
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
        """Process a batch of texts for prediction with caching support.
        
        Returns:
            List[List[int]]: List of binary vectors representing predictions
        """
        predictions, _ = await self._process_batch_with_stats(batch_df, text_column)
        return predictions

    async def _process_batch_with_stats(
        self,
        batch_df: pd.DataFrame,
        text_column: str
    ) -> Tuple[List[List[int]], Dict[str, int]]:
        """Process a batch of texts for prediction with caching support and detailed statistics.
        
        Returns:
            Tuple[List[List[int]], Dict[str, int]]: (predictions, statistics)
        """
        if self.verbose:
            self.logger.debug(f"Processing batch with {len(batch_df)} texts")
        
        texts = batch_df[text_column].tolist()
        batch_predictions = []
        
        # Statistics tracking
        stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'successful': 0,
            'failed': 0
        }
        
        # Check if prompts are already engineered
        if 'engineered_prompt' in batch_df.columns:
            prompts = batch_df['engineered_prompt'].tolist()
        else:
            prompts = [self.prompt_engineer.build_prompt(text) for text in texts]
        
        if self.verbose:
            self.logger.debug(f"Generated {len(prompts)} prompts for LLM calls")

        # Process each text in the batch
        for i, (text, prompt) in enumerate(zip(texts, prompts)):
            try:
                # Check cache first
                if self.enable_cache and self.cache and self.cache.has_prediction(text):
                    cached_pred = self.cache.get_prediction(text)
                    if cached_pred and cached_pred["success"]:
                        batch_predictions.append(cached_pred["prediction"])
                        stats['cache_hits'] += 1
                        stats['successful'] += 1
                        if self.verbose:
                            self.logger.debug(f"Using cached prediction for text {i+1}")
                        continue
                
                # Cache miss - make LLM call
                stats['cache_misses'] += 1
                response = await self._call_llm(prompt)
                prediction = self._parse_prediction_response(response)
                batch_predictions.append(prediction)
                stats['successful'] += 1
                
                # Store in cache
                if self.enable_cache and self.cache:
                    self.cache.store_prediction(
                        text=text,
                        prediction=prediction,
                        response_text=response,
                        prompt=prompt,
                        success=True
                    )
                
            except Exception as e:
                error_prediction = self._handle_error(e)
                batch_predictions.append(error_prediction)
                stats['failed'] += 1
                stats['cache_misses'] += 1  # Failed predictions are also cache misses
                
                # Store failed prediction in cache
                if self.enable_cache and self.cache:
                    self.cache.store_prediction(
                        text=text,
                        prediction=error_prediction,
                        response_text="",
                        prompt=prompt,
                        success=False,
                        error_message=str(e)
                    )
        
        if self.verbose and stats['failed'] > 0:
            self.logger.warning(f"WARNING: {stats['failed']} out of {len(batch_predictions)} predictions failed")
        
        return batch_predictions, stats

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
        predictions: List[List[int]],
        label_columns: List[str]
    ) -> Optional[Dict[str, float]]:
        """Evaluate model performance on test data using provided predictions."""
        if test_df is None or not predictions:
        """Evaluate model performance on test data using provided predictions."""
        if test_df is None or not predictions:
            return None
            
        # Convert true labels to binary vector format (same as predictions)
        true_labels = test_df[label_columns].values.astype(int).tolist()
        # Convert true labels to binary vector format (same as predictions)
        true_labels = test_df[label_columns].values.astype(int).tolist()
        return self._calculate_metrics(predictions, true_labels)

    def _calculate_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
        predictions: List[List[int]],
        true_labels: List[List[int]]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics for binary vector predictions."""
        if not self.multi_label:
        """Calculate evaluation metrics for binary vector predictions."""
        if not self.multi_label:
            return self._calculate_single_label_metrics(predictions, true_labels)
        return self._calculate_multi_label_metrics(predictions, true_labels)

    def _calculate_single_label_metrics(
        self,
        predictions: List[List[int]],
        true_labels: List[List[int]]
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

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled."""
        if self.enable_cache and self.cache:
            return self.cache.get_cache_stats()
        return None
    
    def export_cache(self, output_file: str, format: str = "csv") -> None:
        """Export cached predictions to a file.
        
        Args:
            output_file: Path to output file
            format: Export format ('csv', 'json', 'xlsx')
        """
        if not self.enable_cache or not self.cache:
            raise ValueError("Caching is not enabled")
        
        self.cache.export_predictions(output_file, format)
        if self.verbose:
            self.logger.info(f"Cache exported to {output_file}")
    
    def clear_cache(self, confirm: bool = False) -> None:
        """Clear the prediction cache.
        
        Args:
            confirm: Must be True to actually clear the cache
        """
        if not self.enable_cache or not self.cache:
            raise ValueError("Caching is not enabled")
        
        self.cache.clear_cache(confirm=confirm)
        if self.verbose:
            self.logger.info("Cache cleared")

    async def _engineer_prompts_for_data(
        self,
        test_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        text_column: Optional[str] = None,
        label_columns: Optional[List[str]] = None
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

        # Engineer prompts using PromptEngineer with stratified sampling
        if train_df is not None and len(train_df) > self.batch_size:
            # Use stratified sampling to ensure all classes are represented
            sampled_train_df = self._sample_training_data_stratified(train_df, self.batch_size)
            if self.verbose:
                self.logger.info(f"Applied stratified sampling: {len(sampled_train_df)} samples from {len(train_df)} training examples")
        else:
            sampled_train_df = train_df
        
        engineered_prompts = await self.prompt_engineer.engineer_prompts(
            test_df=test_df,
            train_df=sampled_train_df,
            sample_size=self.batch_size
        )

        if self.verbose:
            self.logger.info(f"Successfully engineered {len(engineered_prompts)} prompts")
            print(f"{len(engineered_prompts)} prompts engineered")


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
