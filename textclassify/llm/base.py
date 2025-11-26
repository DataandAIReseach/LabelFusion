"""Base class for LLM-based text classifiers."""

import asyncio
import json
import re
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
import pandas as pd
from tqdm import tqdm
import datetime

from ..core.base import AsyncBaseClassifier
from ..core.types import ClassificationResult, ClassificationType, ModelType
from ..core.exceptions import PredictionError, ValidationError, APIError
from ..prompt_engineer.base import PromptEngineer
from ..services.llm_content_generator import create_llm_generator
from ..config.api_keys import APIKeyManager
from ..utils.results_manager import ResultsManager, ModelResultsManager
from .prediction_cache import LLMPredictionCache

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
        provider: Optional[str] = None,
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True,
        # Cache management parameters
        auto_use_cache: bool = True,
        cache_dir: str = "cache"
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
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
            auto_use_cache: Whether to automatically check and reuse cached predictions (default: False)
            cache_dir: Directory to search for cached predictions (default: "cache")
        """
        super().__init__(config)
        self.config.model_type = ModelType.LLM
        self.multi_label = multi_label
        self.few_shot_mode = few_shot_mode
        self.verbose = verbose
        self.mode = None
        
        # Set provider - use parameter if provided, otherwise get from config, default to openai
        self.provider = provider or getattr(self.config, 'provider', 'openai')
        # Also set it on config for consistency
        self.config.provider = self.provider
        
        # Cache management settings
        self.auto_use_cache = auto_use_cache
        self.cache_dir = cache_dir
        
        # Setup logging
        if self.verbose:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(self.__class__.__name__)
        
        # Suppress HTTP request console output (but keep logging to file if configured)
        # These loggers will still log to files, but not to console
        for logger_name in ["httpx", "openai", "httpcore", "urllib3", "requests"]:
            http_logger = logging.getLogger(logger_name)
            http_logger.setLevel(logging.WARNING)
            # Remove console handlers but keep file handlers
            http_logger.propagate = False
        
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
        
        # Initialize results management
        self.results_manager = None
        if auto_save_results:
            model_name = self.config.parameters.get("model", "unknown")
            if not experiment_name:
                experiment_name = f"{self.provider}_{model_name.replace('/', '_').replace('-', '_')}"
            
            self.results_manager = ResultsManager(
                base_output_dir=output_dir,
                experiment_name=experiment_name
            )
            self.model_results_manager = ModelResultsManager(
                self.results_manager, 
                f"{self.provider}_classifier_{self.results_manager.experiment_id}"
            )
            
            if self.verbose:
                exp_info = self.results_manager.get_experiment_info()
                self.logger.info(f"📁 Results will be saved to: {exp_info['experiment_dir']}")
                self.logger.info(f"🔬 Experiment ID: {self.results_manager.experiment_id}")
        
        self._setup_config()
    
    def set_mode(self, mode: str) -> None:
        self.mode = mode

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
        
        # 🚀 AUTO-CACHE: Check for cached predictions if enabled
        if self.auto_use_cache:
            if self.verbose:
                print("\n🔍 Auto-cache enabled, checking for cached predictions...")
            
            discovered = self.discover_cached_predictions(self.cache_dir)
            
            # Try to find a matching cache file for test predictions
            if 'test_predictions' in discovered and discovered['test_predictions']:
                cache_file = discovered['test_predictions'][0]  # Use most recent
                
                if self.verbose:
                    print(f"✅ Found cached predictions: {Path(cache_file).name}")
                    print("📥 Loading from cache (1000-5000x faster than inference)...")
                
                try:
                    # Load and return cached predictions
                    result = self.predict_with_cached_predictions(test_df, cache_file, train_df)
                    
                    if self.verbose:
                        print(f"⚡ Cache load completed in {time.time() - start_time:.2f} seconds")
                    
                    return result
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Cache load failed: {e}")
                        print("🔄 Falling back to normal inference...")
            elif self.verbose:
                print("ℹ️  No cached predictions found, running inference...")

        # Initialize incremental prediction cache for resume/skip functionality
        if self.auto_use_cache:
            try:
                # LLMPredictionCache will load any existing per-session cache files
                self._prediction_cache = LLMPredictionCache(cache_dir=self.cache_dir, verbose=self.verbose, auto_save_interval=1)
                if self.verbose:
                    self.logger.info(f"Prediction cache initialized (session: {self._prediction_cache.session_id})")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Could not initialize prediction cache: {e}")
                    self._prediction_cache = None
        else:
            self._prediction_cache = None
        
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
            
            # Save experiment summary if results manager is available
            if self.results_manager:
                try:
                    experiment_summary = {
                        'model_type': 'llm',
                        'provider': self.provider,
                        'model_name': self.config.parameters.get("model", "unknown"),
                        'prediction_samples': len(prepared_test_df),
                        'training_samples': len(prepared_train_df) if prepared_train_df is not None else 0,
                        'processing_time_seconds': total_time,
                        'avg_time_per_sample': total_time/len(prepared_test_df),
                        'multi_label': self.multi_label,
                        'few_shot_mode': self.few_shot_mode,
                        'text_column': self.text_column,
                        'label_columns': self.label_columns,
                        'accuracy': metrics.get('accuracy') if metrics else None,
                        'completed': True
                    }
                    
                    self.results_manager.save_experiment_summary(experiment_summary)
                    
                    if self.verbose:
                        exp_info = self.results_manager.get_experiment_info()
                        self.logger.info(f"📋 Experiment summary saved to: {exp_info['experiment_dir']}")
                        
                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Could not save experiment summary: {e}")
            
            return self._create_result(
                predictions=predictions, 
                metrics=metrics,
                test_df=test_df,
                train_df=train_df
            )
            
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
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValidationError("DataFrame is empty")
            
        # Validate text column exists and contains strings
        if text_column not in df.columns:
            raise ValidationError(f"Text column '{text_column}' not found in DataFrame")
        if not df[text_column].dtype == 'object':
            raise ValidationError(f"Text column '{text_column}' must contain text data")
            
        # Validate label columns exist and contain binary values
        for label_col in label_columns:
            if label_col not in df.columns:
                raise ValidationError(f"Label column '{label_col}' not found in DataFrame")
            if not df[label_col].isin([0, 1]).all():
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
    
    def has_test_cache_for_dataset(self, df: pd.DataFrame) -> bool:
        """Return True if a test cache file matching the given DataFrame exists, else False.

        Computes an 8-char dataset hash from the provided DataFrame and checks
        cached test JSON filenames and their metadata for a match. Uses self.cache_dir.
        """
        from pathlib import Path
        import hashlib
        import json

        cache_dir = getattr(self, 'cache_dir', 'cache')
        p = Path(cache_dir)
        if not p.exists():
            return False

        # Compute stable 8-char hash for DataFrame
        try:
            hashed = pd.util.hash_pandas_object(df, index=True).values
            dataset_hash = hashlib.md5(hashed).hexdigest()[:8]
        except Exception:
            csv_bytes = df.to_csv(index=True).encode('utf-8')
            dataset_hash = hashlib.md5(csv_bytes).hexdigest()[:8]

        discovered = self.discover_cached_predictions(cache_dir)
        if not discovered:
            return False

        candidates = discovered.get('test_predictions', []) or []
        for file_path in candidates:
            try:
                fname = Path(file_path).name
                if fname.endswith(f"_{dataset_hash}.json"):
                    return True

                with open(file_path, 'r') as f:
                    data = json.load(f)
                meta = data.get('metadata', {}) if isinstance(data, dict) else {}
                if meta.get('dataset_hash') == dataset_hash:
                    return True

            except Exception:
                continue

        return False



    async def _generate_predictions(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> List[List[int]]:
        """Generate predictions in batches with progress tracking.
        
        Returns:
            List[List[int]]: List of binary vectors representing predictions
        """
        # If an incremental prediction cache is available, skip already-predicted rows
        predictions: List[Optional[List[int]]] = [None] * len(df)
        
        # Initialize batch-wise cache file
        cache_file_path = self._initialize_batch_cache_file(df)

        # Build list of indices for which we need to run inference
        if self.has_test_cache_for_dataset(df):
            uncached_positions: List[int] = []
            for pos, (_, row) in enumerate(df.iterrows()):
                text = row.get(text_column, "")
                try:
                    if self._prediction_cache.has_prediction(text):
                        cached = self._prediction_cache.get_prediction(text)
                        pred = cached.get('prediction') if isinstance(cached, dict) else None
                        if isinstance(pred, list):
                            predictions[pos] = pred
                            continue
                except Exception:
                    # On any cache error, treat as uncached and continue
                    pass
                uncached_positions.append(pos)

            if len(uncached_positions) == 0:
                # All samples cached — return formatted cached predictions
                final_preds = [p for p in predictions if p is not None]
                return final_preds

            # Create a DataFrame with only uncached rows (preserve order)
            df_uncached = df.iloc[uncached_positions]
            total_batches = (len(df_uncached) + self.batch_size - 1) // self.batch_size
        else:
            df_uncached = df
            total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        if self.verbose:
            self.logger.info(f"Processing {len(df)} samples in {total_batches} batches of size {self.batch_size}")
            print(f"Processing in {total_batches} batches...")
        
        # Use tqdm for progress bar if verbose mode is enabled
        batch_iterator = range(0, len(df_uncached), self.batch_size)
        if self.verbose:
            batch_iterator = tqdm(batch_iterator, desc="Processing batches", total=total_batches)
        
        for i, batch_start in enumerate(batch_iterator):
            batch_df = df_uncached.iloc[batch_start:batch_start + self.batch_size]
            
            if self.verbose and not isinstance(batch_iterator, tqdm):
                self.logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch_df)} samples)")
                print(f"Batch {i+1}/{total_batches} processing...")
            
            batch_predictions = await self._process_batch(batch_df, text_column)

            # If using an incremental cache, write batch predictions back into the
            # main predictions list at their original positions
            if hasattr(self, '_prediction_cache') and self._prediction_cache and df_uncached is not df:
                # Map batch_df positions back to original positional indices in `predictions`.
                # `df_uncached.index` contains DataFrame index labels which may not be
                # 0..N-1 contiguous positions; using them directly as list indices
                # causes IndexError when labels are large or non-sequential. Use the
                # previously-built `uncached_positions` (positional offsets) instead.
                try:
                    batch_positions = uncached_positions[batch_start: batch_start + self.batch_size]
                except Exception:
                    # Fallback: compute positional indices from labels (slower)
                    batch_positions = [df.index.get_loc(lbl) for lbl in df_uncached.index[batch_start: batch_start + self.batch_size]]

                for rel_idx, orig_pos in enumerate(batch_positions):
                    if rel_idx < len(batch_predictions):
                        predictions[orig_pos] = batch_predictions[rel_idx]
            else:
                # No cache in use or full run — append sequentially
                # When not using cache, predictions list may be all None placeholders,
                # so we append to build the final list
                if all(p is None for p in predictions):
                    # fresh append mode
                    predictions = []
                    predictions.extend(batch_predictions)
                else:
                    # mix-mode: fill next available None slots
                    fill_idx = 0
                    for j in range(len(predictions)):
                        if predictions[j] is None and fill_idx < len(batch_predictions):
                            predictions[j] = batch_predictions[fill_idx]
                            fill_idx += 1
            
            if self.verbose and not isinstance(batch_iterator, tqdm):
                self.logger.info(f"Batch {i+1} completed ({len(batch_predictions)} predictions)")
            
            # Write batch predictions to cache file
            self._write_batch_to_cache(cache_file_path, batch_df, batch_predictions, text_column, i+1, total_batches)
        
        # Final cleanup: ensure no None remain (fallback defaults)
        final_predictions: List[List[int]] = []
        for p in predictions:
            if p is None:
                # fallback to a safe default
                if self.multi_label:
                    final_predictions.append([0] * len(self.classes_))
                else:
                    final_predictions.append([1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1])
            else:
                final_predictions.append(p)

        return final_predictions
    
    def _initialize_batch_cache_file(self, df: pd.DataFrame) -> str:
        """Initialize a JSON cache file for batch-wise prediction storage.
        
        Args:
            df: DataFrame being processed
            
        Returns:
            str: Path to the cache file
        """
        import hashlib
        from datetime import datetime as dt
        from pathlib import Path
        
        # Create cache directory if it doesn't exist
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute dataset hash
        try:
            hashed = pd.util.hash_pandas_object(df, index=True).values
            dataset_hash = hashlib.md5(hashed).hexdigest()[:8]
        except Exception:
            csv_bytes = df.to_csv(index=True).encode('utf-8')
            dataset_hash = hashlib.md5(csv_bytes).hexdigest()[:8]
        
        # Create cache filename with timestamp and dataset hash in the format
        # expected by other parts of the code (e.g. test_YYYY-MM-DD-HH-MM-SS_hash.json)
        timestamp = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        cache_filename = f"test_{timestamp}_{dataset_hash}.json"
        cache_file_path = cache_dir / cache_filename
        
        # Initialize cache file with metadata
        provider = getattr(self, 'provider', 'llm')
        cache_data = {
            'metadata': {
                'provider': provider,
                'model': self.config.parameters.get('model', 'unknown'),
                'dataset_hash': dataset_hash,
                'dataset_size': len(df),
                'timestamp': timestamp,
                'multi_label': self.multi_label,
                'label_columns': self.label_columns,
                'text_column': self.text_column,
                'batch_size': self.batch_size
            },
            'predictions': []
        }
        
        with open(cache_file_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        if self.verbose:
            self.logger.info(f"Initialized batch cache file: {cache_file_path}")
        
        return str(cache_file_path)
    
    def _write_batch_to_cache(self, cache_file_path: str, batch_df: pd.DataFrame, 
                              batch_predictions: List[List[int]], text_column: str,
                              batch_num: int, total_batches: int) -> None:
        """Write batch predictions to the cache JSON file.
        
        Args:
            cache_file_path: Path to cache file
            batch_df: DataFrame for current batch
            batch_predictions: Predictions for current batch
            text_column: Name of text column
            batch_num: Current batch number
            total_batches: Total number of batches
        """
        try:
            # Read existing cache data
            with open(cache_file_path, 'r') as f:
                cache_data = json.load(f)
            
            # Add batch predictions
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                if idx < len(batch_predictions):
                    prediction_entry = {
                        'text': row.get(text_column, ''),
                        'prediction': batch_predictions[idx],
                        'batch_num': batch_num
                    }
                    
                    # Add original labels if available
                    if self.label_columns:
                        prediction_entry['true_labels'] = [int(row.get(col, 0)) for col in self.label_columns]
                    
                    cache_data['predictions'].append(prediction_entry)
            
            # Update metadata
            cache_data['metadata']['batches_completed'] = batch_num
            cache_data['metadata']['total_batches'] = total_batches
            cache_data['metadata']['predictions_count'] = len(cache_data['predictions'])
            cache_data['metadata']['last_updated'] = datetime.datetime.now().isoformat()
            
            # Write updated cache data
            with open(cache_file_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            if self.verbose and batch_num % 5 == 0:  # Log every 5 batches to avoid spam
                self.logger.info(f"Wrote batch {batch_num}/{total_batches} to cache ({len(batch_predictions)} predictions)")
                
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Failed to write batch {batch_num} to cache: {e}")

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
        """Process a batch of texts for prediction with detailed logging and retry logic.
        
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
        
        # Parse responses and retry if needed for multi-label with wrong length
        predictions = []
        max_retries = 3
        
        for i, r in enumerate(responses):
            if isinstance(r, Exception):
                predictions.append(self._handle_error(r))
                continue
            
            # Try parsing with retries for multi-label binary format issues
            prediction = None
            for attempt in range(max_retries):
                prediction = self._parse_prediction_response(r)
                
                # Check if we got the right number of labels (only for multi-label)
                if self.multi_label and len(prediction) == len(self.classes_):
                    break  # Success!
                elif not self.multi_label:
                    break  # For single-label, we don't need to retry
                else:
                    # Wrong number of labels in multi-label - retry
                    if attempt < max_retries - 1:
                        if self.verbose:
                            self.logger.warning(f"Retry {attempt + 1}/{max_retries}: Wrong label count ({len(prediction)} != {len(self.classes_)}), retrying LLM call...")
                        # Call LLM again
                        r = await self._call_llm(prompts[i])
                        if isinstance(r, Exception):
                            prediction = self._handle_error(r)
                            break
                    else:
                        if self.verbose:
                            self.logger.warning(f"After {max_retries} retries, still wrong label count. Using best-effort prediction.")
            
            predictions.append(prediction)

        # If incremental cache is enabled, store the raw responses + parsed predictions
        try:
            if hasattr(self, '_prediction_cache') and self._prediction_cache:
                for idx, pred in enumerate(predictions):
                    # Determine success and response text
                    resp = responses[idx]
                    success = not isinstance(resp, Exception)
                    response_text = resp if success else ''
                    prompt = prompts[idx] if idx < len(prompts) else ''
                    text_val = batch_df.iloc[idx].get(self.text_column, '')

                    # Attempt to lift an identifier from the DataFrame row if present
                    meta = {}
                    for id_key in ('id', 'ID', 'Id', 'index', 'row_id'):
                        if id_key in batch_df.columns:
                            try:
                                meta['id'] = batch_df.iloc[idx][id_key]
                                break
                            except Exception:
                                pass

                    try:
                        # store_prediction expects binary vector as prediction
                        self._prediction_cache.store_prediction(
                            text_val,
                            pred,
                            response_text,
                            prompt,
                            success=success,
                            error_message=None if success else str(resp),
                            metadata=meta or None
                        )
                    except Exception:
                        # non-fatal: continue without failing the batch
                        if self.verbose:
                            self.logger.debug("Could not store prediction to cache for a sample")
        except Exception:
            # If anything goes wrong with caching, do not fail predictions
            if self.verbose:
                self.logger.debug("Prediction caching encountered an error; continuing without persisting this batch")

        return predictions

    def _parse_prediction_response(self, response: str) -> List[int]:
        """Parse the LLM response for predictions.
        
        Returns:
            List[int]: Binary vector representation of predictions
        """
        if not response or not response.strip():
            if self.verbose:
                self.logger.warning("Received empty or whitespace-only response from LLM")
            # Return default values for empty responses
            if self.multi_label:
                return [0] * len(self.classes_) if self.classes_ else [0]
            else:
                return [1] + [0] * (len(self.classes_) - 1) if self.classes_ else [1]
        
        response = response.strip()
        
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
            if all(part in ['0', '1'] for part in binary_parts):
                # Check if length matches
                if len(binary_parts) == len(self.classes_):
                    # Perfect match - convert to integer list and return
                    return [int(part) for part in binary_parts]
                else:
                    # Wrong length - pad or truncate to match expected length
                    if self.verbose:
                        self.logger.warning(f"Binary response length ({len(binary_parts)}) doesn't match classes count ({len(self.classes_)})")
                    
                    # Pad with zeros if too short
                    if len(binary_parts) < len(self.classes_):
                        binary_parts.extend(['0'] * (len(self.classes_) - len(binary_parts)))
                        if self.verbose:
                            self.logger.info(f"Padded binary response to {len(self.classes_)} labels")
                    # Truncate if too long
                    elif len(binary_parts) > len(self.classes_):
                        binary_parts = binary_parts[:len(self.classes_)]
                        if self.verbose:
                            self.logger.info(f"Truncated binary response to {len(self.classes_)} labels")
                    
                    return [int(part) for part in binary_parts]
            else:
                if self.verbose:
                    self.logger.warning(f"Binary response contains non-binary values: {binary_parts}")
        
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
            # Prefer displaying the size of the few-shot training data when available
            if train_df is not None and not train_df.empty:
                self.logger.info(
                    f"Engineering prompts using {len(train_df)} training samples for few-shot learning; "
                    f"generating prompts for {len(test_df)} test samples"
                )
            else:
                self.logger.info(f"Engineering prompts for {len(test_df)} test samples")

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
        # Choose progress total: prefer training-sample count when available, else prompt count
        progress_total = len(test_df) if (test_df is not None and not test_df.empty) else len(engineered_prompts)
        # Use tqdm to show progress; fall back to simple iteration if verbose is False
        iterator = enumerate(engineered_prompts)
        if self.verbose:
            iterator = enumerate(tqdm(engineered_prompts, desc="Rendering prompts", total=progress_total))

        for i, prompt in iterator:
            if self.verbose and i % 10 == 0:
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
        metrics: Optional[Dict[str, float]] = None,
        test_df: Optional[pd.DataFrame] = None,
        train_df: Optional[pd.DataFrame] = None
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
        
        result = ClassificationResult(
            predictions=string_predictions,
            model_name=self.config.parameters.get("model", "unknown"),
            model_type=ModelType.LLM,
            classification_type=ClassificationType.MULTI_LABEL if self.multi_label else ClassificationType.SINGLE_LABEL,
            metadata=metadata
        )
        
        # Save results using ResultsManager
        if self.results_manager and test_df is not None:
            try:
                # Determine dataset type based on presence of train_df
                dataset_type = "test" if train_df is not None else "prediction"
                
                # Save predictions
                saved_files = self.results_manager.save_predictions(
                    result, dataset_type, test_df
                )
                
                # Save metrics if available
                if metrics:
                    metrics_file = self.results_manager.save_metrics(
                        metrics, dataset_type, f"{self.provider}_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                # Save model configuration
                model_config = {
                    'provider': self.provider,
                    'model_name': self.config.parameters.get("model", "unknown"),
                    'multi_label': self.multi_label,
                    'few_shot_mode': self.few_shot_mode,
                    'text_column': self.text_column,
                    'label_columns': self.label_columns,
                    'batch_size': self.batch_size,
                    'threshold': self.threshold
                }
                
                config_file = self.results_manager.save_model_config(
                    model_config, f"{self.provider}_classifier"
                )
                saved_files["config"] = config_file
                
                if self.verbose:
                    self.logger.info(f"📁 Results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Could not save results: {e}")
        
        return result
    
    def predict_texts(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict labels for a list of texts (compatibility method for FusionEnsemble).
        
        This method is provided for compatibility with FusionEnsemble which calls 
        LLM models with text lists. It handles metrics calculation and results saving
        automatically when true_labels are provided.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        # Convert texts to DataFrame format
        import pandas as pd
        test_df = pd.DataFrame({self.text_column: texts})
        
        # Add label columns if true_labels provided (for metrics calculation)
        if true_labels is not None and self.label_columns:
            for i, label_col in enumerate(self.label_columns):
                if i < len(true_labels[0]) if true_labels else 0:
                    test_df[label_col] = [labels[i] if i < len(labels) else 0 for labels in true_labels]
                else:
                    test_df[label_col] = 0
        
        # Create an empty train_df to avoid prompt engineering issues when using cached predictions
        train_df = pd.DataFrame(columns=test_df.columns)
        
        # Call the regular predict method with both train_df and test_df
        result = self.predict(train_df=train_df, test_df=test_df)
        
        # If metrics weren't calculated yet but we have true labels, calculate them
        if (true_labels is not None and 
            (not hasattr(result, 'metadata') or not result.metadata or 'metrics' not in result.metadata)):
            
            try:
                # Convert string predictions back to binary format for metric calculation
                predicted_labels = []
                for pred in result.predictions:
                    if self.multi_label:
                        # Multi-label: convert list of class names to binary vector
                        binary_pred = [0] * len(self.classes_)
                        if isinstance(pred, list):
                            for class_name in pred:
                                if class_name in self.classes_:
                                    binary_pred[self.classes_.index(class_name)] = 1
                        predicted_labels.append(binary_pred)
                    else:
                        # Single-label: convert class name to binary vector
                        binary_pred = [0] * len(self.classes_)
                        if pred in self.classes_:
                            binary_pred[self.classes_.index(pred)] = 1
                        else:
                            # Default to first class if prediction not in classes
                            binary_pred[0] = 1 if self.classes_ else 0
                        predicted_labels.append(binary_pred)
                
                # Calculate metrics using the class method
                metrics = self._calculate_metrics(predicted_labels, true_labels)
                
                # Add metrics to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['metrics'] = metrics
                
                # Save metrics to file if results_manager is available
                if self.results_manager:
                    try:
                        metrics_file = self.results_manager.save_metrics(
                            metrics, "test", f"{self.provider}_classifier"
                        )
                        if 'saved_files' not in result.metadata:
                            result.metadata['saved_files'] = {}
                        result.metadata['saved_files']['metrics'] = metrics_file
                        
                        if self.verbose:
                            print(f"📁 {self.provider.title()} classifier metrics saved: {metrics_file}")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Could not save {self.provider} classifier metrics: {e}")
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not calculate metrics for {self.provider} classifier: {e}")
        
        return result
    
    # ========================================================================
    # Cache Management Methods
    # ========================================================================
    
    @classmethod
    def discover_cached_predictions(cls, cache_dir: str = "cache") -> Dict[str, List[str]]:
        """Discover all cached prediction files in the specified directory.
        
        This class method scans a directory for cached prediction files and groups them
        by dataset type (train, validation, test).
        
        Args:
            cache_dir: Directory to search for cache files (default: "cache")
            
        Returns:
            Dictionary mapping dataset types to lists of cache file paths
            Example: {'validation_predictions': ['/path/to/val_file.json'], 
                     'test_predictions': ['/path/to/test_file.json']}
        """
        from pathlib import Path
        import re
        
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            print(f"⚠️  Cache directory not found: {cache_dir}")
            return {}
        
        discovered = {}
        
        # Patterns to match different cache file types
        # Supports both formats: YYYY-MM-DD-HH-MM-SS and YYYY_MM_DD_HH_MM_SS
        patterns = {
            'train_predictions': r'train[_-](?:predictions[_-])?\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[_-][a-f0-9]+\.json',
            'validation_predictions': r'val(?:idation)?[_-](?:predictions[_-])?\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[_-][a-f0-9]+\.json',
            'test_predictions': r'test[_-](?:predictions[_-])?\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[_-][a-f0-9]+\.json'
        }
        
        for dataset_type, pattern in patterns.items():
            matching_files = []
            for file_path in cache_path.rglob("*.json"):
                if re.match(pattern, file_path.name):
                    matching_files.append(str(file_path))
            
            if matching_files:
                # Sort by timestamp (most recent first)
                matching_files.sort(reverse=True)
                discovered[dataset_type] = matching_files
        
        if discovered:
            print(f"📁 Discovered {sum(len(v) for v in discovered.values())} cached prediction files")
            for dataset_type, files in discovered.items():
                print(f"  • {dataset_type}: {len(files)} file(s)")
        else:
            print(f"ℹ️  No cached prediction files found in {cache_dir}")
        
        return discovered
    
    def load_cached_predictions_for_dataset(
        self, 
        cache_file: str,
        dataset_type: str = "validation"
    ) -> Optional[Dict[str, Any]]:
        """Load cached predictions from a specific file.
        
        Args:
            cache_file: Path to the cache file (JSON format)
            dataset_type: Type of dataset ('train', 'validation', or 'test')
            
        Returns:
            Dictionary with cached predictions or None if loading fails
        """
        from pathlib import Path
        
        cache_path = Path(cache_file)
        if not cache_path.exists():
            print(f"❌ Cache file not found: {cache_file}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            if self.verbose:
                print(f"✅ Loaded {len(cached_data.get('predictions', []))} cached predictions from {cache_file}")
            
            return cached_data
            
        except Exception as e:
            print(f"❌ Error loading cache file {cache_file}: {e}")
            return None
    
    def get_cached_predictions_summary(
        self, 
        cache_file: Optional[str] = None,
        cache_dir: str = "cache"
    ) -> Dict[str, Any]:
        """Get a summary of cached predictions.
        
        Args:
            cache_file: Specific cache file to analyze (optional)
            cache_dir: Directory to search if cache_file not provided
            
        Returns:
            Dictionary with summary statistics about cached predictions
        """
        if cache_file:
            cached_data = self.load_cached_predictions_for_dataset(cache_file)
            if not cached_data:
                return {}
        else:
            # Discover all cache files
            discovered = self.discover_cached_predictions(cache_dir)
            if not discovered:
                return {}
            
            # Use the most recent validation cache by default
            if 'validation_predictions' in discovered:
                cache_file = discovered['validation_predictions'][0]
            elif 'test_predictions' in discovered:
                cache_file = discovered['test_predictions'][0]
            else:
                cache_file = list(discovered.values())[0][0]
            
            cached_data = self.load_cached_predictions_for_dataset(cache_file)
            if not cached_data:
                return {}
        
        predictions = cached_data.get('predictions', [])
        
        summary = {
            'cache_file': cache_file,
            'total_predictions': len(predictions),
            'provider': cached_data.get('provider', 'unknown'),
            'model': cached_data.get('model', 'unknown'),
            'timestamp': cached_data.get('timestamp', 'unknown'),
            'has_metrics': 'metrics' in cached_data
        }
        
        if 'metrics' in cached_data:
            summary['metrics'] = cached_data['metrics']
        
        return summary
    
    def predict_with_cached_predictions(
        self,
        test_df: pd.DataFrame,
        cache_file: str,
        train_df: Optional[pd.DataFrame] = None
    ) -> ClassificationResult:
        """Load predictions from cache file instead of running inference.
        
        This method is useful for quickly testing ensemble combinations without
        re-running expensive LLM inference.
        
        Args:
            test_df: Test DataFrame (used for structure and labels)
            cache_file: Path to cached predictions JSON file
            train_df: Optional training DataFrame (not used with cached predictions)
            
        Returns:
            ClassificationResult with cached predictions
        """
        cached_data = self.load_cached_predictions_for_dataset(cache_file)
        if not cached_data:
            raise ValueError(f"Could not load cache file: {cache_file}")
        
        predictions = cached_data.get('predictions', [])

        # If sizes match, we can use predictions directly. If not, try to resolve
        # by matching IDs (if present in the cache and the test_df), otherwise
        # raise a helpful error explaining how to proceed.
        if len(predictions) != len(test_df):
            # Attempt to match cached predictions by explicit ids stored in cache
            cache_ids = cached_data.get('ids') or cached_data.get('id')
            metadata = cached_data.get('metadata', {}) if isinstance(cached_data, dict) else {}

            if cache_ids and isinstance(cache_ids, list):
                # Try to find an id column in test_df that appears in cache metadata
                id_col = None
                if 'columns' in metadata and isinstance(metadata['columns'], list):
                    # prefer exact match with one of the metadata columns
                    for c in test_df.columns:
                        if c in metadata['columns']:
                            id_col = c
                            break

                # fallback common names
                if id_col is None:
                    for cand in ['id', 'ID', 'Id']: 
                        if cand in test_df.columns:
                            id_col = cand
                            break

                if id_col is not None:
                    # build mapping and pick predictions for test_df rows
                    id_to_pred = {cid: pred for cid, pred in zip(cache_ids, predictions)}
                    resolved = []
                    missing = 0
                    for val in test_df[id_col].tolist():
                        if val in id_to_pred:
                            resolved.append(id_to_pred[val])
                        else:
                            # keep placeholder (all-zero or first-class fallback later)
                            resolved.append(None)
                            missing += 1

                    if missing == 0:
                        predictions = resolved
                    else:
                        # If many missing, raise to avoid silent mismatches
                        raise ValueError(
                            f"Cache file contains predictions for {len(cache_ids)} ids but {missing} "
                            f"rows in test_df (using id column '{id_col}') could not be matched.\n"
                            "To use cached predictions for a subset, ensure the test DataFrame contains an 'id' column "
                            "matching the cached 'ids', or run inference on the full dataset."
                        )
                else:
                    raise ValueError(
                        f"Cache file has {len(predictions)} predictions but test_df has {len(test_df)} rows.\n"
                        "The cache contains explicit ids but no matching id column was found in test_df.\n"
                        "Possible fixes: add an 'id' column to test_df that matches the cached ids, or disable auto-cache."
                    )
            else:
                raise ValueError(
                    f"Cache file has {len(predictions)} predictions but test_df has {len(test_df)} rows.\n"
                    "No explicit ids were found in the cache to allow mapping to a subset.\n"
                    "Possible fixes: use the original dataset used to create the cache, or regenerate cache for your subset, "
                    "or include an 'id' column in your test DataFrame and a matching 'ids' array in the cache file."
                )
        
        # Convert cached predictions to the expected format
        # Predictions might be stored as binary vectors or class names
        formatted_predictions = []
        for pred in predictions:
            if isinstance(pred, list) and all(isinstance(x, int) for x in pred):
                # Already in binary format
                formatted_predictions.append(pred)
            else:
                # Convert to binary format
                binary_pred = [0] * len(self.label_columns)
                if isinstance(pred, str):
                    if pred in self.label_columns:
                        binary_pred[self.label_columns.index(pred)] = 1
                elif isinstance(pred, list):
                    for label in pred:
                        if label in self.label_columns:
                            binary_pred[self.label_columns.index(label)] = 1
                formatted_predictions.append(binary_pred)
        
        # Calculate metrics if test_df has labels
        metrics = None
        if all(col in test_df.columns for col in self.label_columns):
            true_labels = test_df[self.label_columns].values.tolist()
            metrics = self._calculate_metrics(formatted_predictions, true_labels)
            
            if self.verbose:
                print(f"📊 Metrics from cached predictions:")
                for metric, value in metrics.items():
                    print(f"  • {metric}: {value:.4f}")
        
        # Create result object
        result = self._create_result(
            predictions=formatted_predictions,
            metrics=metrics,
            test_df=test_df,
            train_df=train_df
        )
        
        # Add cache metadata
        if not result.metadata:
            result.metadata = {}
        result.metadata['from_cache'] = True
        result.metadata['cache_file'] = cache_file
        result.metadata['cache_timestamp'] = cached_data.get('timestamp', 'unknown')
        
        return result
    
    def print_cache_status(self, cache_dir: str = "cache") -> None:
        """Print a formatted summary of available cached predictions.
        
        Args:
            cache_dir: Directory to search for cache files
        """
        print("\n" + "="*60)
        print("📦 LLM CACHE STATUS")
        print("="*60)
        
        discovered = self.discover_cached_predictions(cache_dir)
        
        if not discovered:
            print("\n❌ No cached predictions found")
            print(f"   Cache directory: {cache_dir}")
            print("\n💡 TIP: Run predictions with caching enabled to create cache files")
            return
        
        print(f"\n📁 Cache directory: {cache_dir}")
        print(f"✅ Found {sum(len(v) for v in discovered.values())} cached prediction file(s)\n")
        
        for dataset_type, files in discovered.items():
            print(f"\n{dataset_type.upper().replace('_', ' ')}:")
            print("-" * 60)
            
            for i, file_path in enumerate(files[:3], 1):  # Show up to 3 most recent
                summary = self.get_cached_predictions_summary(file_path)
                if summary:
                    print(f"\n  {i}. {Path(file_path).name}")
                    print(f"     Provider: {summary.get('provider', 'unknown')}")
                    print(f"     Model: {summary.get('model', 'unknown')}")
                    print(f"     Predictions: {summary.get('total_predictions', 0)}")
                    print(f"     Timestamp: {summary.get('timestamp', 'unknown')}")
                    
                    if summary.get('has_metrics') and 'metrics' in summary:
                        metrics = summary['metrics']
                        if 'accuracy' in metrics:
                            print(f"     Accuracy: {metrics['accuracy']:.4f}")
                        if 'f1_macro' in metrics:
                            print(f"     F1 (macro): {metrics['f1_macro']:.4f}")
            
            if len(files) > 3:
                print(f"\n  ... and {len(files) - 3} more file(s)")
        
        print("\n" + "="*60)
        print("💡 Use load_cached_predictions_for_dataset() to load a specific file")
        print("="*60 + "\n")