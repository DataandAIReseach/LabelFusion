"""OpenAI-based text classifier."""

import asyncio
from typing import Dict, List, Optional, Union
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError, ConfigurationError, PredictionError
from .base import BaseLLMClassifier


class OpenAIClassifier(BaseLLMClassifier):
    """Text classifier using OpenAI's GPT models."""
    
    def __init__(
        self,
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot",
        enable_cache: bool = True,
        cache_dir: str = "cache/llm",
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True,
        # Cache management parameters
        auto_use_cache: bool = True
    ):
        """Initialize OpenAI classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
            enable_cache: Whether to enable prediction caching (legacy parameter)
            cache_dir: Directory for caching prediction results
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
            auto_use_cache: Whether to automatically check and reuse cached predictions (default: False)
        """
        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='openai',
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results,
            auto_use_cache=auto_use_cache,
            cache_dir=cache_dir
        )
        
        # Handle legacy caching parameter
        self.enable_cache = enable_cache
        
        # Set up classes and prompt engineer configuration
        self.classes_ = label_columns if label_columns else []
        if text_column:
            self.prompt_engineer.text_column = text_column
        if label_columns:
            self.prompt_engineer.label_columns = label_columns
        
        # Set OpenAI specific parameters (these could be passed to the service layer if needed)
        self.model = self.config.parameters.get('model', 'gpt-3.5-turbo')
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)
        
        # No need to create separate client - use the service layer from BaseLLMClassifier
    
    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Predict using OpenAI classifier with explicit results saving.
        
        This method overrides the base class to add comprehensive results saving
        similar to RoBERTa classifier.
        
        Args:
            train_df: Optional training DataFrame for few-shot examples
            test_df: Test DataFrame with text and optionally label columns
            texts: Optional list of texts (alternative to test_df)
            context: Optional context for classification
            label_definitions: Optional label definitions
            
        Returns:
            ClassificationResult with predictions, metrics, and saved files info
        """
        # Store test_df reference for results saving
        if test_df is not None:
            self._current_test_df = test_df
        
        # Call parent prediction method
        result = super().predict(
            train_df=train_df,
            test_df=test_df,
            texts=texts,
            context=context,
            label_definitions=label_definitions
        )
        
        # 🚀 EXPLICIT RESULTS SAVING (like RoBERTa)
        if self.results_manager and hasattr(self, '_current_test_df'):
            try:
                # 1. Save predictions (CSV + JSON)
                saved_files = self.results_manager.save_predictions(
                    result, "test", self._current_test_df
                )
                
                # 2. Save metrics YAML (if available)
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], "test", "openai_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                # 3. Save model configuration YAML
                model_config_dict = {
                    'provider': 'openai',
                    'model_name': self.model,
                    'temperature': self.temperature,
                    'max_completion_tokens': self.max_completion_tokens,
                    'multi_label': self.multi_label,
                    'text_column': self.text_column,
                    'label_columns': self.label_columns,
                    'classes': self.classes_,
                    'batch_size': getattr(self, 'batch_size', 32),
                    'threshold': getattr(self, 'threshold', 0.5),
                    'few_shot_mode': self.few_shot_mode,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label',
                    'enable_cache': self.enable_cache,
                    'cache_dir': self.cache_dir
                }
                
                config_file = self.results_manager.save_model_config(
                    model_config_dict, 
                    "openai_classifier"
                )
                saved_files["config"] = config_file
                
                # 4. Save experiment summary
                experiment_summary = {
                    'model_type': 'llm',
                    'provider': 'openai',
                    'model_name': self.model,
                    'num_labels': len(self.classes_),
                    'classes': self.classes_,
                    'test_samples': len(self._current_test_df),
                    'train_samples': len(train_df) if train_df is not None else 0,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label',
                    'metrics': result.metadata.get('metrics', {}) if result.metadata else {},
                    'temperature': self.temperature,
                    'max_completion_tokens': self.max_completion_tokens,
                    'batch_size': getattr(self, 'batch_size', 32),
                    'completed': True
                }
                
                self.results_manager.save_experiment_summary(experiment_summary)
                
                # 5. Detailed logging (like RoBERTa)
                if getattr(self, 'verbose', True):
                    exp_info = self.results_manager.get_experiment_info()
                    if hasattr(self, 'logger'):
                        self.logger.info(f"📁 OpenAI prediction results saved to: {exp_info['experiment_dir']}")
                        self.logger.info(f"💾 Files saved:")
                        for file_type, file_path in saved_files.items():
                            self.logger.info(f"   - {file_type}: {file_path}")
                
                print(f"📁 OpenAI prediction results saved: {saved_files}")
                
                # 6. Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
                # Clean up temporary reference
                delattr(self, '_current_test_df')
                
            except Exception as e:
                if getattr(self, 'verbose', True):
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Warning: Could not save OpenAI prediction results: {e}")
                print(f"Warning: Could not save OpenAI prediction results: {e}")
                
                # Clean up temporary reference even if saving failed
                if hasattr(self, '_current_test_df'):
                    delattr(self, '_current_test_df')
        
        return result

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt using the service layer.
        
        This uses the llm_generator from BaseLLMClassifier which handles
        API key management and provides a consistent interface.
        """
        try:
            # Use the service layer instead of direct API calls
            response = await self.llm_generator.generate_content(prompt)
            
            # Handle empty or None responses
            if response is None:
                raise APIError("LLM service returned None response")
            
            response = response.strip()
            if not response:
                raise APIError("LLM service returned empty response")
            
            return response
            
        except Exception as e:
            raise APIError(f"LLM service call failed: {str(e)}")
    
    @property
    def model_info(self) -> Dict[str, any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "multi_label": self.multi_label,
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "classes": self.classes_,
            "classification_type": "multi_label" if self.multi_label else "single_label",
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir
        }
