"""Google Gemini based text classifier."""

import asyncio
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError, ConfigurationError, PredictionError
from .base import BaseLLMClassifier


class GeminiClassifier(BaseLLMClassifier):
    """Text classifier using Google's Gemini models."""
    
    def __init__(
        self,
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot",
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True,
        # Cache management parameters
        auto_use_cache: bool = False,
        cache_dir: str = "cache"
    ):
        """Initialize Gemini classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
            auto_use_cache: Whether to automatically check and reuse cached predictions (default: False)
            cache_dir: Directory to search for cached predictions (default: "cache")
        """
        # Set provider before calling super().__init__
        config.provider = 'gemini'
        
        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='gemini',
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results,
            auto_use_cache=auto_use_cache,
            cache_dir=cache_dir
        )
        
        # Set up classes and prompt engineer configuration
        self.classes_ = label_columns if label_columns else []
        if text_column:
            self.prompt_engineer.text_column = text_column
        if label_columns:
            self.prompt_engineer.label_columns = label_columns
        
        # Set Gemini specific parameters
        self.model = self.config.parameters.get('model', 'gemini-1.5-flash')
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)
        
        # Gemini-specific parameters
        self.top_p = self.config.parameters.get('top_p', 0.95)
        self.top_k = self.config.parameters.get('top_k', 40)
    
    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Predict using Gemini classifier with explicit results saving.
        
        This method overrides the base class to add comprehensive results saving
        similar to RoBERTa classifier.
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
                # Save predictions and configuration
                saved_files = self.results_manager.save_predictions(
                    result, "test", self._current_test_df
                )
                
                # Save metrics YAML
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], "test", "gemini_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                # Save model configuration
                model_config_dict = {
                    'provider': 'gemini',
                    'model_name': self.model,
                    'temperature': self.temperature,
                    'max_completion_tokens': self.max_completion_tokens,
                    'top_p': self.top_p,
                    'top_k': self.top_k,
                    'multi_label': self.multi_label,
                    'text_column': self.text_column,
                    'label_columns': self.label_columns,
                    'classes': self.classes_,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label'
                }
                
                config_file = self.results_manager.save_model_config(
                    model_config_dict, "gemini_classifier"
                )
                saved_files["config"] = config_file
                
                # Save experiment summary
                experiment_summary = {
                    'model_type': 'llm',
                    'provider': 'gemini',
                    'model_name': self.model,
                    'test_samples': len(self._current_test_df),
                    'train_samples': len(train_df) if train_df is not None else 0,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label',
                    'metrics': result.metadata.get('metrics', {}) if result.metadata else {},
                    'completed': True
                }
                
                self.results_manager.save_experiment_summary(experiment_summary)
                
                print(f"📁 Gemini prediction results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
                # Clean up temporary reference
                delattr(self, '_current_test_df')
                
            except Exception as e:
                print(f"Warning: Could not save Gemini prediction results: {e}")
                if hasattr(self, '_current_test_df'):
                    delattr(self, '_current_test_df')
        
        return result
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini API with the given prompt using the service layer.
        
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
    def model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        info = super().model_info
        info.update({
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k
        })
        return info