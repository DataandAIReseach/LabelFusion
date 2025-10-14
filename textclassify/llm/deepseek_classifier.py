"""DeepSeek based text classifier."""

import asyncio
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError, ConfigurationError, PredictionError
from .base import BaseLLMClassifier


class DeepSeekClassifier(BaseLLMClassifier):
    """Text classifier using DeepSeek models."""
    
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
        auto_save_results: bool = True
    ):
        """Initialize DeepSeek classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
        """
        # Set provider before calling super().__init__
        config.provider = 'deepseek'
        
        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='deepseek',
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results
        )
        
        # Set up classes and prompt engineer configuration
        self.classes_ = label_columns if label_columns else []
        if text_column:
            self.prompt_engineer.text_column = text_column
        if label_columns:
            self.prompt_engineer.label_columns = label_columns
        
        # Set DeepSeek specific parameters
        self.model = self.config.parameters.get('model', 'deepseek-chat')
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)
        
        # DeepSeek-specific parameters (similar to OpenAI)
        self.top_p = self.config.parameters.get('top_p', 1.0)
        self.frequency_penalty = self.config.parameters.get('frequency_penalty', 0.0)
        self.presence_penalty = self.config.parameters.get('presence_penalty', 0.0)
    
    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Predict using DeepSeek classifier with explicit results saving."""
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
                saved_files = self.results_manager.save_predictions(
                    result, "test", self._current_test_df
                )
                
                # Save metrics YAML
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], "test", "deepseek_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                # Save model configuration
                model_config_dict = {
                    'provider': 'deepseek',
                    'model_name': self.model,
                    'temperature': self.temperature,
                    'max_completion_tokens': self.max_completion_tokens,
                    'top_p': self.top_p,
                    'frequency_penalty': self.frequency_penalty,
                    'presence_penalty': self.presence_penalty,
                    'multi_label': self.multi_label,
                    'text_column': self.text_column,
                    'label_columns': self.label_columns,
                    'classes': self.classes_,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label'
                }
                
                config_file = self.results_manager.save_model_config(
                    model_config_dict, "deepseek_classifier"
                )
                saved_files["config"] = config_file
                
                # Save experiment summary
                experiment_summary = {
                    'model_type': 'llm',
                    'provider': 'deepseek',
                    'model_name': self.model,
                    'test_samples': len(self._current_test_df),
                    'train_samples': len(train_df) if train_df is not None else 0,
                    'classification_type': 'multi_label' if self.multi_label else 'single_label',
                    'metrics': result.metadata.get('metrics', {}) if result.metadata else {},
                    'completed': True
                }
                
                self.results_manager.save_experiment_summary(experiment_summary)
                
                print(f"📁 DeepSeek prediction results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
                # Clean up temporary reference
                delattr(self, '_current_test_df')
                
            except Exception as e:
                print(f"Warning: Could not save DeepSeek prediction results: {e}")
                if hasattr(self, '_current_test_df'):
                    delattr(self, '_current_test_df')
        
        return result
    
    async def _call_llm(self, prompt: str) -> str:
        """Call DeepSeek API with the given prompt using the service layer.
        
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
    
    def _predict_texts_internal(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Internal method to predict labels for a list of texts.
        
        This method provides compatibility with RoBERTa classifier interface while using
        the async LLM prediction pipeline.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        # Convert texts to DataFrame format expected by async predict method
        import pandas as pd
        test_df = pd.DataFrame({self.text_column: texts})
        
        # Add dummy labels if true_labels provided (for compatibility)
        if true_labels is not None and self.label_columns:
            for i, label_col in enumerate(self.label_columns):
                if i < len(true_labels[0]) if true_labels else 0:
                    test_df[label_col] = [labels[i] if i < len(labels) else 0 for labels in true_labels]
                else:
                    test_df[label_col] = 0
        
        # Store test_df reference for results saving
        if self.results_manager:
            self._current_test_df = test_df
        
        # Call the async prediction method
        result = self.predict(test_df=test_df)
        
        # Extract predictions in the format expected by RoBERTa-style interface
        if hasattr(result, 'predictions'):
            # Convert predictions to the format expected by downstream components
            if self.multi_label:
                # For multi-label: predictions should be list of lists of class names
                formatted_predictions = result.predictions
            else:
                # For single-label: predictions should be list of class names
                formatted_predictions = result.predictions
            
            # Create new result with formatted predictions
            result.predictions = formatted_predictions
        
        # Calculate metrics if true labels are provided
        if true_labels is not None:
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
                            binary_pred[0] = 1
                        predicted_labels.append(binary_pred)
                
                # Calculate metrics using the parent class method from BaseLLMClassifier
                metrics = self._calculate_metrics(predicted_labels, true_labels)
                
                # Add metrics to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['metrics'] = metrics
                
            except Exception as e:
                if getattr(self, 'verbose', True):
                    print(f"Warning: Could not calculate metrics: {e}")
        
        # Save prediction results using ResultsManager (if it's the main predict call)
        if self.results_manager and hasattr(self, '_current_test_df'):
            try:
                saved_files = self.results_manager.save_predictions(
                    result, "test", self._current_test_df
                )
                
                # Save metrics if available
                if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                    metrics_file = self.results_manager.save_metrics(
                        result.metadata['metrics'], "test", "deepseek_classifier"
                    )
                    saved_files["metrics"] = metrics_file
                
                print(f"📁 DeepSeek prediction results saved: {saved_files}")
                
                # Add file paths to result metadata
                if not result.metadata:
                    result.metadata = {}
                result.metadata['saved_files'] = saved_files
                
                # Clean up temporary reference
                delattr(self, '_current_test_df')
                
            except Exception as e:
                print(f"Warning: Could not save DeepSeek prediction results: {e}")
                
                # Clean up temporary reference even if saving failed
                if hasattr(self, '_current_test_df'):
                    delattr(self, '_current_test_df')
        
        return result
    
    def predict_texts(self, texts: List[str], true_labels: Optional[List[List[int]]] = None) -> ClassificationResult:
        """Predict labels for a list of texts (compatibility method for FusionEnsemble).
        
        This method is provided for compatibility with FusionEnsemble which calls 
        LLM models with text lists. For regular usage, use predict(test_df) instead.
        
        Args:
            texts: List of texts to classify
            true_labels: Optional true labels in binary format for evaluation metrics
            
        Returns:
            ClassificationResult with predictions and optional metrics
        """
        return self._predict_texts_internal(texts, true_labels)
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get DeepSeek model information."""
        info = super().model_info
        info.update({
            "provider": "deepseek",
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        })
        return info