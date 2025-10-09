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
        few_shot_mode: str = "few_shot"
    ):
        """Initialize DeepSeek classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
        """
        # Set provider before calling super().__init__
        config.provider = 'deepseek'
        
        super().__init__(
            config=config,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            label_columns=label_columns
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

