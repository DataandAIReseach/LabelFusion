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
        cache_dir: str = "cache/llm"
    ):
        """Initialize OpenAI classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
            enable_cache: Whether to enable prediction caching
            cache_dir: Directory for caching prediction results
        """
        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='openai'
        )
        
        # Handle caching parameters locally since BaseLLMClassifier doesn't support them
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
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
