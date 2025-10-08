"""Google Gemini based text classifier."""

import asyncio
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError, ConfigurationError, PredictionError
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
        enable_cache: bool = True,
        cache_dir: str = "cache/llm"
    ):
        """Initialize Gemini classifier.
        
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
            provider='gemini',
            enable_cache=enable_cache,
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
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)
        
        # Gemini-specific parameters
        # Gemini-specific parameters
        self.top_p = self.config.parameters.get('top_p', 0.95)
        self.top_k = self.config.parameters.get('top_k', 40)
    
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
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k
            "top_k": self.top_k
        })
        return info

