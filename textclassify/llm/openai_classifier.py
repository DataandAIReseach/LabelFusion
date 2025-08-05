"""OpenAI-based text classifier."""

import asyncio
from typing import Dict, List, Optional, Union
import pandas as pd
from openai import OpenAI

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
        few_shot_mode: str = "few_shot"
    ):
        """Initialize OpenAI classifier.
        
        Args:
            config: Configuration object containing API keys and parameters
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: Mode for few-shot learning
        """
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
        
        # Validate OpenAI configuration
        if not self.config.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        # Set OpenAI specific parameters
        self.model = self.config.parameters.get('model', 'gpt-3.5-turbo')
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.api_key)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt.
        
        This is the only method that needs to be implemented from BaseLLMClassifier.
        All other functionality (prompt engineering, batch processing, etc.) is inherited.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Handle empty or None responses
            if content is None:
                raise APIError("OpenAI returned None response content")
            
            content = content.strip()
            if not content:
                raise APIError("OpenAI returned empty response content")
            
            return content
            
        except Exception as e:
            raise APIError(f"OpenAI API call failed: {str(e)}")

