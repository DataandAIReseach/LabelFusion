"""OpenAI-based text classifier."""

import asyncio
import aiohttp
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
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.api_base = "https://api.openai.com/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt.
        
        This is the only method that needs to be implemented from BaseLLMClassifier.
        All other functionality (prompt engineering, batch processing, etc.) is inherited.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            ) as response:
                if response.status != 200:
                    raise APIError(f"OpenAI API call failed: {await response.text()}")
                result = await response.json()
                return result['choices'][0]['message']['content']

