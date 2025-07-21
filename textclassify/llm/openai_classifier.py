"""OpenAI-based text classifier."""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError, ConfigurationError, PredictionError
from ..prompt_engineer.base import PromptEngineer


class OpenAIClassifier:
    """Text classifier using OpenAI's GPT models."""
    
    def __init__(self, config):
        """Initialize OpenAI classifier."""
        self.config = config
        self.prompt_engineer = PromptEngineer(
            multi_label=False,
            few_shot_mode="few_shot"
        )
        
        # Validate required configuration
        if not self.config.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        # Set parameters
        self.model = self.config.parameters.get('model', 'gpt-3.5-turbo')
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.batch_size = self.config.parameters.get('batch_size', 5)
        self.api_base = "https://api.openai.com/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
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

    async def predict_async(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None
    ) -> ClassificationResult:
        """Make predictions for the given texts."""
        predictions = []
        texts = df[text_column].tolist()
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_predictions = []
            
            for text in batch:
                prompt = f"Classify this text: {text}\nIs this text about technology, environment, or science?"
                try:
                    response = await self._call_llm(prompt)
                    batch_predictions.append(response.strip())
                except Exception as e:
                    raise PredictionError(f"Prediction failed: {str(e)}")
            
            predictions.extend(batch_predictions)
        
        return ClassificationResult(predictions=predictions)

