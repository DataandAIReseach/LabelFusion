"""OpenAI-based text classifier."""

import asyncio
import aiohttp
import json
from typing import Dict, Any

from ..core.exceptions import APIError, ConfigurationError
from .base import BaseLLMClassifier


class OpenAIClassifier(BaseLLMClassifier):
    """Text classifier using OpenAI's GPT models."""
    
    def __init__(self, config):
        """Initialize OpenAI classifier.
        
        Args:
            config: Model configuration with OpenAI-specific parameters
        """
        super().__init__(config)
        
        # Validate required configuration
        if not self.config.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        # Set default parameters
        self.model = self.config.parameters.get('model', 'gpt-3.5-turbo')
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.api_base = self.config.api_base or "https://api.openai.com/v1"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            The model's response
            
        Raises:
            APIError: If the API call fails
        """
        url = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add additional parameters if specified
        if 'top_p' in self.config.parameters:
            payload['top_p'] = self.config.parameters['top_p']
        if 'frequency_penalty' in self.config.parameters:
            payload['frequency_penalty'] = self.config.parameters['frequency_penalty']
        if 'presence_penalty' in self.config.parameters:
            payload['presence_penalty'] = self.config.parameters['presence_penalty']
        
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                    async with session.post(url, headers=self.headers, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            return response_data['choices'][0]['message']['content'].strip()
                        
                        elif response.status == 429:  # Rate limit
                            if attempt < self.config.max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise APIError(
                                    f"Rate limit exceeded after {self.config.max_retries} attempts",
                                    provider="openai",
                                    status_code=response.status
                                )
                        
                        elif response.status == 401:
                            raise APIError(
                                "Invalid API key",
                                provider="openai",
                                status_code=response.status
                            )
                        
                        elif response.status == 400:
                            error_msg = response_data.get('error', {}).get('message', 'Bad request')
                            raise APIError(
                                f"Bad request: {error_msg}",
                                provider="openai",
                                status_code=response.status
                            )
                        
                        else:
                            error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                            raise APIError(
                                f"API error: {error_msg}",
                                provider="openai",
                                status_code=response.status
                            )
            
            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        f"Network error after {self.config.max_retries} attempts: {str(e)}",
                        provider="openai"
                    )
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        f"Request timeout after {self.config.max_retries} attempts",
                        provider="openai"
                    )
        
        raise APIError(
            f"Failed to get response after {self.config.max_retries} attempts",
            provider="openai"
        )
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        info = super().model_info
        info.update({
            "provider": "openai",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        })
        return info

