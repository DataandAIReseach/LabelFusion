"""DeepSeek based text classifier."""

import asyncio
import aiohttp
import json
from typing import Dict, Any

from ..core.exceptions import APIError, ConfigurationError
from .base import BaseLLMClassifier


class DeepSeekClassifier(BaseLLMClassifier):
    """Text classifier using DeepSeek models."""
    
    def __init__(self, config):
        """Initialize DeepSeek classifier.
        
        Args:
            config: Model configuration with DeepSeek-specific parameters
        """
        super().__init__(config)
        
        # Validate required configuration
        if not self.config.api_key:
            raise ConfigurationError("DeepSeek API key is required")
        
        # Set default parameters
        self.model = self.config.parameters.get('model', 'deepseek-chat')
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.api_base = self.config.api_base or "https://api.deepseek.com"
        
        # Headers for API requests (DeepSeek uses OpenAI-compatible API)
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call DeepSeek API with the given prompt.
        
        Args:
            prompt: The prompt to send to DeepSeek
            
        Returns:
            The model's response
            
        Raises:
            APIError: If the API call fails
        """
        url = f"{self.api_base}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        # Add additional parameters if specified
        if 'top_p' in self.config.parameters:
            payload['top_p'] = self.config.parameters['top_p']
        if 'frequency_penalty' in self.config.parameters:
            payload['frequency_penalty'] = self.config.parameters['frequency_penalty']
        if 'presence_penalty' in self.config.parameters:
            payload['presence_penalty'] = self.config.parameters['presence_penalty']
        if 'stop' in self.config.parameters:
            payload['stop'] = self.config.parameters['stop']
        
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                    async with session.post(url, headers=self.headers, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            choices = response_data.get('choices', [])
                            if choices and len(choices) > 0:
                                message = choices[0].get('message', {})
                                content = message.get('content', '').strip()
                                if content:
                                    return content
                                else:
                                    raise APIError(
                                        "Empty content in DeepSeek response",
                                        provider="deepseek",
                                        status_code=response.status
                                    )
                            else:
                                raise APIError(
                                    "No choices in DeepSeek response",
                                    provider="deepseek",
                                    status_code=response.status
                                )
                        
                        elif response.status == 429:  # Rate limit
                            if attempt < self.config.max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise APIError(
                                    f"Rate limit exceeded after {self.config.max_retries} attempts",
                                    provider="deepseek",
                                    status_code=response.status
                                )
                        
                        elif response.status == 401:
                            raise APIError(
                                "Invalid API key",
                                provider="deepseek",
                                status_code=response.status
                            )
                        
                        elif response.status == 400:
                            error_msg = response_data.get('error', {}).get('message', 'Bad request')
                            raise APIError(
                                f"Bad request: {error_msg}",
                                provider="deepseek",
                                status_code=response.status
                            )
                        
                        else:
                            error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                            raise APIError(
                                f"API error: {error_msg}",
                                provider="deepseek",
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
                        provider="deepseek"
                    )
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        f"Request timeout after {self.config.max_retries} attempts",
                        provider="deepseek"
                    )
        
        raise APIError(
            f"Failed to get response after {self.config.max_retries} attempts",
            provider="deepseek"
        )
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get DeepSeek model information."""
        info = super().model_info
        info.update({
            "provider": "deepseek",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        })
        return info

