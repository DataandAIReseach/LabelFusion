"""Claude (Anthropic) based text classifier."""

import asyncio
import aiohttp
import json
from typing import Dict, Any

from ..core.exceptions import APIError, ConfigurationError
from .base import BaseLLMClassifier


class ClaudeClassifier(BaseLLMClassifier):
    """Text classifier using Anthropic's Claude models."""
    
    def __init__(self, config):
        """Initialize Claude classifier.
        
        Args:
            config: Model configuration with Claude-specific parameters
        """
        super().__init__(config)
        
        # Validate required configuration
        if not self.config.api_key:
            raise ConfigurationError("Claude API key is required")
        
        # Set default parameters
        self.model = self.config.parameters.get('model', 'claude-3-haiku-20240307')
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.api_base = self.config.api_base or "https://api.anthropic.com"
        
        # Headers for API requests
        self.headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Claude API with the given prompt.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            The model's response
            
        Raises:
            APIError: If the API call fails
        """
        url = f"{self.api_base}/v1/messages"
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add additional parameters if specified
        if 'top_p' in self.config.parameters:
            payload['top_p'] = self.config.parameters['top_p']
        if 'top_k' in self.config.parameters:
            payload['top_k'] = self.config.parameters['top_k']
        if 'system' in self.config.parameters:
            payload['system'] = self.config.parameters['system']
        
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                    async with session.post(url, headers=self.headers, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            # Claude returns content in a different format
                            content = response_data.get('content', [])
                            if content and len(content) > 0:
                                return content[0].get('text', '').strip()
                            else:
                                raise APIError(
                                    "Empty response from Claude",
                                    provider="claude",
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
                                    provider="claude",
                                    status_code=response.status
                                )
                        
                        elif response.status == 401:
                            raise APIError(
                                "Invalid API key",
                                provider="claude",
                                status_code=response.status
                            )
                        
                        elif response.status == 400:
                            error_msg = response_data.get('error', {}).get('message', 'Bad request')
                            raise APIError(
                                f"Bad request: {error_msg}",
                                provider="claude",
                                status_code=response.status
                            )
                        
                        else:
                            error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                            raise APIError(
                                f"API error: {error_msg}",
                                provider="claude",
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
                        provider="claude"
                    )
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        f"Request timeout after {self.config.max_retries} attempts",
                        provider="claude"
                    )
        
        raise APIError(
            f"Failed to get response after {self.config.max_retries} attempts",
            provider="claude"
        )
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get Claude model information."""
        info = super().model_info
        info.update({
            "provider": "claude",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.api_base
        })
        return info

