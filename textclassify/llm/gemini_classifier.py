"""Google Gemini based text classifier."""

import asyncio
import aiohttp
import json
from typing import Dict, Any

from ..core.exceptions import APIError, ConfigurationError
from .base import BaseLLMClassifier


class GeminiClassifier(BaseLLMClassifier):
    """Text classifier using Google's Gemini models."""
    
    def __init__(self, config):
        """Initialize Gemini classifier.
        
        Args:
            config: Model configuration with Gemini-specific parameters
        """
        super().__init__(config)
        
        # Validate required configuration
        if not self.config.api_key:
            raise ConfigurationError("Gemini API key is required")
        
        # Set default parameters
        self.model = self.config.parameters.get('model', 'gemini-1.5-flash')
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_tokens = self.config.parameters.get('max_tokens', 150)
        self.api_base = self.config.api_base or "https://generativelanguage.googleapis.com"
        
        # Gemini uses different parameter names
        self.top_p = self.config.parameters.get('top_p', 0.95)
        self.top_k = self.config.parameters.get('top_k', 40)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Gemini API with the given prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            The model's response
            
        Raises:
            APIError: If the API call fails
        """
        url = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
                "maxOutputTokens": self.max_tokens,
                "candidateCount": 1
            }
        }
        
        # Add safety settings if specified
        if 'safety_settings' in self.config.parameters:
            payload['safetySettings'] = self.config.parameters['safety_settings']
        
        params = {"key": self.config.api_key}
        
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                    async with session.post(url, params=params, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            # Extract text from Gemini response format
                            candidates = response_data.get('candidates', [])
                            if candidates and len(candidates) > 0:
                                content = candidates[0].get('content', {})
                                parts = content.get('parts', [])
                                if parts and len(parts) > 0:
                                    return parts[0].get('text', '').strip()
                                else:
                                    raise APIError(
                                        "No text content in Gemini response",
                                        provider="gemini",
                                        status_code=response.status
                                    )
                            else:
                                # Check for safety filter or other issues
                                if 'promptFeedback' in response_data:
                                    feedback = response_data['promptFeedback']
                                    if feedback.get('blockReason'):
                                        raise APIError(
                                            f"Content blocked by safety filter: {feedback.get('blockReason')}",
                                            provider="gemini",
                                            status_code=response.status
                                        )
                                
                                raise APIError(
                                    "Empty response from Gemini",
                                    provider="gemini",
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
                                    provider="gemini",
                                    status_code=response.status
                                )
                        
                        elif response.status == 400:
                            error_msg = response_data.get('error', {}).get('message', 'Bad request')
                            raise APIError(
                                f"Bad request: {error_msg}",
                                provider="gemini",
                                status_code=response.status
                            )
                        
                        elif response.status == 403:
                            raise APIError(
                                "Invalid API key or insufficient permissions",
                                provider="gemini",
                                status_code=response.status
                            )
                        
                        else:
                            error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                            raise APIError(
                                f"API error: {error_msg}",
                                provider="gemini",
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
                        provider="gemini"
                    )
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise APIError(
                        f"Request timeout after {self.config.max_retries} attempts",
                        provider="gemini"
                    )
        
        raise APIError(
            f"Failed to get response after {self.config.max_retries} attempts",
            provider="gemini"
        )
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get Gemini model information."""
        info = super().model_info
        info.update({
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "api_base": self.api_base
        })
        return info

