from abc import ABC, abstractmethod
from typing import Optional
import openai
from google.generativeai import GenerativeModel
from openai import OpenAI
from ..core.exceptions import APIError
import os

class BaseLLMContentGenerator(ABC):
    """Abstract base class for LLM content generation."""
    
    @abstractmethod
    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        """Generate content using the LLM.
        
        Args:
            prompt: Main prompt text
            role_prompt: Optional system/role prompt
            
        Returns:
            Generated content from LLM
            
        Raises:
            APIError: If LLM API call fails
        """
        pass

class OpenAIContentGenerator(BaseLLMContentGenerator):
    """Content generator using OpenAI's API."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        """Generate content for a single prompt."""
        try:
            messages = []
            if role_prompt:
                messages.append({"role": "system", "content": role_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise APIError(f"OpenAI API call failed: {str(e)}")

class GeminiContentGenerator(BaseLLMContentGenerator):
    """Content generator using Google's Gemini API."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model = GenerativeModel(model_name)
        if api_key:
            self.model.configure(api_key=api_key)

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        try:
            complete_prompt = f"{role_prompt}\n\n{prompt}" if role_prompt else prompt
            response = await self.model.generate_content_async(complete_prompt)
            return response.text
        except Exception as e:
            raise APIError(f"Gemini API call failed: {str(e)}")

class DeepseekContentGenerator(BaseLLMContentGenerator):
    """Content generator using Deepseek's API."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        # Initialize DeepSeek client with OpenAI-compatible SDK
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if role_prompt:
                messages.append({"role": "system", "content": role_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise APIError(f"Deepseek API call failed: {str(e)}")

def create_llm_generator(provider: str, model_name: str, api_key: Optional[str] = None) -> BaseLLMContentGenerator:
    """Factory function to create appropriate LLM generator.
    
    Args:
        provider: LLM provider ('openai', 'gemini', or 'deepseek')
        model_name: Name of the model to use
        api_key: Optional API key
        
    Returns:
        Configured LLM content generator
    """
    generators = {
        'openai': OpenAIContentGenerator,
        'gemini': GeminiContentGenerator,
        'deepseek': DeepseekContentGenerator
    }
    
    if provider not in generators:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(generators.keys())}")
        
    return generators[provider](model_name, api_key)