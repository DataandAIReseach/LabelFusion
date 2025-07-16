from abc import ABC, abstractmethod
from typing import Optional
import openai
from google.generativeai import GenerativeModel
import deepseek
from ..core.exceptions import APIError

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
        if api_key:
            openai.api_key = api_key

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        try:
            messages = []
            if role_prompt:
                messages.append({"role": "system", "content": role_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
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
        if api_key:
            deepseek.api_key = api_key

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        try:
            complete_prompt = f"{role_prompt}\n\n{prompt}" if role_prompt else prompt
            response = await deepseek.Completion.create(
                model=self.model_name,
                prompt=complete_prompt,
                temperature=0.0
            )
            return response.choices[0].text
        except Exception as e:
            raise APIError(f"Deepseek API call failed: {str(e)}")

def create_llm_generator(provider: str, model_name: str, api_key: str) -> BaseLLMContentGenerator:
    """Create LLM generator based on provider.
    
    Args:
        provider: Name of LLM provider ('openai', 'gemini', 'deepseek')
        model_name: Name of the model to use
        api_key: API key for the provider
        
    Returns:
        Initialized LLM content generator
        
    Raises:
        ValueError: If provider is not supported
    """
    generators = {
        'openai': OpenAIContentGenerator,
        'gemini': GeminiContentGenerator,
        'deepseek': DeepseekContentGenerator
    }
    
    if provider not in generators:
        supported = list(generators.keys())
        raise ValueError(f"Provider '{provider}' not supported. Use one of: {supported}")
        
    generator_class = generators[provider]
    return generator_class(model_name=model_name, api_key=api_key)