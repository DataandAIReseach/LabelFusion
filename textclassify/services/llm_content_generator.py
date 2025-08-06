from abc import ABC, abstractmethod
from typing import Optional
import openai
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
        import google.generativeai as genai
        
        # Get API key from parameter or environment variable
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        try:
            # Combine role prompt and main prompt if role prompt is provided
            complete_prompt = f"{role_prompt}\n\n{prompt}" if role_prompt else prompt
            
            # Generate content - Gemini's generate_content is synchronous, so we need to run it in executor
            import asyncio
            import concurrent.futures
            
            def _generate_sync():
                response = self.model.generate_content(complete_prompt)
                return response.text
            
            # Run the synchronous call in a thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, _generate_sync)
            
            return result.strip() if result else ""
            
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