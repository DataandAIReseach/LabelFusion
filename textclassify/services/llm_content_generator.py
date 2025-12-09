from abc import ABC, abstractmethod
from typing import Optional
import os

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ModuleNotFoundError:
    OPENAI_AVAILABLE = False

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
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK not available in this environment")
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
        try:
            import google.generativeai as genai
        except ModuleNotFoundError:
            raise RuntimeError("Gemini SDK (google.generativeai) not available in this environment")
        
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
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI SDK not available (required for Deepseek compatibility)")
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
            
            # DeepSeek's OpenAI-compatible API is synchronous like OpenAI
            response = self.client.chat.completions.create(
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
    # Lightweight dummy generator (used for smoke tests or when SDKs are explicitly disabled)
    class LocalDummyContentGenerator(BaseLLMContentGenerator):
        def __init__(self, model_name: str, api_key: Optional[str] = None):
            self.model_name = model_name

        async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
            """Return a dataset-sampled response for smoke tests.

            Strategy:
            - If the prompt contains labelled examples (lines with 'Labels:' or 'Ratings:'),
              extract those and return one of the label lines (randomly sampled).
            - Otherwise, if the prompt contains explicit 'Text:' lines, return one of those texts.
            - Fallback: return a fixed placeholder.

            This keeps the dummy lightweight while producing outputs that are tied to the
            dataset/examples embedded in the prompt, making smoke metrics more meaningful.
            """
            import re
            import random

            if not prompt:
                return "[DUMMY_LLM_RESPONSE]"

            # Try to find labelled example lines (common templates use 'Labels:' or 'Ratings:')
            label_patterns = re.findall(r"(?:Labels|Ratings)[:\s]*([^\n\r]+)", prompt, flags=re.IGNORECASE)
            if label_patterns:
                # Return one of the found label lines (trimmed)
                choice = random.choice(label_patterns).strip()
                # If labels are given as binary vectors (e.g. '1 | 0 | 0'), return as-is
                return choice

            # If no explicit label lines, look for example text lines and return one text sample
            text_patterns = re.findall(r"(?:Text\s*\d*[:]?)\s*([^\n\r]+)", prompt, flags=re.IGNORECASE)
            if text_patterns:
                return random.choice([t.strip() for t in text_patterns if t.strip()])

            # As a last resort, return a short deterministic placeholder
            return "[DUMMY_LLM_RESPONSE]"

    # Honor environment variable to force dummy LLM (avoids any external API calls)
    if os.environ.get("USE_DUMMY_LLM", "").lower() in ("1", "true", "yes"):
        return LocalDummyContentGenerator(model_name, api_key)
    generators = {
        'openai': OpenAIContentGenerator if OPENAI_AVAILABLE else LocalDummyContentGenerator,
        'gemini': GeminiContentGenerator,
        'deepseek': DeepseekContentGenerator
    }

    if provider not in generators:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(generators.keys())}")

    generator_cls = generators[provider]
    return generator_cls(model_name, api_key)