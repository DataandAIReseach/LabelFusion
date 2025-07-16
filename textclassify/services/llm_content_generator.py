from typing import Optional
import openai
from ..core.exceptions import APIError

class LLMContentGenerator:
    """Service for generating content through LLM API interactions."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key

    async def generate_content(self, prompt: str, role_prompt: Optional[str] = None) -> str:
        """Generate content from LLM based on prompt.
        
        Args:
            prompt: The main prompt to send to LLM
            role_prompt: Optional system role prompt
            
        Returns:
            Generated content from LLM
            
        Raises:
            APIError: If LLM API call fails
        """
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
            raise APIError(f"LLM content generation failed: {str(e)}")