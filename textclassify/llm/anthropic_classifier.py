"""Anthropic (Claude) based text classifier, using the Anthropic API directly.

The request itself is sent by AnthropicContentGenerator in services/llm_content_generator.py.

API key: set ANTHROPIC_API_KEY (keys look like sk-ant-...; CLAUDE_API_KEY works too), e.g. in the .env file.
Model names are Anthropic's own ids, e.g. "claude-haiku-5-5" or "claude-sonnet-5-5". (Claude models are also
available through OpenRouterClassifier with an OpenRouter key and ids like "anthropic/claude-sonnet-4.5".)
"""

from typing import Dict, List, Optional
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError
from .base import BaseLLMClassifier


class AnthropicClassifier(BaseLLMClassifier):
    """Text classifier using Claude models through the Anthropic API."""

    def __init__(
        self,
        config,
        text_column: str = 'text',
        label_columns: Optional[List[str]] = None,
        multi_label: bool = False,
        few_shot_mode: str = "few_shot",
        # Results management parameters
        output_dir: str = "outputs",
        experiment_name: Optional[str] = None,
        auto_save_results: bool = True,
        # Cache management parameters
        auto_use_cache: bool = True,
        cache_dir: str = "cache",
        # Nearest-neighbour sampling parameters
        use_nearest_neighbours: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_nearest_neighbors: Optional[bool] = None,
        fixed_examples: bool = False,
    ):
        """Initialize the Anthropic classifier. The arguments are those of OpenRouterClassifier;
        config.parameters["model"] must be an Anthropic model id such as "claude-haiku-5-5"."""
        if use_nearest_neighbors is not None:
            use_nearest_neighbours = use_nearest_neighbors

        config.provider = 'anthropic'

        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='anthropic',
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results,
            auto_use_cache=auto_use_cache,
            cache_dir=cache_dir,
            use_nearest_neighbours=use_nearest_neighbours,
            embedding_model=embedding_model,
            fixed_examples=fixed_examples,
        )

        self.classes_ = label_columns if label_columns else []
        if text_column:
            self.prompt_engineer.text_column = text_column
        if label_columns:
            self.prompt_engineer.label_columns = label_columns

        self.model = self.config.parameters.get('model', 'claude-haiku-5-5')
        self.temperature = self.config.parameters.get('temperature', 0.1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 1024)
        # the generator (built by the base class) sends the sampling parameters of the config
        self.llm_generator.temperature = self.temperature
        self.llm_generator.max_tokens = self.max_completion_tokens

        if not hasattr(self, 'mode'):
            self.mode = None

    async def _call_llm(self, prompt: str) -> str:
        """Call the Anthropic API with the given prompt using the service layer."""
        try:
            response = await self.llm_generator.generate_content(prompt)
            if response is None:
                raise APIError("LLM service returned None response")
            response = response.strip()
            if not response:
                raise APIError("LLM service returned empty response")
            return response
        except Exception as e:
            raise APIError(f"LLM service call failed: {str(e)}")

    @property
    def model_info(self) -> Dict:
        info = super().model_info
        info.update({
            "provider": "anthropic",
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "few_shot_mode": self.few_shot_mode,
        })
        return info
