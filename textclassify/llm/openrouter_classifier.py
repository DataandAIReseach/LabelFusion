"""OpenRouter based text classifier.

OpenRouter (https://openrouter.ai) exposes many LLM providers (OpenAI, Anthropic,
Google, Meta, Mistral, DeepSeek, ...) behind a single OpenAI-compatible API. The
request itself is sent by OpenRouterContentGenerator in
services/llm_content_generator.py.

API key: set OPENROUTER_API_KEY (keys look like sk-or-v1-...), e.g. in the .env file.

Model names use OpenRouter's "<vendor>/<model>" format, e.g.:
    "openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat"
"""

from typing import Dict, List, Optional, Any
import pandas as pd

from ..core.types import ClassificationResult
from ..core.exceptions import APIError
from .base import BaseLLMClassifier


class OpenRouterClassifier(BaseLLMClassifier):
    """Text classifier using any model available on OpenRouter."""

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
        use_nearest_neighbours: bool = False,              # UK spelling
        embedding_model: str = "all-MiniLM-L6-v2",
        use_nearest_neighbors: Optional[bool] = None,       # US alias
        fixed_examples: bool = False
    ):
        """Initialize OpenRouter classifier.

        Args:
            config: Configuration object; config.parameters["model"] must be an
                OpenRouter model id such as "openai/gpt-4o-mini"
            text_column: Name of the column containing text data
            label_columns: List of column names containing labels
            multi_label: Whether this is a multi-label classifier
            few_shot_mode: "zero_shot", "one_shot", "few_shot", "full_coverage", or an
                int to control how many few-shot examples are drawn per prompt
            output_dir: Base directory for saving results (default: "outputs")
            experiment_name: Name for this experiment (default: auto-generated)
            auto_save_results: Whether to automatically save results (default: True)
            auto_use_cache: Whether to automatically check and reuse cached predictions (default: True)
            cache_dir: Directory to search for cached predictions (default: "cache")
            use_nearest_neighbours: Enable nearest-neighbour few-shot sampling
            embedding_model: Embedding model used by nearest-neighbour sampler
            use_nearest_neighbors: US spelling alias for use_nearest_neighbours
            fixed_examples: If True, sample the few-shot examples once and reuse the
                same set for every prompt. If False (default), resample per test row.
        """
        # Normalize US/UK spelling
        if use_nearest_neighbors is not None:
            use_nearest_neighbours = use_nearest_neighbors

        # Set provider before calling super().__init__
        config.provider = 'openrouter'

        super().__init__(
            config=config,
            text_column=text_column,
            label_columns=label_columns,
            multi_label=multi_label,
            few_shot_mode=few_shot_mode,
            provider='openrouter',
            output_dir=output_dir,
            experiment_name=experiment_name,
            auto_save_results=auto_save_results,
            auto_use_cache=auto_use_cache,
            cache_dir=cache_dir,
            use_nearest_neighbours=use_nearest_neighbours,
            embedding_model=embedding_model,
            fixed_examples=fixed_examples,
        )

        # Set up classes and prompt engineer configuration
        self.classes_ = label_columns if label_columns else []
        if text_column:
            self.prompt_engineer.text_column = text_column
        if label_columns:
            self.prompt_engineer.label_columns = label_columns

        # Set OpenRouter specific parameters
        self.model = self.config.parameters.get('model', 'openai/gpt-4o-mini')
        self.temperature = self.config.parameters.get('temperature', 1)
        self.max_completion_tokens = self.config.parameters.get('max_completion_tokens', 150)

        # Mode tracking (train/val/test) - inherited from base but can be set here too
        if not hasattr(self, 'mode'):
            self.mode = None

    def predict(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        texts: Optional[List[str]] = None,
        context: Optional[str] = None,
        label_definitions: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """Predict using OpenRouter classifier with explicit results saving."""
        # Keep externally configured mode (e.g., train/val/test) intact.
        # BaseLLMClassifier will still default to "test" if mode is unset.

        # Store test_df reference for results saving
        if test_df is not None:
            self._current_test_df = test_df

        # Call parent prediction method
        result = super().predict(
            train_df=train_df,
            test_df=test_df,
            texts=texts,
            context=context,
            label_definitions=label_definitions
        )

        #  EXPLICIT RESULTS SAVING (like RoBERTa)
        if self.results_manager:
            dataset_type = getattr(self, '_current_dataset_type', None) or self.mode or 'test'
            current_df = getattr(self, '_current_test_df', None)

            if current_df is not None:
                try:
                    saved_files = self.results_manager.save_predictions(
                        result, dataset_type, current_df
                    )

                    # Save metrics YAML
                    if hasattr(result, 'metadata') and result.metadata and 'metrics' in result.metadata:
                        metrics_file = self.results_manager.save_metrics(
                            result.metadata['metrics'], dataset_type, "openrouter_classifier"
                        )
                        saved_files["metrics"] = metrics_file

                    # Save model configuration
                    model_config_dict = {
                        'provider': 'openrouter',
                        'model_name': self.model,
                        'temperature': self.temperature,
                        'max_completion_tokens': self.max_completion_tokens,
                        'multi_label': self.multi_label,
                        'text_column': self.text_column,
                        'label_columns': self.label_columns,
                        'classes': self.classes_,
                        'few_shot_mode': self.few_shot_mode,
                        'classification_type': 'multi_label' if self.multi_label else 'single_label',
                        'cache_dir': self.cache_dir
                    }

                    config_file = self.results_manager.save_model_config(
                        model_config_dict, "openrouter_classifier"
                    )
                    saved_files["config"] = config_file

                    # Save experiment summary
                    experiment_summary = {
                        'model_type': 'llm',
                        'provider': 'openrouter',
                        'model_name': self.model,
                        'num_labels': len(self.classes_),
                        'classes': self.classes_,
                        'test_samples': len(self._current_test_df),
                        'train_samples': len(train_df) if train_df is not None else 0,
                        'few_shot_mode': self.few_shot_mode,
                        'classification_type': 'multi_label' if self.multi_label else 'single_label',
                        'metrics': result.metadata.get('metrics', {}) if result.metadata else {},
                        'completed': True
                    }

                    self.results_manager.save_experiment_summary(experiment_summary)

                    print(f" OpenRouter prediction results saved: {saved_files}")

                    # Add file paths to result metadata
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata['saved_files'] = saved_files

                except Exception as e:
                    print(f"Warning: Could not save OpenRouter prediction results: {e}")

        return result

    async def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter API with the given prompt using the service layer.

        This uses the llm_generator from BaseLLMClassifier which handles
        API key management and provides a consistent interface.
        """
        try:
            # Use the service layer instead of direct API calls
            response = await self.llm_generator.generate_content(prompt)

            # Handle empty or None responses
            if response is None:
                raise APIError("LLM service returned None response")

            response = response.strip()
            if not response:
                raise APIError("LLM service returned empty response")

            return response

        except Exception as e:
            raise APIError(f"LLM service call failed: {str(e)}")

    @property
    def model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information."""
        info = super().model_info
        info.update({
            "provider": "openrouter",
            "model": self.model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "few_shot_mode": self.few_shot_mode
        })
        return info
