"""OpenRouter-based text classifier.

OpenRouter (https://openrouter.ai) exposes many LLM providers (OpenAI, Anthropic,
Google, Meta, Mistral, DeepSeek, ...) behind a single OpenAI-compatible API, so
this classifier reuses OpenAIClassifier and only swaps the provider. The request
itself is sent by OpenRouterContentGenerator in services/llm_content_generator.py.

API key: set OPENROUTER_API_KEY (keys look like sk-or-v1-...).

Model names use OpenRouter's "<vendor>/<model>" format, e.g.:
    "openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat"

Usage:
    config = ModelConfig(
        model_name="openai/gpt-4o-mini",
        model_type=ModelType.LLM,
        parameters={"model": "openai/gpt-4o-mini", "temperature": 0.0},
    )
    clf = OpenRouterClassifier(config, text_column="text", label_columns=labels)
    clf.fit(train_df)
    result = clf.predict(test_df=test_df)
"""

from .openai_classifier import OpenAIClassifier


class OpenRouterClassifier(OpenAIClassifier):
    """Text classifier using any model available on OpenRouter.

    Accepts the same arguments as OpenAIClassifier (few-shot mode, nearest-neighbour
    sampling, fixed examples, caching, results saving). The only difference is that
    requests go to OpenRouter, so config.parameters["model"] must be an OpenRouter
    model id such as "anthropic/claude-sonnet-4.5".
    """

    PROVIDER = "openrouter"
    PROVIDER_DISPLAY_NAME = "OpenRouter"
