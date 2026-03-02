"""LLM-based text classifiers module.

This module provides LLM-based text classifiers with support for:
- Multiple LLM providers (OpenAI, Gemini, DeepSeek)
- Few-shot learning
- Multi-label and multi-class classification
- Automatic and manual prediction caching

Automatic Cache (NEW!):
----------------------
Enable automatic cache checking and reuse in the constructor:

    from textclassify.llm import OpenAIClassifier
    
    # Auto-cache: automatically loads from cache if available
    classifier = OpenAIClassifier(
        config, 
        auto_use_cache=True,  #  NEW!
        cache_dir="cache"
    )
    
    # predict() automatically checks cache (1000-5000x faster!)
    result = classifier.predict(train_df, test_df)

Manual Cache Management:
-----------------------
Or use manual cache control:

    # Discover cached predictions
    discovered = OpenAIClassifier.discover_cached_predictions("cache")
    
    # Explicitly load and reuse cached predictions
    result = classifier.predict_with_cached_predictions(test_df, cache_file)
    
    # Check cache status
    classifier.print_cache_status()

See docs/AUTO_CACHE_FEATURE.md and docs/LLM_CACHE_MANAGEMENT.md for details.
"""

from .base import BaseLLMClassifier
from .openai_classifier import OpenAIClassifier
from .gemini_classifier import GeminiClassifier
from .deepseek_classifier import DeepSeekClassifier

__all__ = [
    "BaseLLMClassifier",
    "OpenAIClassifier",
    "GeminiClassifier",
    "DeepSeekClassifier",
]

