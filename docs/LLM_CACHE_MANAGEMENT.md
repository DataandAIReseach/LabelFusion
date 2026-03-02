# LLM Cache Management Guide

This guide explains how to use the cache management functions added to LLM classifiers in the LabelFusion package.

## Overview

The LLM classifiers (OpenAI, Gemini, DeepSeek) now include comprehensive cache management capabilities that allow you to:

1. **Discover** cached prediction files
2. **Load** predictions from cache files
3. **Reuse** predictions without re-running expensive LLM inference
4. **Inspect** cache statistics and summaries
5. **Debug** prediction workflows

## New Methods

All methods are available on `BaseLLMClassifier` and inherited by all LLM classifier implementations.

### 1. `discover_cached_predictions()` (Class Method)

Discovers all cached prediction files in a directory.

```python
from textclassify.llm import OpenAIClassifier, GeminiClassifier

# Discover cache files (default: "cache" directory)
discovered = OpenAIClassifier.discover_cached_predictions()

# Discover in a custom directory
discovered = OpenAIClassifier.discover_cached_predictions("cache/fusion_openai_cache")

# Returns a dictionary like:
# {
#     'validation_predictions': ['cache/val_predictions_2024-10-15-16-46-06_e9f7f5d5.json'],
#     'test_predictions': ['cache/test_predictions_2024-10-15-15-49-38_9cec6c12.json']
# }
```

**When to use:** When you want to see what cache files are available before loading them.

### 2. `load_cached_predictions_for_dataset()`

Loads predictions from a specific cache file.

```python
from textclassify.llm import OpenAIClassifier
from textclassify.config.model_config import ModelConfig

# Initialize classifier
config = ModelConfig(model_type="llm", parameters={"model": "gpt-3.5-turbo"})
classifier = OpenAIClassifier(config)

# Load cached predictions
cache_file = "cache/val_predictions_2024-10-15-16-46-06_e9f7f5d5.json"
cached_data = classifier.load_cached_predictions_for_dataset(cache_file)

# Returns:
# {
#     'predictions': [[1, 0, 0], [0, 1, 0], ...],
#     'provider': 'openai',
#     'model': 'gpt-3.5-turbo',
#     'timestamp': '2024-10-15T16:46:06',
#     'metrics': {...}  # if available
# }
```

**When to use:** When you want to inspect or manually work with cached predictions.

### 3. `get_cached_predictions_summary()`

Gets a summary of cached predictions with statistics.

```python
# Get summary of a specific cache file
summary = classifier.get_cached_predictions_summary(
    cache_file="cache/val_predictions_2024-10-15-16-46-06_e9f7f5d5.json"
)

# Or auto-discover and summarize the most recent validation cache
summary = classifier.get_cached_predictions_summary(cache_dir="cache")

# Returns:
# {
#     'cache_file': 'cache/val_predictions_2024-10-15-16-46-06_e9f7f5d5.json',
#     'total_predictions': 1000,
#     'provider': 'openai',
#     'model': 'gpt-3.5-turbo',
#     'timestamp': '2024-10-15T16:46:06',
#     'has_metrics': True,
#     'metrics': {'accuracy': 0.95, 'f1_macro': 0.93}
# }
```

**When to use:** When you want to quickly check cache file contents without loading all predictions.

### 4. `predict_with_cached_predictions()`

Use cached predictions instead of running inference (fast ensemble testing).

```python
import pandas as pd

# Load your test data
test_df = pd.read_csv("data/test.csv")

# Use cached predictions instead of running LLM inference
result = classifier.predict_with_cached_predictions(
    test_df=test_df,
    cache_file="cache/test_predictions_2024-10-15-15-49-38_9cec6c12.json"
)

# Result contains predictions and metrics (if labels available in test_df)
print(f"Accuracy: {result.metadata['metrics']['accuracy']:.4f}")
```

**When to use:** 
- Testing ensemble combinations without re-running expensive LLM calls
- Debugging prediction workflows
- Quick evaluation of different fusion strategies

### 5. `print_cache_status()`

Print a formatted summary of all available cache files.

```python
# Print comprehensive cache status
classifier.print_cache_status()

# Or specify a custom cache directory
classifier.print_cache_status(cache_dir="cache/fusion_openai_cache")
```

**Output:**
```
============================================================
📦 LLM CACHE STATUS
============================================================

📁 Cache directory: cache
✅ Found 3 cached prediction file(s)

VALIDATION PREDICTIONS:
------------------------------------------------------------

  1. val_predictions_2024-10-15-16-46-06_e9f7f5d5.json
     Provider: openai
     Model: gpt-3.5-turbo
     Predictions: 1000
     Timestamp: 2024-10-15T16:46:06
     Accuracy: 0.9500
     F1 (macro): 0.9300

TEST PREDICTIONS:
------------------------------------------------------------

  1. test_predictions_2024-10-15-15-49-38_9cec6c12.json
     Provider: gemini
     Model: gemini-1.5-flash
     Predictions: 500
     Timestamp: 2024-10-15T15:49:38

============================================================
💡 Use load_cached_predictions_for_dataset() to load a specific file
============================================================
```

**When to use:** At the start of your workflow to see what cached predictions are available.

## Typical Workflows

### Workflow 1: Quick Cache Check

```python
from textclassify.llm import OpenAIClassifier
from textclassify.config.model_config import ModelConfig

# Initialize classifier
config = ModelConfig(model_type="llm", parameters={"model": "gpt-3.5-turbo"})
classifier = OpenAIClassifier(config)

# Check what's available
classifier.print_cache_status()
```

### Workflow 2: Reuse Cached Predictions in Ensemble

```python
from textclassify.llm import OpenAIClassifier, GeminiClassifier
from textclassify.ensemble.fusion import FusionEnsemble
import pandas as pd

# Load test data
test_df = pd.read_csv("data/test.csv")

# Initialize classifiers
openai_clf = OpenAIClassifier(config_openai)
gemini_clf = GeminiClassifier(config_gemini)

# Use cached predictions (fast!)
openai_result = openai_clf.predict_with_cached_predictions(
    test_df, "cache/openai_test_predictions.json"
)
gemini_result = gemini_clf.predict_with_cached_predictions(
    test_df, "cache/gemini_test_predictions.json"
)

# Create fusion ensemble with cached results
fusion = FusionEnsemble(base_models=[openai_clf, gemini_clf])
# ... train and test fusion
```

### Workflow 3: Discover and Load Latest Cache

```python
# Discover available caches
discovered = classifier.discover_cached_predictions("cache")

# Get the most recent validation cache
if 'validation_predictions' in discovered:
    latest_cache = discovered['validation_predictions'][0]
    
    # Get summary
    summary = classifier.get_cached_predictions_summary(latest_cache)
    print(f"Cache has {summary['total_predictions']} predictions")
    
    # Use it
    result = classifier.predict_with_cached_predictions(test_df, latest_cache)
```

### Workflow 4: Compare Different LLM Providers

```python
import pandas as pd

test_df = pd.read_csv("data/test.csv")

# Load cached predictions from different providers
providers = {
    'openai': 'cache/openai_test_predictions.json',
    'gemini': 'cache/gemini_test_predictions.json',
    'deepseek': 'cache/deepseek_test_predictions.json'
}

results = {}
for provider_name, cache_file in providers.items():
    # Get summary
    summary = classifier.get_cached_predictions_summary(cache_file)
    results[provider_name] = summary['metrics']
    
# Compare metrics
for provider, metrics in results.items():
    print(f"\n{provider.upper()}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
```

## Cache File Format

Cache files are stored as JSON with the following structure:

```json
{
  "predictions": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ],
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "timestamp": "2024-10-15T16:46:06",
  "metrics": {
    "accuracy": 0.95,
    "f1_macro": 0.93,
    "precision_macro": 0.94,
    "recall_macro": 0.92
  }
}
```

## Integration with FusionEnsemble

The cache functions work seamlessly with FusionEnsemble:

```python
from textclassify.ensemble.fusion import FusionEnsemble

# FusionEnsemble also has cache functions
discovered = FusionEnsemble.discover_cached_predictions("cache")

# Load and use cached predictions
fusion = FusionEnsemble(base_models=[...])
fusion.fit_with_cached_predictions(train_df, discovered['train_predictions'][0])
result = fusion.predict_with_cached_predictions(test_df, discovered['test_predictions'][0])
```

## Best Practices

1. **Always check cache status first** using `print_cache_status()` to see what's available

2. **Use descriptive cache directories** to organize predictions by experiment or model type

3. **Cache validation predictions** for quick ensemble testing and hyperparameter tuning

4. **Keep timestamp information** in cache files to track when predictions were made

5. **Include metrics** in cache files when available for quick comparisons

## Error Handling

All cache functions include error handling and return None or empty dictionaries on failure:

```python
# If cache file doesn't exist
result = classifier.load_cached_predictions_for_dataset("nonexistent.json")
# Returns: None (with error message)

# If cache directory doesn't exist
discovered = classifier.discover_cached_predictions("nonexistent_dir")
# Returns: {} (with warning message)
```

## Compatibility

- **Backward Compatible**: Existing code continues to work without changes
- **All LLM Classifiers**: OpenAI, Gemini, DeepSeek all support these methods
- **FusionEnsemble**: Similar cache functions available for ensemble workflows
- **JSON Format**: Standard format ensures compatibility across different Python versions

## Performance Benefits

Using cached predictions can save significant time:

- **LLM Inference**: ~1-5 seconds per sample
- **Cached Loading**: ~0.001 seconds per sample
- **Speedup**: **1000-5000x faster** for 1000 samples

This is especially valuable for:
- Testing different ensemble strategies
- Hyperparameter optimization
- Debugging fusion models
- Quick prototype iterations
