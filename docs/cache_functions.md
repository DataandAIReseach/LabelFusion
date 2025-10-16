# LLM Cache Functions for FusionEnsemble

## Overview

The `FusionEnsemble` class now includes several new functions to efficiently read and reuse cached LLM predictions for validation and test data, avoiding the need to regenerate expensive LLM predictions every time you train or test your model.

## Problem Solved

Previously, every time you wanted to train a fusion ensemble or make predictions, the LLM would need to generate new predictions, which is:
- ⏰ **Time-consuming**: LLM inference can take minutes or hours for large datasets
- 💰 **Expensive**: API calls to commercial LLMs cost money
- 🔄 **Redundant**: Same predictions for same data are generated repeatedly

## New Functions

### 1. `load_cached_predictions_for_dataset(df, dataset_type="auto")`

Load cached LLM predictions for a specific dataset.

```python
# Load cached validation predictions
cached_val_preds = fusion_ensemble.load_cached_predictions_for_dataset(
    val_df, dataset_type="validation"
)

# Load cached test predictions  
cached_test_preds = fusion_ensemble.load_cached_predictions_for_dataset(
    test_df, dataset_type="test"
)

# Auto-detect which cache to use
cached_preds = fusion_ensemble.load_cached_predictions_for_dataset(
    df, dataset_type="auto"
)
```

### 2. `fit_with_cached_predictions(train_df, val_df, val_llm_predictions=None, force_load_from_cache=False)`

Train the fusion ensemble with automatic cache loading for validation predictions.

```python
# Automatically try to load cached validation predictions
training_results = fusion_ensemble.fit_with_cached_predictions(
    train_df=train_data,
    val_df=val_data
)

# Force using cached predictions (fail if not found)
training_results = fusion_ensemble.fit_with_cached_predictions(
    train_df=train_data,
    val_df=val_data,
    force_load_from_cache=True
)
```

### 3. `predict_with_cached_predictions(test_df, true_labels=None, test_llm_predictions=None, force_load_from_cache=False)`

Make predictions with automatic cache loading for test predictions.

```python
# Automatically try to load cached test predictions
test_results = fusion_ensemble.predict_with_cached_predictions(
    test_df=test_data
)

# Force using cached predictions (fail if not found)
test_results = fusion_ensemble.predict_with_cached_predictions(
    test_df=test_data,
    force_load_from_cache=True
)
```

### 4. `get_cached_predictions_summary()`

Get a summary of available cached predictions.

```python
summary = fusion_ensemble.get_cached_predictions_summary()
print(summary)
# Output:
# {
#     "validation_cache": {
#         "path": "./cache/val_predictions",
#         "files": ["./cache/val_predictions_20241015_hash123.json"],
#         "latest_file": "./cache/val_predictions_20241015_hash123.json",
#         "available": True
#     },
#     "test_cache": {
#         "path": "./cache/test_predictions", 
#         "files": [],
#         "latest_file": None,
#         "available": False
#     }
# }
```

### 5. `print_cache_status()`

Print a detailed, user-friendly status of cached predictions.

```python
fusion_ensemble.print_cache_status()
```

Output:
```
============================================================
🗂️  FUSION ENSEMBLE CACHE STATUS
============================================================

📊 VALIDATION CACHE:
   Path: ./cache/validation_predictions
   Status: ✅ Available (3 files)
   Latest: validation_predictions_20241015_abc123.json

🧪 TEST CACHE:
   Path: ./cache/test_predictions
   Status: ❌ No cached files found

💡 USAGE TIPS:
   • Use fit_with_cached_predictions() to automatically load cached validation predictions
   • Use predict_with_cached_predictions() to automatically load cached test predictions
   • Use load_cached_predictions_for_dataset() for manual cache loading
   • Set force_load_from_cache=True to ensure cached predictions are used
============================================================
```

### 6. `discover_cached_predictions(cache_directory)` (Class Method)

Discover all cached LLM prediction files in a directory.

```python
# Find all cached files in a directory
cached_files = FusionEnsemble.discover_cached_predictions("./llm_cache")
print(cached_files)
# Output:
# {
#     "validation_predictions": [
#         "./llm_cache/validation_predictions_20241015_abc123.json",
#         "./llm_cache/validation_predictions_20241014_def456.json"
#     ],
#     "test_predictions": [
#         "./llm_cache/test_predictions_20241015_ghi789.json"
#     ]
# }
```

## Configuration

To use caching, configure cache paths in your ensemble config:

```python
ensemble_config = EnsembleConfig(
    ensemble_method="fusion",
    models=[ml_model, llm_model],
    parameters={
        'val_llm_cache_path': './cache/validation_predictions',
        'test_llm_cache_path': './cache/test_predictions',
        'fusion_epochs': 10,
        'hidden_dims': [64, 32]
    }
)
```

## Cache File Format

Cache files are stored as JSON with metadata:

```json
{
    "predictions": ["positive", "negative", "neutral", ...],
    "metadata": {
        "num_samples": 1000,
        "dataset_hash": "abc123def456",
        "timestamp": "2024-10-15T10:30:00",
        "model_info": "deepseek-chat",
        "classification_type": "multi_class"
    }
}
```

## Best Practices

### 1. **Use Automatic Cache Loading (Recommended)**
```python
# Training
results = fusion_ensemble.fit_with_cached_predictions(train_df, val_df)

# Prediction  
results = fusion_ensemble.predict_with_cached_predictions(test_df)
```

### 2. **Check Cache Status Before Training**
```python
fusion_ensemble.print_cache_status()
# This helps you understand what cached data is available
```

### 3. **Use Force Cache for Production**
```python
# In production, ensure you only use validated cached predictions
results = fusion_ensemble.predict_with_cached_predictions(
    test_df, 
    force_load_from_cache=True
)
```

### 4. **Organize Cache Files**
```python
# Use descriptive cache paths
ensemble_config = EnsembleConfig(
    ensemble_method="fusion",
    parameters={
        'val_llm_cache_path': './cache/sentiment_analysis/deepseek/validation',
        'test_llm_cache_path': './cache/sentiment_analysis/deepseek/test',
    }
)
```

### 5. **Validate Cache Compatibility**
The caching system automatically validates:
- ✅ Dataset hash (ensures same data)
- ✅ Sample count (ensures same number of samples)
- ✅ File integrity (ensures valid JSON)

## Error Handling

```python
try:
    # Try to use cached predictions
    results = fusion_ensemble.predict_with_cached_predictions(
        test_df, 
        force_load_from_cache=True
    )
except EnsembleError as e:
    print(f"Cache loading failed: {e}")
    # Fallback to generating new predictions
    results = fusion_ensemble.predict(test_df)
```

## Benefits

- ⚡ **Speed**: Skip expensive LLM inference
- 💰 **Cost**: Reduce API costs
- 🔄 **Reproducibility**: Use exact same predictions across experiments
- 🧪 **Experimentation**: Quickly test different fusion configurations
- 📊 **Consistency**: Ensure fair comparison between models

## Migration from Old Code

### Before (Manual Cache Management)
```python
# Old way - manual cache handling
if os.path.exists("cached_predictions.json"):
    with open("cached_predictions.json", 'r') as f:
        cached_preds = json.load(f)
    results = fusion_ensemble.fit(train_df, val_df, cached_preds)
else:
    results = fusion_ensemble.fit(train_df, val_df)
```

### After (Automatic Cache Management)
```python
# New way - automatic cache handling
results = fusion_ensemble.fit_with_cached_predictions(train_df, val_df)
```

The new functions handle all the complexity of cache validation, file discovery, and error handling automatically!