# LLM Prediction Caching System

## Overview

The LLM Prediction Caching System provides robust persistence and recovery capabilities for LLM-based text classification. This system automatically saves prediction results to disk, enabling recovery from connection failures, rate limits, and other interruptions.

## Key Features

### 🔄 **Automatic Recovery**
- Resumes from where you left off if interrupted
- Handles API failures gracefully
- Skips already-processed texts on restart

### 💾 **Persistent Storage**
- CSV files for human-readable results
- JSON metadata for session tracking  
- Pickle cache for fast loading
- Configurable compression

### 📊 **Progress Tracking**
- Real-time cache statistics
- Success/failure rates
- Processing time estimates
- Cache size monitoring

### 🛡️ **Error Handling**
- Failed predictions stored with error messages
- Retry capability for failed items
- Graceful degradation on cache errors

## Usage

### Basic Usage

```python
from textclassify.llm.openai_classifier import OpenAIClassifier
from textclassify.core.types import ModelConfig

# Create classifier with caching enabled (default)
classifier = OpenAIClassifier(
    config=config,
    text_column='text',
    label_columns=['positive', 'negative', 'neutral'],
    enable_cache=True,  # Default: True
    cache_dir="my_cache"  # Default: "llm_cache"
)

# Run prediction - results automatically cached
result = classifier.predict(train_df=train_df, test_df=test_df)
```

### Advanced Configuration

```python
from textclassify.llm.prediction_cache import LLMPredictionCache

# Custom cache configuration
cache = LLMPredictionCache(
    cache_dir="custom_cache",
    session_id="experiment_001", 
    auto_save_interval=5,  # Save every 5 predictions
    enable_compression=True,
    verbose=True
)

classifier = OpenAIClassifier(
    config=config,
    text_column='text',
    label_columns=label_columns,
    enable_cache=True,
    cache_dir="custom_cache"
)
```

### Recovery from Interruption

```python
# If your process was interrupted, simply run again
# The cache will automatically detect existing results
classifier = OpenAIClassifier(config=config, ...)

# This will skip already-processed texts and continue from where you left off
result = classifier.predict(train_df=train_df, test_df=test_df)

# Check what was recovered
stats = classifier.get_cache_stats()
print(f"Recovered {stats['total_predictions']} cached predictions")
```

### Cache Management

```python
# Get cache statistics
stats = classifier.get_cache_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Success rate: {stats['successful_predictions']}/{stats['total_predictions']}")
print(f"Cache size: {stats['cache_size_mb']:.2f} MB")

# Export cache to different formats
classifier.export_cache("results.csv", format="csv")
classifier.export_cache("results.json", format="json") 
classifier.export_cache("results.xlsx", format="xlsx")

# Clear cache if needed
classifier.clear_cache(confirm=True)
```

## File Structure

The cache system creates several files in your cache directory:

```
llm_cache/
├── predictions_20241007_143022_a1b2c3d4.csv     # Human-readable results
├── metadata_20241007_143022_a1b2c3d4.json       # Session metadata
└── cache_20241007_143022_a1b2c3d4.pkl          # Fast-loading cache
```

### CSV File Contents

| Column | Description |
|--------|-------------|
| `text_hash` | Unique identifier for the text |
| `text` | Original input text (truncated) |
| `prediction` | Binary vector as JSON |
| `response_text` | Raw LLM response |
| `success` | Whether prediction succeeded |
| `error_message` | Error details if failed |
| `timestamp` | When prediction was made |

### Metadata File Contents

```json
{
  "session_id": "20241007_143022_a1b2c3d4",
  "created_at": "2024-10-07T14:30:22.123456",
  "total_predictions": 150,
  "successful_predictions": 147,
  "failed_predictions": 3,
  "last_updated": "2024-10-07T14:35:45.789012"
}
```

## Benefits

### 💰 **Cost Savings**
- Avoid re-processing texts on failures
- Skip duplicate API calls
- Reduce token usage

### ⏱️ **Time Savings**  
- Resume large jobs instantly
- No need to restart from beginning
- Parallel processing support

### 🔒 **Reliability**
- Survive network interruptions
- Handle API rate limits
- Graceful error recovery

### 🔍 **Debugging**
- Inspect failed predictions
- Review LLM responses
- Track processing history

## Best Practices

### 1. **Use Unique Cache Directories**
```python
# For different experiments
classifier1 = OpenAIClassifier(..., cache_dir="experiment_1")
classifier2 = OpenAIClassifier(..., cache_dir="experiment_2") 
```

### 2. **Monitor Cache Size**
```python
stats = classifier.get_cache_stats()
if stats['cache_size_mb'] > 100:  # 100 MB limit
    print("Cache getting large, consider clearing old sessions")
```

### 3. **Export Important Results**
```python
# After successful completion
classifier.export_cache("final_results.csv")
```

### 4. **Handle Large Datasets**
```python
# For very large datasets, increase auto-save frequency
cache = LLMPredictionCache(auto_save_interval=1)  # Save after each prediction
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_cache` | `True` | Enable/disable caching |
| `cache_dir` | `"llm_cache"` | Directory for cache files |
| `session_id` | `auto` | Unique session identifier |
| `auto_save_interval` | `10` | Predictions between saves |
| `enable_compression` | `True` | Compress cache files |
| `verbose` | `True` | Log cache operations |

## Troubleshooting

### Cache Not Working
```python
# Check if caching is enabled
print(f"Cache enabled: {classifier.enable_cache}")
print(f"Cache object: {classifier.cache}")
```

### Large Cache Files
```python
# Clear old sessions periodically
import shutil
shutil.rmtree("old_cache_directory")
```

### Memory Issues
```python
# Disable caching for very large datasets
classifier = OpenAIClassifier(..., enable_cache=False)
```

### Corrupted Cache
```python
# Clear and restart
classifier.clear_cache(confirm=True)
```

## Example: Large Dataset Processing

```python
import pandas as pd
from textclassify.llm.openai_classifier import OpenAIClassifier

# Load large dataset
df = pd.read_csv("large_dataset.csv")  # 10,000 samples
train_df = df[:8000]
test_df = df[8000:]

# Create classifier with optimized caching
classifier = OpenAIClassifier(
    config=config,
    text_column='text', 
    label_columns=labels,
    enable_cache=True,
    cache_dir="large_dataset_cache"
)

try:
    # This might take hours and could be interrupted
    result = classifier.predict(train_df=train_df, test_df=test_df)
    
    # Export final results
    classifier.export_cache("large_dataset_results.xlsx")
    
except KeyboardInterrupt:
    print("Interrupted! Progress saved to cache.")
    stats = classifier.get_cache_stats()
    print(f"Completed: {stats['successful_predictions']} predictions")
    
    # Resume later by running the same code again
```

This caching system makes LLM classification robust and production-ready, saving both time and money while providing excellent debugging capabilities.