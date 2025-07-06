# TextClassify

A comprehensive Python package for multi-class and multi-label text classification using Large Language Models (LLMs) and traditional machine learning models, with advanced ensemble methods for optimized performance.

## Features

### 🤖 Multiple Model Types
- **LLM Providers**: OpenAI GPT, Claude (Anthropic), Google Gemini, DeepSeek
- **Traditional ML**: RoBERTa-based classifiers with fine-tuning
- **Ensemble Methods**: Voting, weighted, and class-specific routing

### 📊 Classification Types
- **Multi-class**: Single label per text (mutually exclusive)
- **Multi-label**: Multiple labels per text (non-exclusive)

### 🔧 Advanced Features
- Asynchronous LLM processing for better performance
- Configurable prompt engineering for LLMs
- Model combination strategies for optimal results
- Comprehensive evaluation metrics
- Easy configuration management
- Secure API key handling

## Installation

```bash
pip install textclassify
```

### Optional Dependencies

For RoBERTa classifier:
```bash
pip install transformers torch
```

For configuration file support:
```bash
pip install pyyaml
```

## Quick Start

### Basic Multi-class Classification

```python
from textclassify import OpenAIClassifier, ClassificationType, ModelConfig, ModelType
from textclassify.utils import DataLoader

# Create configuration
config = ModelConfig(
    model_name="gpt-3.5-turbo",
    model_type=ModelType.LLM,
    api_key="your-openai-api-key"
)

# Create classifier
classifier = OpenAIClassifier(config)

# Prepare training data
training_data = DataLoader.from_lists(
    texts=["I love this movie!", "This film is terrible.", "It was okay."],
    labels=["positive", "negative", "neutral"],
    classification_type=ClassificationType.MULTI_CLASS
)

# Train and predict
classifier.fit(training_data)
result = classifier.predict(["This movie is amazing!"])
print(result.predictions)  # ['positive']
```

### Multi-label Classification

```python
from textclassify import GeminiClassifier, ClassificationType

# Multi-label training data
training_data = DataLoader.from_lists(
    texts=[
        "Action movie with great special effects",
        "Romantic comedy with funny moments",
        "Scary horror film with supernatural elements"
    ],
    labels=[
        ["action", "special_effects"],
        ["romance", "comedy"],
        ["horror", "supernatural"]
    ],
    classification_type=ClassificationType.MULTI_LABEL
)

# Create and train classifier
config = ModelConfig(
    model_name="gemini-1.5-flash",
    model_type=ModelType.LLM,
    api_key="your-gemini-api-key"
)

classifier = GeminiClassifier(config)
classifier.fit(training_data)

# Predict multiple labels
result = classifier.predict(["Funny action movie with romance"])
print(result.predictions)  # [['action', 'comedy', 'romance']]
```

### Ensemble Methods

```python
from textclassify import VotingEnsemble, WeightedEnsemble, EnsembleConfig

# Create ensemble with multiple models
ensemble_config = EnsembleConfig(
    models=[openai_config, claude_config],
    ensemble_method="voting"
)

ensemble = VotingEnsemble(ensemble_config)
ensemble.add_model(OpenAIClassifier(openai_config), "openai")
ensemble.add_model(ClaudeClassifier(claude_config), "claude")

# Train and predict
ensemble.fit(training_data)
result = ensemble.predict(["This is a great movie!"])
```

## Supported Models

### LLM Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| OpenAI | gpt-3.5-turbo, gpt-4, gpt-4-turbo | ✅ |
| Claude | claude-3-haiku, claude-3-sonnet, claude-3-opus | ✅ |
| Gemini | gemini-1.5-flash, gemini-1.5-pro | ✅ |
| DeepSeek | deepseek-chat, deepseek-coder | ✅ |

### Traditional ML Models

| Model | Description | Dependencies |
|-------|-------------|--------------|
| RoBERTa | Fine-tunable transformer model | transformers, torch |

## Ensemble Methods

### Voting Ensemble
Combines predictions through majority or plurality voting.

```python
from textclassify import VotingEnsemble

ensemble = VotingEnsemble(ensemble_config)
# Supports: 'majority', 'plurality' voting strategies
```

### Weighted Ensemble
Combines predictions using weighted averages based on model performance.

```python
from textclassify import WeightedEnsemble

ensemble_config = EnsembleConfig(
    models=[model1_config, model2_config],
    ensemble_method="weighted",
    weights=[0.7, 0.3]  # Custom weights
)
ensemble = WeightedEnsemble(ensemble_config)
```

### Class Routing Ensemble
Routes different classes to different models for optimized performance.

```python
from textclassify import ClassRoutingEnsemble

routing_rules = {
    "technical": "model1",
    "creative": "model2"
}

ensemble_config = EnsembleConfig(
    models=[model1_config, model2_config],
    ensemble_method="routing",
    routing_rules=routing_rules
)
ensemble = ClassRoutingEnsemble(ensemble_config)
```

## Configuration Management

### API Key Management

```python
from textclassify import APIKeyManager

# Set up API keys
api_manager = APIKeyManager()
api_manager.set_key("openai", "your-openai-key")
api_manager.set_key("claude", "your-claude-key")

# Or use environment variables
# export OPENAI_API_KEY="your-key"
# export CLAUDE_API_KEY="your-key"
```

### Configuration Files

```python
from textclassify import Config

# Create configuration
config = Config()
config.set('llm.default_provider', 'openai')
config.set('general.batch_size', 32)

# Save configuration
config.save('my_config.yaml')

# Load configuration
config = Config('my_config.yaml')
```

## Data Loading and Processing

### From Files

```python
from textclassify.utils import DataLoader

# Load from CSV
data = DataLoader.from_csv(
    'data.csv',
    text_column='text',
    label_column='label',
    classification_type=ClassificationType.MULTI_CLASS
)

# Load from JSON
data = DataLoader.from_json(
    'data.json',
    text_field='text',
    label_field='labels',
    classification_type=ClassificationType.MULTI_LABEL
)
```

### Data Utilities

```python
from textclassify.utils import split_data, balance_data, get_data_statistics

# Split data
train_data, val_data = split_data(data, train_ratio=0.8, stratify=True)

# Balance classes
balanced_data = balance_data(data, method='oversample')

# Get statistics
stats = get_data_statistics(data)
print(f"Total samples: {stats['total_samples']}")
print(f"Classes: {stats['num_classes']}")
```

## Evaluation and Metrics

```python
from textclassify.utils import evaluate_predictions, compare_models

# Evaluate single model
metrics = evaluate_predictions(result, true_labels)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['macro_f1']:.3f}")

# Compare multiple models
comparison = compare_models(
    [result1, result2, result3],
    true_labels,
    model_names=['Model A', 'Model B', 'Model C']
)
```

## Examples

The package includes comprehensive examples:

- `examples/multi_class_example.py` - Multi-class classification
- `examples/multi_label_example.py` - Multi-label classification  
- `examples/ensemble_example.py` - Advanced ensemble methods

Run examples:
```bash
python -m textclassify.examples.multi_class_example
python -m textclassify.examples.multi_label_example
python -m textclassify.examples.ensemble_example
```

## API Reference

### Core Classes

- `BaseClassifier` - Abstract base class for all classifiers
- `ClassificationResult` - Container for prediction results
- `TrainingData` - Container for training data
- `ModelConfig` - Configuration for individual models
- `EnsembleConfig` - Configuration for ensemble methods

### LLM Classifiers

- `OpenAIClassifier` - OpenAI GPT models
- `ClaudeClassifier` - Anthropic Claude models
- `GeminiClassifier` - Google Gemini models
- `DeepSeekClassifier` - DeepSeek models

### ML Classifiers

- `RoBERTaClassifier` - RoBERTa-based classifier

### Ensemble Methods

- `VotingEnsemble` - Voting-based ensemble
- `WeightedEnsemble` - Weighted ensemble
- `ClassRoutingEnsemble` - Class-specific routing

### Utilities

- `DataLoader` - Data loading and saving
- `Config` - Configuration management
- `APIKeyManager` - API key management
- `ClassificationMetrics` - Evaluation metrics

## Advanced Usage

### Custom Preprocessing

```python
from textclassify.ml.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=False,
    remove_numbers=True
)

# Use with RoBERTa classifier
config = ModelConfig(
    model_name="roberta-base",
    model_type=ModelType.TRADITIONAL_ML,
    parameters={
        "preprocessing": preprocessor.get_config()
    }
)
```

### Async Processing

```python
import asyncio
from textclassify import OpenAIClassifier

async def async_classification():
    classifier = OpenAIClassifier(config)
    classifier.fit(training_data)
    
    # Async prediction
    result = await classifier.predict_async(texts)
    return result

# Run async
result = asyncio.run(async_classification())
```

### Custom Prompts

```python
from textclassify.llm.prompts import MultiClassPromptTemplate

# Create custom prompt template
template = MultiClassPromptTemplate()
custom_prompt = template.format_prompt(
    text="Your text here",
    classes=["class1", "class2", "class3"],
    examples=[{"text": "example", "label": "class1"}]
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/textclassify.git
cd textclassify
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://textclassify.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/your-org/textclassify/issues)
- 💬 [Discussions](https://github.com/your-org/textclassify/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**TextClassify** - Making text classification simple, powerful, and flexible.

