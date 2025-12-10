# TextClassify Package Overview

## 🎯 Project Summary

TextClassify is a comprehensive Python package for multi-class and multi-label text classification that combines the power of Large Language Models (LLMs) with traditional machine learning approaches. The package provides advanced ensemble methods to optimize classification performance by intelligently combining different models.

## 📦 Package Structure

```
textclassify/
├── __init__.py                 # Main package entry point
├── auto_fusion.py              # High-level AutoFusion API
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── types.py               # Data types and enums
│   ├── base.py                # Abstract base classes
│   └── exceptions.py          # Custom exceptions
├── llm/                       # LLM-based classifiers
│   ├── __init__.py
│   ├── base.py                # Base LLM classifier
│   ├── openai_classifier.py   # OpenAI GPT models
│   ├── claude_classifier.py   # Anthropic Claude models
│   ├── gemini_classifier.py   # Google Gemini models
│   └── deepseek_classifier.py # DeepSeek models
├── ml/                        # Traditional ML classifiers
│   ├── __init__.py
│   ├── base.py                # Base ML classifier
│   └── roberta_classifier.py  # RoBERTa-based classifier
├── ensemble/                  # Ensemble methods
│   ├── __init__.py
│   ├── base.py                # Base ensemble class
│   ├── fusion.py              # Fusion ensemble (ML+LLM+MLP)
│   ├── auto_fusion.py         # Simplified AutoFusion wrapper
│   ├── voting.py              # Voting ensemble
│   ├── weighted.py            # Weighted ensemble
│   └── routing.py             # Class routing ensemble
├── prompt_engineer/           # Prompt engineering utilities
│   └── __init__.py
├── services/                  # Service layer components
│   └── __init__.py
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── settings.py            # Configuration handling
│   └── api_keys.py            # API key management
└── utils/                     # Utility functions
    ├── __init__.py
    ├── cache_helpers.py       # LLM prediction caching
    ├── logging.py             # Logging utilities
    ├── metrics.py             # Evaluation metrics
    ├── results_manager.py     # Results management
    └── data.py                # Data handling utilities

examples/                       # Top-level example scripts
├── ml_standalone_example.py
├── llm_standalone_example.py
├── fusion_ensemble_example.py
├── test_singlelabel_ml.py
├── test_singlelabel_autofusion.py
├── test_multilabel_autofusion.py
├── ml_cache_mock.py
├── llm_cache_mock.py
├── llm_cache_usage_example.py
├── cache_usage_demo.py
├── ensemble_cache_interrupt_demo.py
└── minimal_precache_demo.py
```

## 🚀 Key Features

### 1. Multiple Model Types
- **LLM Providers**: OpenAI GPT, Claude, Google Gemini, DeepSeek
- **Traditional ML**: RoBERTa-based fine-tunable models
- **Flexible Architecture**: Easy to add new model types

### 2. Classification Types
- **Multi-class**: Single label per text (mutually exclusive)
- **Multi-label**: Multiple labels per text (non-exclusive)
- **Unified API**: Same interface for both classification types

### 3. Advanced Ensemble Methods
- **Voting Ensemble**: Majority/plurality voting strategies
- **Weighted Ensemble**: Performance-based model weighting
- **Class Routing**: Class-specific model assignment for optimization

### 4. Comprehensive Configuration
- **API Key Management**: Secure storage and environment variable support
- **Configuration Files**: YAML/JSON support with hierarchical settings
- **Flexible Parameters**: Model-specific and global configuration options

### 5. Data Handling & Evaluation
- **Data Loading**: CSV/JSON format support with automatic type detection
- **Data Utilities**: Splitting, balancing, and preprocessing tools
- **Evaluation Metrics**: Comprehensive metrics for both classification types
- **Model Comparison**: Side-by-side performance analysis

## 🛠️ Technical Implementation

### Core Architecture
- **Abstract Base Classes**: Consistent interface across all model types
- **Type Safety**: Comprehensive type hints and validation
- **Error Handling**: Custom exceptions with detailed error messages
- **Async Support**: Asynchronous processing for LLM models

### LLM Integration
- **Prompt Engineering**: Configurable prompt templates for different tasks
- **API Abstraction**: Unified interface for different LLM providers
- **Rate Limiting**: Built-in throttling and retry mechanisms
- **Response Parsing**: Robust parsing of LLM outputs with fallback strategies

### Traditional ML
- **Fine-tuning Support**: Full training pipeline for RoBERTa models
- **Preprocessing**: Comprehensive text preprocessing utilities
- **Model Persistence**: Save and load trained models
- **GPU Support**: Automatic GPU detection and utilization

### Ensemble Intelligence
- **Dynamic Weighting**: Automatic weight calculation based on performance
- **Class-specific Routing**: Route different classes to specialized models
- **Fallback Strategies**: Graceful handling of model failures
- **Performance Optimization**: Intelligent model selection strategies

## 📊 Performance Features

### Optimization Strategies
1. **Model Specialization**: Different models for different types of content
2. **Ensemble Voting**: Combine predictions for improved accuracy
3. **Weighted Combinations**: Performance-based model weighting
4. **Class Routing**: Route specific classes to best-performing models

### Evaluation & Monitoring
- **Comprehensive Metrics**: Accuracy, F1, precision, recall for both classification types
- **Model Comparison**: Side-by-side performance analysis
- **Confusion Matrices**: Detailed error analysis for multi-class tasks
- **Performance Tracking**: Monitor model performance over time

## 🔧 Usage Examples

### Quick Start
```python
from textclassify import OpenAIClassifier, ClassificationType, ModelConfig, ModelType

# Create configuration
config = ModelConfig(
    model_name="gpt-3.5-turbo",
    model_type=ModelType.LLM,
    api_key="your-api-key"
)

# Create and train classifier
classifier = OpenAIClassifier(config)
classifier.fit(training_data)

# Make predictions
result = classifier.predict(["This is a great product!"])
```

### Ensemble Usage
```python
from textclassify import VotingEnsemble, EnsembleConfig

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
result = ensemble.predict(test_texts)
```

## 📋 Dependencies

### Core Dependencies (Required)
- `aiohttp>=3.8.0` - Async HTTP client for LLM APIs
- `requests>=2.28.0` - HTTP client for synchronous requests
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.4.0` - Data manipulation
- `tqdm>=4.64.0` - Progress bars
- `python-dateutil>=2.8.0` - Date utilities

### Optional Dependencies
- `transformers>=4.20.0` - For RoBERTa classifier
- `torch>=1.12.0` - For RoBERTa classifier
- `scikit-learn>=1.1.0` - For ML utilities
- `pyyaml>=6.0` - For YAML configuration support

### LLM API Dependencies
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.3.0` - Claude API client
- `google-generativeai>=0.3.0` - Gemini API client

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Core functionality, types, configuration
- **Integration Tests**: End-to-end package functionality
- **Example Tests**: Verify all examples work correctly
- **Dependency Tests**: Optional dependency handling

### Test Structure
```
tests/
├── unit/
│   ├── test_core_types.py
│   ├── test_config.py
│   └── ...
└── integration/
    ├── test_package_integration.py
    └── ...
```

## 📚 Documentation

### Comprehensive Documentation
- **README.md**: Complete usage guide with examples
- **API Reference**: Detailed documentation of all classes and methods
- **Examples**: Working examples for all major features
- **Configuration Guide**: Complete configuration options
- **Troubleshooting**: Common issues and solutions

### Example Scripts
- `multi_class_example.py`: Multi-class classification examples
- `multi_label_example.py`: Multi-label classification examples
- `ensemble_example.py`: Advanced ensemble strategies

## 🔒 Security & Best Practices

### API Key Security
- **Secure Storage**: API keys stored with restricted file permissions
- **Environment Variables**: Support for environment-based configuration
- **Key Validation**: Format validation for different providers
- **No Hardcoding**: No API keys in code or configuration files

### Error Handling
- **Custom Exceptions**: Specific exceptions for different error types
- **Graceful Degradation**: Fallback strategies for model failures
- **Detailed Logging**: Comprehensive logging for debugging
- **Input Validation**: Thorough validation of user inputs

## 🚀 Installation & Setup

### Installation
```bash
pip install textclassify
```

### Optional Features
```bash
# For RoBERTa classifier
pip install textclassify[ml]

# For YAML configuration
pip install textclassify[config]

# For all features
pip install textclassify[all]
```

### Development Setup
```bash
git clone <repository>
cd textclassify
pip install -e ".[dev]"
```

## 🎯 Use Cases

### Business Applications
- **Customer Support**: Automatic ticket classification and routing
- **Content Moderation**: Multi-label content classification
- **Market Research**: Sentiment and topic classification
- **Document Processing**: Automatic document categorization

### Research Applications
- **Academic Research**: Text classification experiments
- **Model Comparison**: Benchmark different approaches
- **Ensemble Studies**: Research on ensemble methods
- **Performance Analysis**: Detailed evaluation metrics

## 🔮 Future Enhancements

### Planned Features
- Additional LLM providers (Llama, Mistral)
- More traditional ML models (BERT, DistilBERT)
- Advanced ensemble strategies (stacking, boosting)
- Model performance optimization tools
- Batch processing capabilities
- Web API interface
- CLI tools for common tasks

### Extensibility
- **Plugin Architecture**: Easy to add new model types
- **Custom Ensembles**: Framework for custom ensemble methods
- **Preprocessing Pipeline**: Extensible text preprocessing
- **Evaluation Metrics**: Custom metric implementations

## 📈 Performance Characteristics

### Scalability
- **Async Processing**: Concurrent LLM requests
- **Batch Processing**: Efficient handling of large datasets
- **Memory Management**: Optimized memory usage for large models
- **GPU Utilization**: Automatic GPU detection and usage

### Reliability
- **Error Recovery**: Robust error handling and recovery
- **Rate Limiting**: Respect API rate limits
- **Retry Logic**: Automatic retry with exponential backoff
- **Fallback Strategies**: Graceful degradation on failures

---

**TextClassify** provides a complete, production-ready solution for text classification that combines the latest in LLM technology with proven traditional ML approaches, all wrapped in an easy-to-use, well-documented package.

