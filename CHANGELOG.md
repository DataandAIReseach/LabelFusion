# Changelog

All notable changes to the TextClassify project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-07

### Added
- Initial release of TextClassify package
- Support for multi-class and multi-label text classification
- LLM-based classifiers for OpenAI, Claude, Gemini, and DeepSeek
- Traditional ML classifier using RoBERTa (optional dependency)
- Ensemble methods: Voting, Weighted, and Class Routing
- Comprehensive configuration management system
- Secure API key management
- Data loading utilities for CSV and JSON formats
- Evaluation metrics and model comparison tools
- Comprehensive examples and documentation
- Async support for LLM processing
- Configurable prompt engineering for LLMs
- Text preprocessing utilities
- Logging and monitoring capabilities

### Features
- **Multi-class Classification**: Single label per text (mutually exclusive)
- **Multi-label Classification**: Multiple labels per text (non-exclusive)
- **LLM Providers**: OpenAI GPT, Claude, Google Gemini, DeepSeek
- **Traditional ML**: RoBERTa-based fine-tunable models
- **Ensemble Methods**: 
  - Voting ensemble with majority/plurality strategies
  - Weighted ensemble with performance-based weighting
  - Class routing ensemble for class-specific model assignment
- **Configuration**: YAML/JSON configuration files with environment variable support
- **Data Handling**: CSV/JSON loading, data splitting, class balancing
- **Evaluation**: Comprehensive metrics for both classification types
- **Examples**: Complete working examples for all features

### Dependencies
- Core: aiohttp, requests, numpy, pandas, tqdm, python-dateutil
- Optional: transformers, torch, scikit-learn (for RoBERTa classifier)
- Optional: pyyaml (for YAML configuration support)

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Multiple example scripts demonstrating features
- Configuration management guide
- Troubleshooting and FAQ

## [Unreleased]

### Planned Features
- Additional LLM providers (Llama, Mistral)
- More traditional ML models (BERT, DistilBERT)
- Advanced ensemble strategies (stacking, boosting)
- Model performance optimization tools
- Batch processing capabilities
- Model caching and persistence
- Web API interface
- CLI tools for common tasks

