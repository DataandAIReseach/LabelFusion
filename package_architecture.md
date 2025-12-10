# Text Classification Package Architecture

## Package Name: `textclassify`

## Directory Structure
```
textclassify/
├── __init__.py
├── auto_fusion.py           # High-level AutoFusion API
├── core/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── types.py             # Data types and enums
│   └── exceptions.py        # Custom exceptions
├── llm/
│   ├── __init__.py
│   ├── base.py              # Abstract LLM classifier
│   ├── openai_classifier.py
│   ├── claude_classifier.py
│   ├── gemini_classifier.py
│   └── deepseek_classifier.py
├── ml/
│   ├── __init__.py
│   ├── base.py              # Abstract ML classifier
│   └── roberta_classifier.py
├── ensemble/
│   ├── __init__.py
│   ├── base.py              # Abstract ensemble methods
│   ├── fusion.py            # Fusion ensemble (ML+LLM with MLP)
│   ├── auto_fusion.py       # Simplified AutoFusion wrapper
│   ├── voting.py            # Voting ensemble
│   ├── weighted.py          # Weighted ensemble
│   └── routing.py           # Class-specific routing
├── prompt_engineer/
│   ├── __init__.py
│   └── ...                  # Prompt engineering utilities
├── services/
│   ├── __init__.py
│   └── ...                  # Service layer components
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration management
│   └── api_keys.py          # API key management
└── utils/
    ├── __init__.py
    ├── cache_helpers.py     # LLM prediction caching
    ├── logging.py           # Logging utilities
    ├── metrics.py           # Evaluation metrics
    ├── results_manager.py   # Results management
    └── data.py              # Data handling utilities

examples/                     # Top-level examples directory
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

## Core Design Principles

### 1. Classification Types
- **Multi-class**: Single label per text (mutually exclusive)
- **Multi-label**: Multiple labels per text (non-exclusive)

### 2. Model Categories
- **LLM-based**: OpenAI, Claude, Gemini, DeepSeek
- **Traditional ML**: RoBERTa (with potential for extension)

### 3. Key Interfaces

#### BaseClassifier
```python
class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, texts: List[str], labels: Union[List[str], List[List[str]]]) -> None
    
    @abstractmethod
    def predict(self, texts: List[str]) -> ClassificationResult
    
    @abstractmethod
    def predict_proba(self, texts: List[str]) -> ClassificationResult
```

#### ClassificationResult
```python
@dataclass
class ClassificationResult:
    predictions: List[Union[str, List[str]]]
    probabilities: Optional[List[Dict[str, float]]]
    confidence_scores: Optional[List[float]]
    metadata: Dict[str, Any]
```

### 4. Configuration System
- YAML/JSON configuration files
- Environment variable support
- API key management
- Model-specific parameters

### 5. Ensemble Strategies
- **Fusion Ensemble**: Learned fusion of ML and LLM predictions via trainable MLP
- **AutoFusion**: Simplified high-level API for fusion ensemble
- **Voting Ensemble**: Majority/plurality voting
- **Weighted Ensemble**: Weighted combination based on model performance
- **Class-specific Routing**: Different models for different classes

### 6. Key Features
- **Fusion Learning**: Trainable MLP that learns to combine ML logits and LLM scores
- **LLM Prediction Caching**: Robust caching system with hash-based validation
- **Incremental LLM Generation**: Support for interrupt/resume workflows
- **Results Management**: Standardized output structure with metrics and metadata
- **Async Support**: Async LLM calls for improved performance
- **Batch Processing**: Efficient batch processing capabilities
- **Comprehensive Logging**: Detailed logging and monitoring
- **Extensible Architecture**: Easy to add new models and ensemble strategies
- **Performance Metrics**: Built-in evaluation tools

