# Text Classification Package Architecture

## Package Name: `textclassify`

## Directory Structure
```
textclassify/
├── __init__.py
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
│   ├── deepseek_classifier.py
│   └── prompts.py           # Prompt templates
├── ml/
│   ├── __init__.py
│   ├── base.py              # Abstract ML classifier
│   ├── roberta_classifier.py
│   └── preprocessing.py     # Text preprocessing utilities
├── ensemble/
│   ├── __init__.py
│   ├── base.py              # Abstract ensemble methods
│   ├── voting.py            # Voting ensemble
│   ├── weighted.py          # Weighted ensemble
│   └── routing.py           # Class-specific routing
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration management
│   └── api_keys.py          # API key management
├── utils/
│   ├── __init__.py
│   ├── logging.py           # Logging utilities
│   ├── metrics.py           # Evaluation metrics
│   └── data.py              # Data handling utilities
└── examples/
    ├── __init__.py
    ├── multi_class_example.py
    ├── multi_label_example.py
    └── ensemble_example.py
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
- **Voting Ensemble**: Majority/plurality voting
- **Weighted Ensemble**: Weighted combination based on model performance
- **Class-specific Routing**: Different models for different classes
- **Confidence-based Selection**: Choose model based on confidence scores

### 6. Key Features
- Async support for LLM calls
- Batch processing capabilities
- Caching for repeated predictions
- Comprehensive logging and monitoring
- Extensible architecture for new models
- Performance metrics and evaluation tools

