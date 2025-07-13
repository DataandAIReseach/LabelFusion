# Contributing to TextClassify

🎉 Thank you for your interest in contributing to TextClassify! We welcome contributions from everyone and are grateful for every pull request, bug report, and suggestion.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@textclassify.com](mailto:conduct@textclassify.com).

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and text classification
- Familiarity with async Python programming (for LLM classifiers)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/textclassify.git
   cd textclassify
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .  # Install package in editable mode
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   python -c "import textclassify; print('Setup successful!')"
   pytest tests/ -v
   ```

### Environment Variables

Create a `.env` file for development:
```bash
# API Keys for testing (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Development settings
TEXTCLASSIFY_DEBUG=true
TEXTCLASSIFY_LOG_LEVEL=DEBUG
```

## 💡 Contributing Guidelines

### What We're Looking For

- 🐛 **Bug Fixes**: Help us identify and fix issues
- ✨ **New Features**: Implement new classifiers, ensemble methods, or utilities
- 📚 **Documentation**: Improve existing docs or add new examples
- 🧪 **Tests**: Add test coverage for existing or new functionality
- 🔧 **Performance**: Optimize existing code for better performance
- 🎨 **Code Quality**: Improve code organization and readability

### Areas for Contribution

1. **New LLM Providers**
   - Implement classifiers for new LLM APIs
   - Follow the existing pattern in `textclassify.llm.base`

2. **Machine Learning Models**
   - Add new transformer-based classifiers
   - Implement traditional ML algorithms
   - Create hybrid approaches

3. **Ensemble Methods**
   - Develop new voting strategies
   - Create adaptive ensemble techniques
   - Implement meta-learning approaches

4. **Utilities and Tools**
   - Data preprocessing utilities
   - Evaluation metrics
   - Visualization tools
   - Performance monitoring

5. **Examples and Tutorials**
   - Real-world use cases
   - Integration examples
   - Performance benchmarks
   - Best practices guides

## 🔄 Pull Request Process

### Before You Start

1. **Check Existing Issues**: Look for related issues or feature requests
2. **Create an Issue**: If none exists, create one to discuss your proposal
3. **Get Feedback**: Wait for maintainer feedback before starting large changes

### Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   git checkout -b fix/issue-number-description
   ```

2. **Make Your Changes**
   - Write clean, readable code
   - Follow existing code patterns
   - Add appropriate comments and docstrings

3. **Write Tests**
   ```bash
   # Run tests during development
   pytest tests/unit/test_your_module.py -v
   
   # Run all tests
   pytest tests/ -v
   
   # Check coverage
   pytest tests/ --cov=textclassify --cov-report=html
   ```

4. **Update Documentation**
   - Update docstrings for new functions/classes
   - Add examples to relevant documentation
   - Update README if necessary

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new ensemble voting strategy"
   
   # Follow conventional commit format:
   # feat: new feature
   # fix: bug fix
   # docs: documentation changes
   # style: formatting changes
   # refactor: code refactoring
   # test: adding tests
   # chore: maintenance tasks
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Checklist

- [ ] **Code Quality**
  - [ ] Code follows project style guidelines
  - [ ] No lint errors (`flake8 textclassify/`)
  - [ ] Type hints added where appropriate
  - [ ] Code is properly formatted (`black textclassify/`)

- [ ] **Testing**
  - [ ] New code has appropriate test coverage
  - [ ] All tests pass (`pytest tests/ -v`)
  - [ ] No regression in existing functionality

- [ ] **Documentation**
  - [ ] Docstrings added/updated for new code
  - [ ] Examples updated if needed
  - [ ] README updated if necessary

- [ ] **Functionality**
  - [ ] Feature works as described
  - [ ] Error handling is appropriate
  - [ ] Performance impact considered

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_config.py
│   ├── test_core_types.py
│   └── test_classifiers/
├── integration/          # Integration tests
│   └── test_package_integration.py
└── conftest.py          # Pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py -v

# Run tests with coverage
pytest --cov=textclassify --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only integration tests
pytest tests/integration/ -v
```

### Writing Tests

```python
import pytest
from textclassify import OpenAIClassifier, ModelConfig

class TestOpenAIClassifier:
    def test_initialization(self):
        config = ModelConfig(model_name="gpt-3.5-turbo")
        classifier = OpenAIClassifier(config)
        assert classifier.config.model_name == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_async_prediction(self):
        # Test async functionality
        pass
```

## 🎨 Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 88 characters (Black default)
- **Imports**: Use `isort` for import organization
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings

### Code Formatting

```bash
# Format code
black textclassify/

# Sort imports
isort textclassify/

# Check style
flake8 textclassify/

# Type checking
mypy textclassify/
```

### Example Code Style

```python
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ExampleClassifier:
    """Example classifier following project conventions.
    
    Args:
        config: Model configuration object
        timeout: Request timeout in seconds
        
    Example:
        >>> config = ModelConfig(model_name="example")
        >>> classifier = ExampleClassifier(config)
        >>> result = classifier.predict(["Hello world"])
    """
    
    def __init__(
        self, 
        config: ModelConfig, 
        timeout: Optional[float] = None
    ) -> None:
        self.config = config
        self.timeout = timeout or 30.0
        
    async def predict(
        self, 
        texts: List[str]
    ) -> List[str]:
        """Predict labels for input texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted labels
            
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
            
        # Implementation here
        return ["prediction"] * len(texts)
```

## 📚 Documentation

### Documentation Types

1. **API Documentation**: Docstrings for all public functions
2. **Examples**: Working code examples in `examples/`
3. **README**: Keep updated with new features
4. **Tutorials**: Step-by-step guides for complex features

### Docstring Format

```python
def classify_text(
    text: str, 
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0
) -> ClassificationResult:
    """Classify a single text using the specified model.
    
    This function performs text classification using the configured
    model and returns detailed results including confidence scores.
    
    Args:
        text: The text to classify
        model: Name of the model to use for classification
        temperature: Sampling temperature (0.0 for deterministic)
        
    Returns:
        ClassificationResult object containing predictions and metadata
        
    Raises:
        ValueError: If text is empty or model is not supported
        APIError: If the model API returns an error
        
    Example:
        >>> result = classify_text("I love this movie!", model="gpt-4")
        >>> print(result.prediction)
        'positive'
        >>> print(result.confidence)
        0.95
    """
```

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Environment**: Python version, OS, package versions
3. **Reproduction**: Minimal code to reproduce the bug
4. **Expected vs Actual**: What you expected vs what happened
5. **Logs**: Any relevant error messages or logs

### Feature Requests

For feature requests, please include:

1. **Use Case**: Why is this feature needed?
2. **Description**: Detailed description of the proposed feature
3. **Examples**: How would the feature be used?
4. **Alternatives**: What alternatives have you considered?

### Issue Template

```markdown
## Bug Report / Feature Request

**Type**: [Bug / Enhancement / Question]

**Description**:
A clear description of the issue or feature request.

**Environment**:
- Python version:
- TextClassify version:
- OS:
- Relevant dependencies:

**Code to Reproduce** (for bugs):
```python
# Minimal example that reproduces the issue
```

**Expected Behavior**:
What you expected to happen.

**Actual Behavior** (for bugs):
What actually happened.

**Additional Context**:
Any other relevant information.
```

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Email**: [contact@textclassify.com](mailto:contact@textclassify.com)

### Getting Help

1. **Documentation**: Check the README and examples first
2. **Search Issues**: Look for similar problems in existing issues
3. **Ask Questions**: Create a GitHub Discussion for usage questions
4. **Report Bugs**: Create a GitHub Issue for bugs

### Recognition

We believe in recognizing contributors! Contributors will be:

- Added to the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

## 📝 License

By contributing to TextClassify, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to TextClassify! 🚀**

Your contributions help make text classification more accessible and powerful for everyone.
