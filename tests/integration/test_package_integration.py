"""Integration tests for package functionality."""
import sys
import importlib
sys.path.insert(0, '.')

# Force clear all cached modules
modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('textclassify')]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

import pytest
import sys
import os

from textclassify.core.types import ModelConfig

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestPackageImports:
    """Test that all package components can be imported."""
    
    def test_core_imports(self):
        """Test core module imports."""
        from textclassify.core.types import (
            ClassificationType, ModelType, LLMProvider,
            TrainingData, ModelConfig, EnsembleConfig, ClassificationResult
        )
        from textclassify.core.base import BaseClassifier
        from textclassify.core.exceptions import (
            TextClassifyError, ConfigurationError, ModelError, EnsembleError
        )
        
        # Test enum values
        assert ClassificationType.MULTI_CLASS.value == "multi_class"
        assert ModelType.LLM.value == "llm"
        assert LLMProvider.OPENAI.value == "openai"
    
    def test_llm_imports(self):
        """Test LLM module imports."""
        from textclassify.llm.base import BaseLLMClassifier
        from textclassify.llm.openai_classifier import OpenAIClassifier
        from textclassify.llm.claude_classifier import ClaudeClassifier
        from textclassify.llm.gemini_classifier import GeminiClassifier
        from textclassify.llm.deepseek_classifier import DeepSeekClassifier
        from textclassify.llm.prompts import (
            MultiClassPromptTemplate, MultiLabelPromptTemplate
        )
        
        # Test that classes exist
        assert issubclass(OpenAIClassifier, BaseLLMClassifier)
        assert issubclass(ClaudeClassifier, BaseLLMClassifier)
    
    def test_ml_imports(self):
        """Test ML module imports."""
        from textclassify.ml.base import BaseMLClassifier
        from textclassify.ml.preprocessing import TextPreprocessor
        
        # RoBERTa classifier might not be available without transformers
        try:
            from textclassify.ml.roberta_classifier import RoBERTaClassifier
            assert issubclass(RoBERTaClassifier, BaseMLClassifier)
        except ImportError:
            # Expected if transformers not installed
            pass
    
    def test_ensemble_imports(self):
        """Test ensemble module imports."""
        from textclassify.ensemble.base import BaseEnsemble
        from textclassify.ensemble.voting import VotingEnsemble
        from textclassify.ensemble.weighted import WeightedEnsemble
        from textclassify.ensemble.routing import ClassRoutingEnsemble
        
        # Test inheritance
        assert issubclass(VotingEnsemble, BaseEnsemble)
        assert issubclass(WeightedEnsemble, BaseEnsemble)
        assert issubclass(ClassRoutingEnsemble, BaseEnsemble)
    
    def test_config_imports(self):
        """Test config module imports."""
        from textclassify.config import Config, APIKeyManager, load_config, save_config
        
        # Test basic functionality
        config = Config()
        assert config.get('general.default_batch_size') == 32
        
        api_manager = APIKeyManager()
        assert hasattr(api_manager, 'set_key')
        assert hasattr(api_manager, 'get_key')
    
    def test_utils_imports(self):
        """Test utils module imports."""
        from textclassify.utils import (
            DataLoader, split_data, balance_data,
            ClassificationMetrics, evaluate_predictions,
            setup_logging, get_logger
        )
        
        # Test basic functionality
        logger = get_logger("test")
        assert logger.name == "textclassify.test"
    
    def test_main_package_imports(self):
        """Test main package imports."""
        import textclassify
        
        # Test that main classes are available
        from textclassify import (
            OpenAIClassifier, ClaudeClassifier, GeminiClassifier, DeepSeekClassifier,
            VotingEnsemble, WeightedEnsemble, ClassRoutingEnsemble,
            Config, APIKeyManager,
            ClassificationType, ModelType, LLMProvider
        )
        
        # Test version
        assert hasattr(textclassify, '__version__')
    
    def test_openai_classifier_on_ag_news(self):
        """Test OpenAI classifier on AG News dataset."""
        import pandas as pd
        from textclassify.llm.openai_classifier import OpenAIClassifier
        from textclassify.core.types import ModelConfig
        
        # Load your dataset
        train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_train_balanced.csv")
        test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_test_balanced.csv")
        
        # Configure the model
        from textclassify.core.types import ModelConfig, ModelType
        config = ModelConfig(
            model_name="gpt-5-2025-08-07",  # Required
            model_type=ModelType.LLM,    # Required
            parameters={
                "model": "gpt-5-2025-08-07",  # The actual model to use
                "temperature": 0.1,
                "max_completion_tokens": 150,
                "top_p": 1.0
            }
        )
        
        # Initialize classifier
        classifier = OpenAIClassifier(
            config=config,
            text_column='description',
            label_columns=["label_1", "label_2", "label_3", "label_4"],
            enable_cache=True,
            cache_dir="cache/experimente/ag_news_klassifikation",
            multi_label=False
        )
        
        # Test prediction (use small sample for testing)
        result = classifier.predict(
            train_df=train_df,  # Few-shot examples
            test_df=test_df     # Test samples
        )
        
        # Assert results
        assert len(result.predictions) == 3
        assert result.model_name == "gpt-5-2025-08-07"  # Should match parameters["model"]
        assert result.model_type.value == "llm"


class TestBasicFunctionality:
    """Test basic package functionality."""
    
    def test_training_data_creation(self):
        """Test creating training data."""
        from textclassify.core.types import TrainingData, ClassificationType
        
        # Multi-class data
        data = TrainingData(
            texts=["text1", "text2", "text3"],
            labels=["label1", "label2", "label1"],
            classification_type=ClassificationType.MULTI_CLASS
        )
        
        assert len(data) == 3
        assert data.get_classes() == ["label1", "label2"]
        
        # Multi-label data
        data_ml = TrainingData(
            texts=["text1", "text2"],
            labels=[["label1", "label2"], ["label2", "label3"]],
            classification_type=ClassificationType.MULTI_LABEL
        )
        
        assert len(data_ml) == 2
        assert set(data_ml.get_classes()) == {"label1", "label2", "label3"}
    
    def test_model_config_creation(self):
        """Test creating model configurations."""
        from textclassify.core.types import ModelConfig, ModelType
        
        # LLM config
        llm_config = ModelConfig(
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            api_key="test-key",
            parameters={"temperature": 0.1}
        )
        
        assert llm_config.model_name == "gpt-3.5-turbo"
        assert llm_config.model_type == ModelType.LLM
        assert llm_config.parameters["temperature"] == 0.1
        
        # ML config
        ml_config = ModelConfig(
            model_name="roberta-base",
            model_type=ModelType.TRADITIONAL_ML,
            parameters={"learning_rate": 2e-5}
        )
        
        assert ml_config.model_name == "roberta-base"
        assert ml_config.model_type == ModelType.TRADITIONAL_ML
    
    def test_ensemble_config_creation(self):
        """Test creating ensemble configurations."""
        from textclassify.core.types import EnsembleConfig, ModelConfig, ModelType
        
        model_configs = [
            ModelConfig("model1", ModelType.LLM),
            ModelConfig("model2", ModelType.LLM)
        ]
        
        ensemble_config = EnsembleConfig(
            models=model_configs,
            ensemble_method="voting"
        )
        
        assert len(ensemble_config.models) == 2
        assert ensemble_config.ensemble_method == "voting"
    
    def test_data_loader_functionality(self):
        """Test data loader functionality."""
        from textclassify.utils import DataLoader
        from textclassify.core.types import TrainingData, ClassificationType
        
        # Create sample data
        texts = ["text1", "text2", "text3"]
        labels = ["label1", "label2", "label1"]
        
        training_data = DataLoader.from_lists(
            texts=texts,
            labels=labels,
            classification_type=ClassificationType.MULTI_CLASS
        )
        
        assert isinstance(training_data, TrainingData)
        assert training_data.texts == texts
        assert training_data.labels == labels
    
    def test_configuration_management(self):
        """Test configuration management."""
        from textclassify.config import Config
        
        config = Config()
        
        # Test default values
        assert config.get('general.default_batch_size') == 32
        assert config.get('llm.default_provider') == 'openai'
        
        # Test setting values
        config.set('test.value', 'hello')
        assert config.get('test.value') == 'hello'
        
        # Test nested values
        config.set('nested.deep.value', 42)
        assert config.get('nested.deep.value') == 42
    
    def test_metrics_functionality(self):
        """Test metrics functionality."""
        from textclassify.utils import ClassificationMetrics
        from textclassify.core.types import ClassificationType
        
        # Multi-class metrics
        metrics = ClassificationMetrics(ClassificationType.MULTI_CLASS)
        
        y_true = ["class1", "class2", "class1", "class2"]
        y_pred = ["class1", "class2", "class1", "class1"]
        
        result = metrics.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in result
        assert 'macro_f1' in result
        assert 'per_class_metrics' in result
        assert result['accuracy'] == 0.75  # 3 out of 4 correct


class TestErrorHandling:
    """Test error handling."""
    
    def test_training_data_validation(self):
        """Test training data validation."""
        from textclassify.core.types import TrainingData, ClassificationType
        
        with pytest.raises(ValueError):
            # Mismatched lengths
            TrainingData(
                texts=["text1", "text2"],
                labels=["label1"],
                classification_type=ClassificationType.MULTI_CLASS
            )
    
    def test_configuration_errors(self):
        """Test configuration error handling."""
        from textclassify.config import Config
        from textclassify.core.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            # Non-existent config file
            Config("nonexistent_file.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

