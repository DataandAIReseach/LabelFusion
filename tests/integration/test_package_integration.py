"""Integration tests for package functionality."""
import sys
import importlib
import pytest
import os

# Force clear all cached modules
modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('textclassify')]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

from textclassify.core.types import ModelConfig

# Add the package to the path for testing
sys.path.insert(0, '.')
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
    
    # def test_openai_classifier_on_ag_news(self):
    #     """Test OpenAI classifier on AG News dataset."""
    #     import pandas as pd
    #     from textclassify.llm.openai_classifier import OpenAIClassifier
    #     from textclassify.core.types import ModelConfig
        
    #     # Load your dataset
    #     train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_train_balanced.csv")
    #     test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_test_balanced.csv")
        
    #     # Configure the model
    #     from textclassify.core.types import ModelConfig, ModelType
    #     config = ModelConfig(
    #         model_name="gpt-5-2025-08-07",  # Required
    #         model_type=ModelType.LLM,    # Required
    #         parameters={
    #             "model": "gpt-5-2025-08-07",  # The actual model to use
    #             "temperature": 0.1,
    #             "max_completion_tokens": 150,
    #             "top_p": 1.0
    #         }
    #     )
        
    #     # Initialize classifier
    #     classifier = OpenAIClassifier(
    #         config=config,
    #         text_column='description',
    #         label_columns=["label_1", "label_2", "label_3", "label_4"],
    #         enable_cache=True,
    #         cache_dir="cache/experimente/ag_news_klassifikation",
    #         multi_label=False
    #     )
        
    #     # Test prediction (use small sample for testing)
    #     result = classifier.predict(
    #         train_df=train_df,  # Few-shot examples
    #         test_df=test_df     # Test samples
    #     )
        
    #     # Assert results
    #     assert len(result.predictions) == 3
    #     assert result.model_name == "gpt-5-2025-08-07"  # Should match parameters["model"]
    #     assert result.model_type.value == "llm"
    
    
    # def test_openai_classifier_on_goemotions(self):
    #     """Test OpenAI classifier on GoEmotions multi-label dataset."""
    #     import pandas as pd
    #     from textclassify.llm.openai_classifier import OpenAIClassifier
    #     from textclassify.core.types import ModelConfig, ModelType
        
    #     # Load GoEmotions dataset
    #     train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/goemotions/goemotions_all_train_balanced.csv")
    #     test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/goemotions/goemotions_all_test_balanced.csv")
        
    #     # Get emotion columns (excluding metadata columns)
    #     label_columns = [
    #     "admiration","amusement","anger","annoyance","approval","caring",
    #     "confusion","curiosity","desire","disappointment","disapproval","disgust",
    #     "embarrassment","excitement","fear","gratitude","grief","joy","love",
    #     "nervousness","optimism","pride","realization","relief","remorse",
    #     "sadness","surprise","neutral"
    #     ]
        
    #     # Configure the model for multi-label emotion classification
    #     config = ModelConfig(
    #         model_name="gpt-5-2025-08-07",
    #         model_type=ModelType.LLM,
    #         parameters={
    #             "model": "gpt-5-2025-08-07",
    #             "temperature": 0.1,
    #             "max_completion_tokens": 200,
    #             "top_p": 1.0
    #         }
    #     )
        
    #     # Initialize classifier for multi-label classification
    #     classifier = OpenAIClassifier(
    #         config=config,
    #         text_column='text',
    #         label_columns=label_columns,
    #         enable_cache=True,
    #         cache_dir="cache/experimente/goemotions_classifikation",
    #         multi_label=True  # Enable multi-label classification
    #     )
        
    #     # Test prediction
    #     result = classifier.predict(
    #         train_df=train_df,
    #         test_df=test_df
    #     )
        
    #     # Assert results for multi-label classification
    #     assert len(result.predictions) == 5
    #     assert result.model_name == "gpt-5-2025-08-07"
    #     assert result.model_type.value == "llm"
        
    #     # Validate multi-label prediction format
    #     for prediction in result.predictions:
    #         # Each prediction should be a binary vector of length = number of emotions
    #         assert len(prediction) == len(label_columns)
    #         # All values should be 0 or 1
    #         assert all(val in [0, 1] for val in prediction)
    
    # def test_openai_classifier_on_goemotions_advanced(self):
    #     """Test OpenAI classifier on GoEmotions multi-label dataset with advanced features."""
    #     import pandas as pd
    #     from textclassify.llm.openai_classifier import OpenAIClassifier
    #     from textclassify.core.types import ModelConfig, ModelType
        
    #     # Load GoEmotions dataset (small sample for testing)
    #     train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/goemotions/goemotions_all_train_balanced.csv")
    #     test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/goemotions/goemotions_all_test_balanced.csv")
        
    #     # Get emotion columns
    #     label_columns = [
    #         "admiration","amusement","anger","annoyance","approval","caring",
    #         "confusion","curiosity","desire","disappointment","disapproval","disgust",
    #         "embarrassment","excitement","fear","gratitude","grief","joy","love",
    #         "nervousness","optimism","pride","realization","relief","remorse",
    #         "sadness","surprise","neutral"
    #     ]
        
    #     # Configure advanced OpenAI model
    #     config = ModelConfig(
    #         model_name="gpt-4",
    #         model_type=ModelType.LLM,
    #         parameters={
    #             "model": "gpt-4",
    #             "temperature": 0.1,
    #             "max_completion_tokens": 200,
    #             "top_p": 1.0
    #         }
    #     )
        
    #     # Create OpenAI classifier with advanced caching
    #     classifier = OpenAIClassifier(
    #         config=config,
    #         text_column='text',
    #         label_columns=label_columns,
    #         enable_cache=True,
    #         cache_dir="cache/experimente/goemotions_advanced",
    #         multi_label=True
    #     )
        
    #     # Test prediction with advanced features
    #     result = classifier.predict(
    #         train_df=train_df.head(5),  # Few-shot examples
    #         test_df=test_df.head(3)     # Test samples
    #     )
        
    #     # Assert advanced results
    #     assert len(result.predictions) == 3
    #     assert result.model_name == "gpt-4"
    #     assert result.model_type.value == "llm"
        
    #     # Validate multi-label prediction format
    #     for prediction in result.predictions:
    #         # Each prediction should be a binary vector of length = number of emotions
    #         assert len(prediction) == len(label_columns)
    #         # All values should be 0 or 1
    #         assert all(val in [0, 1] for val in prediction)
        
    #     # Test that caching is working
    #     cache_stats = classifier.cache.get_cache_stats()
    #     assert cache_stats is not None
    
    # def test_roberta_classifier_on_ag_news(self):
    #     """Test RoBERTa classifier on AG News dataset."""
    #     import pandas as pd
    #     from textclassify.ml.roberta_classifier import RoBERTaClassifier
    #     from textclassify.core.types import ModelConfig, ModelType
        
    #     # Load your dataset
    #     train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_train_balanced.csv")
    #     val_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_val_balanced.csv")
    #     test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_test_balanced.csv")
        
    #     # Configure the model
    #     config = ModelConfig(
    #         model_name="roberta-base",  # Required
    #         model_type=ModelType.TRADITIONAL_ML,  # Required
    #         parameters={
    #             "model_name": "roberta-base",  # The actual model to use
    #             "learning_rate": 2e-5,
    #             "num_epochs": 1,  # Use 1 epoch for faster testing
    #             "batch_size": 8,
    #             "max_length": 256
    #         }
    #     )
        
    #     # Initialize classifier
    #     classifier = RoBERTaClassifier(
    #         config=config,
    #         text_column='description',
    #         label_columns=["label_1", "label_2", "label_3", "label_4"],
    #         multi_label=False,
    #         auto_save_path="cache/experimente/ag_news_roberta_model"
    #     )
        
    #     # Use small samples for testing
    #     train_sample = train_df.head(50)  # Use small sample for faster training
    #     val_sample = val_df.head(20)      # Use small validation set from separate file
    #     test_sample = test_df.head(10)    # Use small test set
        
    #     # Train the model
    #     training_result = classifier.fit(train_sample, val_sample)
        
    #     # Test prediction
    #     result = classifier.predict(test_df=test_sample)
        
    #     # Assert training results
    #     assert training_result['model_name'] == 'roberta-base'
    #     assert training_result['training_samples'] == 50
    #     assert training_result['validation_samples'] == 20
    #     assert training_result['num_labels'] == 4
    #     assert training_result['classes'] == ["label_1", "label_2", "label_3", "label_4"]
        
    #     # Assert prediction results
    #     assert len(result.predictions) == 10
    #     assert all(pred in ["label_1", "label_2", "label_3", "label_4"] for pred in result.predictions)
    #     assert classifier.is_trained

    def test_fusion_ensemble_on_ag_news(self):
        """Test Fusion Ensemble combining RoBERTa and OpenAI LLM on AG News dataset."""
        import pandas as pd
        from textclassify.ml.roberta_classifier import RoBERTaClassifier
        from textclassify.llm.openai_classifier import OpenAIClassifier
        from textclassify.ensemble.fusion import FusionEnsemble
        from textclassify.core.types import ModelConfig, ModelType, EnsembleConfig
        
        # Load your dataset (same as RoBERTa test)
        train_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_train_balanced.csv")
        val_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_val_balanced.csv")
        test_df = pd.read_csv("/home/michaelschlee/ownCloud/GIT/classifyfusion/data/ag_news/ag_test_balanced.csv")
        
        # Configure ML model (RoBERTa)
        ml_config = ModelConfig(
            model_name="roberta-base",
            model_type=ModelType.TRADITIONAL_ML,
            parameters={
                "model_name": "roberta-base",
                "learning_rate": 2e-5,
                "num_epochs": 1,  # Fast training for testing
                "batch_size": 8,
                "max_length": 256
            }
        )
        
        # Create RoBERTa classifier
        ml_classifier = RoBERTaClassifier(
            config=ml_config,
            text_column='description',
            label_columns=["label_1", "label_2", "label_3", "label_4"],
            multi_label=False,
            auto_save_path="cache/experimente/fusion_roberta_model"
        )
        
        # Configure LLM model (OpenAI)
        llm_config = ModelConfig(
            model_name="gpt-4o-mini",
            model_type=ModelType.LLM,
            parameters={
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_completion_tokens": 150,
                "top_p": 1.0
            }
        )
        
        # Create OpenAI LLM classifier
        llm_classifier = OpenAIClassifier(
            config=llm_config,
            text_column='description',
            label_columns=["label_1", "label_2", "label_3", "label_4"],
            enable_cache=True,
            cache_dir="cache/experimente/fusion_openai_cache",
            multi_label=False
        )
        
        # Configure Fusion Ensemble
        fusion_config = EnsembleConfig(
            ensemble_method="fusion",
            models=[ml_classifier, llm_classifier],
            parameters={
                "fusion_hidden_dims": [32, 16],  # Smaller network for testing
                "ml_lr": 1e-5,
                "fusion_lr": 1e-3,
                "num_epochs": 5,  # Fewer epochs for testing
                "batch_size": 8,
                "classification_type": "multi_class"
            }
        )
        
        # Create Fusion Ensemble
        fusion_ensemble = FusionEnsemble(fusion_config)
        
        # Add models to ensemble
        fusion_ensemble.add_ml_model(ml_classifier)
        fusion_ensemble.add_llm_model(llm_classifier)
        
        # Use small samples for testing (same as RoBERTa test)
        train_sample = train_df.sample(50, random_state=42)  # Zufällig 50 Zeilen
        val_sample = val_df.sample(5, random_state=42)       # Zufällig 5 Zeilen
        test_sample = test_df.sample(5, random_state=42)     # Zufällig 5 Zeilen

        # Train the fusion ensemble (like RoBERTa fit method)
        training_result = fusion_ensemble.fit(train_sample, val_sample)
        
        # Test prediction (like RoBERTa predict method)
        result = fusion_ensemble.predict(test_df=test_sample)
        
        # Assert training results
        assert training_result['ensemble_method'] == 'fusion'
        assert training_result['training_samples'] == 50
        assert training_result['validation_samples'] == 20
        assert training_result['num_labels'] == 4
        
        print(f"Fusion Ensemble Predictions: {result.predictions}")
        print(f"Training Result: {training_result}")

class TestBasicFunctionality:
    """Test basic package functionality."""
    
    
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
    
    
    def test_configuration_errors(self):
        """Test configuration error handling."""
        from textclassify.config import Config
        from textclassify.core.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            # Non-existent config file
            Config("nonexistent_file.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

