"""Unit tests for core types."""

import pytest
from textclassify.core.types import (
    ClassificationType, ModelType, LLMProvider,
    TrainingData, ModelConfig, EnsembleConfig, ClassificationResult
)


class TestEnums:
    """Test enum classes."""
    
    def test_classification_type(self):
        """Test ClassificationType enum."""
        assert ClassificationType.MULTI_CLASS.value == "multi_class"
        assert ClassificationType.MULTI_LABEL.value == "multi_label"
    
    def test_model_type(self):
        """Test ModelType enum."""
        assert ModelType.LLM.value == "llm"
        assert ModelType.TRADITIONAL_ML.value == "traditional_ml"
    
    def test_llm_provider(self):
        """Test LLMProvider enum."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.CLAUDE.value == "claude"
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.DEEPSEEK.value == "deepseek"


class TestTrainingData:
    """Test TrainingData class."""
    
    def test_multiclass_training_data(self):
        """Test multi-class training data."""
        texts = ["text1", "text2", "text3"]
        labels = ["label1", "label2", "label1"]
        
        data = TrainingData(
            texts=texts,
            labels=labels,
            classification_type=ClassificationType.MULTI_CLASS
        )
        
        assert data.texts == texts
        assert data.labels == labels
        assert data.classification_type == ClassificationType.MULTI_CLASS
        assert len(data) == 3
        assert data.get_classes() == ["label1", "label2"]
    
    def test_multilabel_training_data(self):
        """Test multi-label training data."""
        texts = ["text1", "text2"]
        labels = [["label1", "label2"], ["label2", "label3"]]
        
        data = TrainingData(
            texts=texts,
            labels=labels,
            classification_type=ClassificationType.MULTI_LABEL
        )
        
        assert data.texts == texts
        assert data.labels == labels
        assert data.classification_type == ClassificationType.MULTI_LABEL
        assert len(data) == 2
        assert set(data.get_classes()) == {"label1", "label2", "label3"}
    
    def test_training_data_validation(self):
        """Test training data validation."""
        with pytest.raises(ValueError):
            # Mismatched lengths
            TrainingData(
                texts=["text1", "text2"],
                labels=["label1"],
                classification_type=ClassificationType.MULTI_CLASS
            )


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_llm_model_config(self):
        """Test LLM model configuration."""
        config = ModelConfig(
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            api_key="test-key",
            parameters={"temperature": 0.1}
        )
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.model_type == ModelType.LLM
        assert config.api_key == "test-key"
        assert config.parameters["temperature"] == 0.1
        assert config.batch_size == 32  # default
    
    def test_ml_model_config(self):
        """Test ML model configuration."""
        config = ModelConfig(
            model_name="roberta-base",
            model_type=ModelType.TRADITIONAL_ML,
            parameters={"learning_rate": 2e-5}
        )
        
        assert config.model_name == "roberta-base"
        assert config.model_type == ModelType.TRADITIONAL_ML
        assert config.api_key is None
        assert config.parameters["learning_rate"] == 2e-5


class TestEnsembleConfig:
    """Test EnsembleConfig class."""
    
    def test_voting_ensemble_config(self):
        """Test voting ensemble configuration."""
        model_configs = [
            ModelConfig("model1", ModelType.LLM),
            ModelConfig("model2", ModelType.LLM)
        ]
        
        config = EnsembleConfig(
            models=model_configs,
            ensemble_method="voting"
        )
        
        assert len(config.models) == 2
        assert config.ensemble_method == "voting"
        assert config.require_all_models is False  # default
    
    def test_weighted_ensemble_config(self):
        """Test weighted ensemble configuration."""
        model_configs = [
            ModelConfig("model1", ModelType.LLM),
            ModelConfig("model2", ModelType.LLM)
        ]
        
        config = EnsembleConfig(
            models=model_configs,
            ensemble_method="weighted",
            weights=[0.7, 0.3]
        )
        
        assert config.weights == [0.7, 0.3]
        assert config.ensemble_method == "weighted"
    
    def test_routing_ensemble_config(self):
        """Test routing ensemble configuration."""
        model_configs = [
            ModelConfig("model1", ModelType.LLM),
            ModelConfig("model2", ModelType.LLM)
        ]
        
        routing_rules = {"class1": "model1", "class2": "model2"}
        
        config = EnsembleConfig(
            models=model_configs,
            ensemble_method="routing",
            routing_rules=routing_rules
        )
        
        assert config.routing_rules == routing_rules
        assert config.ensemble_method == "routing"


class TestClassificationResult:
    """Test ClassificationResult class."""
    
    def test_multiclass_result(self):
        """Test multi-class classification result."""
        result = ClassificationResult(
            predictions=["class1", "class2", "class1"],
            classification_type=ClassificationType.MULTI_CLASS,
            model_name="test-model",
            model_type=ModelType.LLM,
            processing_time=1.5
        )
        
        assert result.predictions == ["class1", "class2", "class1"]
        assert result.classification_type == ClassificationType.MULTI_CLASS
        assert result.model_name == "test-model"
        assert result.processing_time == 1.5
        assert result.probabilities is None
    
    def test_multilabel_result(self):
        """Test multi-label classification result."""
        predictions = [["class1", "class2"], ["class2"], ["class1", "class3"]]
        
        result = ClassificationResult(
            predictions=predictions,
            classification_type=ClassificationType.MULTI_LABEL,
            model_name="test-model",
            model_type=ModelType.LLM
        )
        
        assert result.predictions == predictions
        assert result.classification_type == ClassificationType.MULTI_LABEL
    
    def test_result_with_probabilities(self):
        """Test result with probabilities."""
        predictions = ["class1", "class2"]
        probabilities = [
            {"class1": 0.8, "class2": 0.2},
            {"class1": 0.3, "class2": 0.7}
        ]
        
        result = ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            classification_type=ClassificationType.MULTI_CLASS,
            model_name="test-model",
            model_type=ModelType.LLM
        )
        
        assert result.probabilities == probabilities
        assert len(result.probabilities) == len(result.predictions)


if __name__ == "__main__":
    pytest.main([__file__])

