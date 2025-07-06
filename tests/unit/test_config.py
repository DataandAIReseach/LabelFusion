"""Unit tests for configuration management."""

import pytest
import tempfile
import os
import json
from pathlib import Path

from textclassify.config import Config, APIKeyManager
from textclassify.core.types import ModelType, LLMProvider
from textclassify.core.exceptions import ConfigurationError


class TestConfig:
    """Test Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.get('general.default_batch_size') == 32
        assert config.get('llm.default_provider') == 'openai'
        assert config.get('ensemble.default_method') == 'voting'
    
    def test_config_get_set(self):
        """Test getting and setting configuration values."""
        config = Config()
        
        # Test setting and getting
        config.set('test.value', 'hello')
        assert config.get('test.value') == 'hello'
        
        # Test nested setting
        config.set('nested.deep.value', 42)
        assert config.get('nested.deep.value') == 42
        
        # Test default value
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_config_save_load_json(self):
        """Test saving and loading JSON configuration."""
        config = Config()
        config.set('test.value', 'test_data')
        config.set('test.number', 123)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            config.save(config_path)
            assert os.path.exists(config_path)
            
            # Load configuration
            new_config = Config(config_path)
            assert new_config.get('test.value') == 'test_data'
            assert new_config.get('test.number') == 123
            
        finally:
            os.unlink(config_path)
    
    def test_create_model_config(self):
        """Test creating model configuration."""
        config = Config()
        config.set('api_keys.openai', 'test-key')
        
        model_config = config.create_model_config(
            model_name="gpt-3.5-turbo",
            model_type=ModelType.LLM,
            provider=LLMProvider.OPENAI
        )
        
        assert model_config.model_name == "gpt-3.5-turbo"
        assert model_config.model_type == ModelType.LLM
        assert model_config.api_key == 'test-key'
        assert model_config.batch_size == 32  # default
    
    def test_create_ensemble_config(self):
        """Test creating ensemble configuration."""
        config = Config()
        
        model_configs = [
            config.create_model_config("model1", ModelType.LLM),
            config.create_model_config("model2", ModelType.LLM)
        ]
        
        ensemble_config = config.create_ensemble_config(
            models=model_configs,
            ensemble_method="voting"
        )
        
        assert len(ensemble_config.models) == 2
        assert ensemble_config.ensemble_method == "voting"
    
    def test_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(ConfigurationError):
            Config("nonexistent_file.json")


class TestAPIKeyManager:
    """Test APIKeyManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary key file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.key_file = self.temp_file.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.key_file):
            os.unlink(self.key_file)
    
    def test_set_get_key(self):
        """Test setting and getting API keys."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            # Set key
            manager.set_key("openai", "test-openai-key")
            
            # Get key
            key = manager.get_key("openai")
            assert key == "test-openai-key"
            
            # Test case insensitive
            key = manager.get_key("OPENAI")
            assert key == "test-openai-key"
            
        finally:
            self.tearDown()
    
    def test_remove_key(self):
        """Test removing API keys."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            # Set and remove key
            manager.set_key("claude", "test-claude-key")
            assert manager.has_key("claude")
            
            removed = manager.remove_key("claude")
            assert removed is True
            assert not manager.has_key("claude")
            
            # Try removing non-existent key
            removed = manager.remove_key("nonexistent")
            assert removed is False
            
        finally:
            self.tearDown()
    
    def test_list_providers(self):
        """Test listing providers with keys."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            # Initially empty
            assert manager.list_providers() == []
            
            # Add keys
            manager.set_key("openai", "key1")
            manager.set_key("claude", "key2")
            
            providers = manager.list_providers()
            assert "openai" in providers
            assert "claude" in providers
            assert len(providers) == 2
            
        finally:
            self.tearDown()
    
    def test_environment_variable_fallback(self):
        """Test environment variable fallback."""
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = "env-openai-key"
        
        try:
            self.setUp()
            manager = APIKeyManager(self.key_file)
            
            # Should get from environment
            key = manager.get_key("openai")
            assert key == "env-openai-key"
            
        finally:
            # Clean up environment
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            self.tearDown()
    
    def test_validate_key_format(self):
        """Test API key format validation."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            # Valid OpenAI key format
            assert manager.validate_key_format("openai", "sk-1234567890abcdef1234567890abcdef")
            
            # Invalid OpenAI key format
            assert not manager.validate_key_format("openai", "invalid-key")
            assert not manager.validate_key_format("openai", "")
            
            # Valid Claude key format
            assert manager.validate_key_format("claude", "sk-ant-1234567890abcdef1234567890abcdef")
            
            # Invalid Claude key format
            assert not manager.validate_key_format("claude", "sk-1234567890abcdef")
            
        finally:
            self.tearDown()
    
    def test_empty_key_validation(self):
        """Test empty key validation."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            with pytest.raises(ConfigurationError):
                manager.set_key("openai", "")
            
            with pytest.raises(ConfigurationError):
                manager.set_key("openai", "   ")
            
        finally:
            self.tearDown()
    
    def test_get_all_keys(self):
        """Test getting all keys."""
        self.setUp()
        try:
            manager = APIKeyManager(self.key_file)
            
            # Set some keys
            manager.set_key("openai", "openai-key")
            manager.set_key("claude", "claude-key")
            
            # Set environment variable
            os.environ["GEMINI_API_KEY"] = "gemini-env-key"
            
            all_keys = manager.get_all_keys()
            
            assert all_keys["openai"] == "openai-key"
            assert all_keys["claude"] == "claude-key"
            assert all_keys["gemini"] == "gemini-env-key"
            
        finally:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            self.tearDown()


if __name__ == "__main__":
    pytest.main([__file__])

