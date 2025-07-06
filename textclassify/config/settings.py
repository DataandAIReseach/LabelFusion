"""Configuration settings management."""

import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.types import ModelConfig, EnsembleConfig, ClassificationType, ModelType, LLMProvider
from ..core.exceptions import ConfigurationError


class Config:
    """Main configuration class for textclassify package."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.data = {}
        
        # Default configuration
        self._set_defaults()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.data = {
            "general": {
                "default_classification_type": "multi_class",
                "default_batch_size": 32,
                "default_timeout": 30.0,
                "default_max_retries": 3,
                "enable_caching": True,
                "cache_dir": "~/.textclassify/cache",
                "log_level": "INFO"
            },
            "llm": {
                "default_provider": "openai",
                "default_models": {
                    "openai": "gpt-3.5-turbo",
                    "claude": "claude-3-haiku-20240307",
                    "gemini": "gemini-1.5-flash",
                    "deepseek": "deepseek-chat"
                },
                "default_parameters": {
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "max_examples": 5
                }
            },
            "ml": {
                "default_model": "roberta-base",
                "default_parameters": {
                    "max_length": 512,
                    "batch_size": 16,
                    "learning_rate": 2e-5,
                    "num_epochs": 3,
                    "warmup_steps": 0,
                    "weight_decay": 0.01
                },
                "preprocessing": {
                    "lowercase": True,
                    "remove_punctuation": False,
                    "remove_numbers": False,
                    "remove_extra_whitespace": True,
                    "min_length": 1,
                    "max_length": None
                }
            },
            "ensemble": {
                "default_method": "voting",
                "default_voting_strategy": "majority",
                "default_threshold": 0.5,
                "require_all_models": False
            },
            "api_keys": {
                "openai": None,
                "claude": None,
                "gemini": None,
                "deepseek": None
            }
        }
    
    def load(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If loading fails
        """
        try:
            config_path = Path(config_path).expanduser()
            
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Merge with defaults
            self._merge_config(self.data, loaded_data)
            self.config_path = str(config_path)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration (optional, uses loaded path if not provided)
            
        Raises:
            ConfigurationError: If saving fails
        """
        save_path = config_path or self.config_path
        
        if not save_path:
            raise ConfigurationError("No configuration path specified")
        
        try:
            save_path = Path(save_path).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.data, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self.data, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {save_path.suffix}")
            
            self.config_path = str(save_path)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def _merge_config(self, base: Dict, update: Dict) -> None:
        """Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            update: Update configuration dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.default_provider')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'llm.default_provider')
            value: Value to set
        """
        keys = key.split('.')
        config = self.data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def create_model_config(
        self,
        model_name: str,
        model_type: Union[str, ModelType],
        provider: Optional[Union[str, LLMProvider]] = None,
        **kwargs
    ) -> ModelConfig:
        """Create a ModelConfig from the current configuration.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            provider: LLM provider (for LLM models)
            **kwargs: Additional parameters
            
        Returns:
            ModelConfig instance
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        # Get default parameters based on model type
        if model_type == ModelType.LLM:
            default_params = self.get('llm.default_parameters', {}).copy()
            if provider:
                if isinstance(provider, str):
                    provider = LLMProvider(provider)
                api_key = self.get(f'api_keys.{provider.value}')
            else:
                api_key = None
        else:
            default_params = self.get('ml.default_parameters', {}).copy()
            api_key = None
        
        # Merge with provided parameters
        default_params.update(kwargs)
        
        return ModelConfig(
            model_name=model_name,
            model_type=model_type,
            parameters=default_params,
            api_key=api_key,
            batch_size=self.get('general.default_batch_size', 32),
            max_retries=self.get('general.default_max_retries', 3),
            timeout=self.get('general.default_timeout', 30.0),
            enable_caching=self.get('general.enable_caching', True)
        )
    
    def create_ensemble_config(
        self,
        models: List[ModelConfig],
        ensemble_method: str = None,
        **kwargs
    ) -> EnsembleConfig:
        """Create an EnsembleConfig from the current configuration.
        
        Args:
            models: List of model configurations
            ensemble_method: Ensemble method to use
            **kwargs: Additional parameters
            
        Returns:
            EnsembleConfig instance
        """
        if ensemble_method is None:
            ensemble_method = self.get('ensemble.default_method', 'voting')
        
        return EnsembleConfig(
            models=models,
            ensemble_method=ensemble_method,
            require_all_models=self.get('ensemble.require_all_models', False),
            **kwargs
        )
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory path.
        
        Returns:
            Path to cache directory
        """
        cache_dir = self.get('general.cache_dir', '~/.textclassify/cache')
        return Path(cache_dir).expanduser()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self.data.copy()


def load_config(config_path: str) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration instance
        config_path: Path to save configuration
    """
    config.save(config_path)


def create_default_config(config_path: str) -> Config:
    """Create a default configuration file.
    
    Args:
        config_path: Path to save the default configuration
        
    Returns:
        Config instance with defaults
    """
    config = Config()
    config.save(config_path)
    return config

