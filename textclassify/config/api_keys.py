"""API key management for LLM providers."""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import getpass

from ..core.exceptions import ConfigurationError


class APIKeyManager:
    """Manager for API keys with secure storage and retrieval."""
    
    def __init__(self, key_file: Optional[str] = None):
        """Initialize API key manager.
        
        Args:
            key_file: Path to API key file (optional)
        """
        self.key_file = key_file or self._get_default_key_file()
        self._keys = {}
        self._load_keys()
    
    def _get_default_key_file(self) -> str:
        """Get default API key file path.
        
        Returns:
            Default path for API key file
        """
        home_dir = Path.home()
        config_dir = home_dir / '.textclassify'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'api_keys.json')
    
    def _load_keys(self) -> None:
        """Load API keys from file."""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'r') as f:
                    self._keys = json.load(f)
            else:
                self._keys = {}
        except Exception as e:
            print(f"Warning: Failed to load API keys: {str(e)}")
            self._keys = {}
    
    def _save_keys(self) -> None:
        """Save API keys to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            
            with open(self.key_file, 'w') as f:
                json.dump(self._keys, f, indent=2)
            
            # Set restrictive permissions (Unix-like systems only)
            try:
                os.chmod(self.key_file, 0o600)
            except (OSError, AttributeError):
                pass  # Windows or permission error
                
        except Exception as e:
            raise ConfigurationError(f"Failed to save API keys: {str(e)}")
    
    def set_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'claude')
            api_key: API key string
        """
        if not api_key or not api_key.strip():
            raise ConfigurationError("API key cannot be empty")
        
        self._keys[provider.lower()] = api_key.strip()
        self._save_keys()
    
    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            API key or None if not found
        """
        # First check stored keys
        key = self._keys.get(provider.lower())
        
        if key:
            return key
        
        # Then check environment variables
        env_var_names = [
            f"{provider.upper()}_API_KEY",
            f"TEXTCLASSIFY_{provider.upper()}_API_KEY",
            f"{provider.upper()}_KEY"
        ]
        
        for env_var in env_var_names:
            key = os.getenv(env_var)
            if key:
                return key.strip()
        
        return None
    
    def remove_key(self, provider: str) -> bool:
        """Remove API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if key was removed, False if not found
        """
        provider_lower = provider.lower()
        if provider_lower in self._keys:
            del self._keys[provider_lower]
            self._save_keys()
            return True
        return False
    
    def list_providers(self) -> list:
        """List providers with stored API keys.
        
        Returns:
            List of provider names
        """
        return list(self._keys.keys())
    
    def has_key(self, provider: str) -> bool:
        """Check if API key exists for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get_key(provider) is not None
    
    def prompt_for_key(self, provider: str, save: bool = True) -> Optional[str]:
        """Prompt user for API key.
        
        Args:
            provider: Provider name
            save: Whether to save the key after prompting
            
        Returns:
            API key or None if cancelled
        """
        try:
            print(f"\nAPI key required for {provider}")
            print(f"You can also set the environment variable {provider.upper()}_API_KEY")
            
            api_key = getpass.getpass(f"Enter {provider} API key (input hidden): ")
            
            if api_key and api_key.strip():
                api_key = api_key.strip()
                if save:
                    self.set_key(provider, api_key)
                    print(f"API key saved for {provider}")
                return api_key
            else:
                print("No API key provided")
                return None
                
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled")
            return None
    
    def validate_key_format(self, provider: str, api_key: str) -> bool:
        """Validate API key format for a provider.
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            True if format appears valid, False otherwise
        """
        if not api_key or not api_key.strip():
            return False
        
        api_key = api_key.strip()
        provider_lower = provider.lower()
        
        # Basic format validation
        if provider_lower == 'openai':
            return api_key.startswith('sk-') and len(api_key) > 20
        elif provider_lower == 'claude':
            return api_key.startswith('sk-ant-') and len(api_key) > 20
        elif provider_lower == 'gemini':
            return len(api_key) > 20  # Google API keys vary in format
        elif provider_lower == 'deepseek':
            return api_key.startswith('sk-') and len(api_key) > 20
        else:
            # For unknown providers, just check it's not empty
            return len(api_key) > 5
    
    def setup_provider(self, provider: str) -> bool:
        """Interactive setup for a provider's API key.
        
        Args:
            provider: Provider name
            
        Returns:
            True if setup successful, False otherwise
        """
        print(f"\nSetting up {provider} API key...")
        
        # Check if key already exists
        existing_key = self.get_key(provider)
        if existing_key:
            print(f"API key for {provider} already exists")
            response = input("Do you want to replace it? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return True
        
        # Prompt for new key
        api_key = self.prompt_for_key(provider, save=False)
        
        if not api_key:
            return False
        
        # Validate format
        if not self.validate_key_format(provider, api_key):
            print(f"Warning: API key format for {provider} appears invalid")
            response = input("Do you want to save it anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return False
        
        # Save the key
        try:
            self.set_key(provider, api_key)
            print(f"API key for {provider} saved successfully")
            return True
        except Exception as e:
            print(f"Failed to save API key: {str(e)}")
            return False
    
    def get_all_keys(self) -> Dict[str, str]:
        """Get all stored API keys.
        
        Returns:
            Dictionary of provider -> API key
        """
        all_keys = {}
        
        # Get stored keys
        all_keys.update(self._keys)
        
        # Check environment variables for missing keys
        providers = ['openai', 'claude', 'gemini', 'deepseek']
        for provider in providers:
            if provider not in all_keys:
                env_key = self.get_key(provider)
                if env_key:
                    all_keys[provider] = env_key
        
        return all_keys
    
    def clear_all_keys(self) -> None:
        """Clear all stored API keys."""
        self._keys = {}
        self._save_keys()
        print("All API keys cleared")

