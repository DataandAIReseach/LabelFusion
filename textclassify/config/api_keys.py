"""API key management for LLM providers."""

from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv


class APIKeyManager:
    """Simplified API key management."""
    
    def __init__(self):
        load_dotenv()  # Load .env file if it exists
        self.config_dir = Path.home() / '.textclassify'
        self.config_dir.mkdir(exist_ok=True)

    def get_key(self, provider: str) -> Optional[str]:
        """Get API key with environment variable priority."""
        # 1. Check environment first
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key

        # 2. Check config file as fallback
        config_file = self.config_dir / 'config.env'
        if config_file.exists():
            load_dotenv(config_file)
            return os.getenv(f"{provider.upper()}_API_KEY")

        return None

    def set_key(self, provider: str, key: str) -> None:
        """Save key to config file."""
        config_file = self.config_dir / 'config.env'
        with open(config_file, 'a') as f:
            f.write(f"{provider.upper()}_API_KEY={key}\n")
        os.chmod(config_file, 0o600)

