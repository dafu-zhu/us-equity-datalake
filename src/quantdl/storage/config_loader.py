"""
Storage Configuration Loader
Loads S3 client and transfer settings from configs/storage.yaml

Note: Fundamental data fields are now managed via approved_mapping.yaml
      using concept-based extraction. This config only handles S3 settings.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class UploadConfig:

    def __init__(self, config_path: str="configs/storage.yaml"):
        """
        Initialize the config loader.

        :param config_path: Path to storage.yaml
        """
        self.config_path = Path(config_path)
        self._config = None

    def load(self):
        """Load configuration from storage.yaml"""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def transfer(self) -> Dict[str, Any]:
        """Get transfer config"""
        if self._config is None:
            self.load()
        if self._config is None:
            raise ValueError("Failed to load configuration")
        return self._config.get('transfer', {})

    @property
    def client(self) -> Dict[str, Any]:
        """Get client config"""
        if self._config is None:
            self.load()
        if self._config is None:
            raise ValueError("Failed to load configuration")
        return self._config.get('client', {})