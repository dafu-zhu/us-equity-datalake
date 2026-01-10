"""
Unit tests for storage.config_loader module
Tests configuration loading from YAML files
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from quantdl.storage.config_loader import UploadConfig


class TestUploadConfig:
    """Test UploadConfig class"""

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file for testing"""
        config_data = {
            'client': {
                'region_name': 'us-east-1',
                'max_pool_connections': 50
            },
            'transfer': {
                'multipart_threshold': 8388608,  # 8MB
                'max_concurrency': 10
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_initialization_with_custom_path(self, sample_config_file):
        """Test initialization with custom config path"""
        config = UploadConfig(config_path=sample_config_file)
        assert config.config_path == Path(sample_config_file)
        assert config._config is None  # Not loaded yet

    def test_initialization_with_default_path(self):
        """Test initialization with default path"""
        config = UploadConfig()
        assert config.config_path == Path("configs/storage.yaml")

    def test_load_config(self, sample_config_file):
        """Test loading configuration from file"""
        config = UploadConfig(config_path=sample_config_file)
        config.load()

        assert config._config is not None
        assert 'client' in config._config
        assert 'transfer' in config._config

    def test_client_property(self, sample_config_file):
        """Test accessing client configuration"""
        config = UploadConfig(config_path=sample_config_file)
        client_config = config.client

        assert client_config['region_name'] == 'us-east-1'
        assert client_config['max_pool_connections'] == 50

    def test_transfer_property(self, sample_config_file):
        """Test accessing transfer configuration"""
        config = UploadConfig(config_path=sample_config_file)
        transfer_config = config.transfer

        assert transfer_config['multipart_threshold'] == 8388608
        assert transfer_config['max_concurrency'] == 10

    def test_lazy_loading(self, sample_config_file):
        """Test that config is loaded lazily on first access"""
        config = UploadConfig(config_path=sample_config_file)

        # Config should not be loaded yet
        assert config._config is None

        # Access property triggers load
        _ = config.client
        assert config._config is not None

    def test_property_returns_empty_dict_for_missing_keys(self, sample_config_file):
        """Test that properties return empty dict for missing keys"""
        # Create config with missing keys
        config_data = {'client': {'region_name': 'us-west-2'}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = UploadConfig(config_path=temp_path)

            # Client exists
            assert config.client['region_name'] == 'us-west-2'

            # Transfer doesn't exist, should return empty dict
            assert config.transfer == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name

        try:
            config = UploadConfig(config_path=temp_path)

            with pytest.raises(yaml.YAMLError):
                config.load()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_missing_config_file(self):
        """Test handling of missing config file"""
        config = UploadConfig(config_path="nonexistent/path/config.yaml")

        with pytest.raises(FileNotFoundError):
            config.load()

    def test_empty_config_file(self):
        """Test handling of empty config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write empty file
            f.write("")
            temp_path = f.name

        try:
            config = UploadConfig(config_path=temp_path)
            config.load()

            # Empty file should load as None or empty dict
            assert config._config is None or config._config == {}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_multiple_loads(self, sample_config_file):
        """Test that config can be loaded multiple times"""
        config = UploadConfig(config_path=sample_config_file)

        # First load
        config.load()
        first_config = config._config

        # Second load
        config.load()
        second_config = config._config

        # Both should have same content
        assert first_config == second_config

    def test_config_with_nested_structure(self):
        """Test config with nested YAML structure"""
        config_data = {
            'client': {
                'region_name': 'us-east-1',
                'nested': {
                    'deeply': {
                        'value': 42
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = UploadConfig(config_path=temp_path)
            client_config = config.client

            # Access nested values
            assert client_config['nested']['deeply']['value'] == 42
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_failed_load_raises_value_error(self):
        """Test that accessing properties after failed load raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            config = UploadConfig(config_path=temp_path)
            config.load()

            # _config should be None after loading empty file
            if config._config is None:
                with pytest.raises(ValueError, match="Failed to load configuration"):
                    _ = config.transfer
        finally:
            Path(temp_path).unlink(missing_ok=True)
