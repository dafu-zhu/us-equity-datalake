"""
Unit tests for storage.s3_client.S3Client class
Tests S3 client configuration and initialization
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml


class TestS3Client:
    """Test S3Client class"""

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_create_boto_config_with_all_optional_params(self, mock_boto_client):
        """Test _create_boto_config with all optional parameters - covers lines 65, 71, 74, 77, 80"""
        from quantdl.storage.s3_client import S3Client

        # Create a temporary config file with all optional parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-east-1',
                    'us_east_1_regional_endpoint': 'regional',
                    'max_pool_connections': 50,
                    'request_min_compression_size_bytes': 1024,
                    'disable_request_compression': False,
                    'request_checksum_calculation': 'when_supported',
                    'response_checksum_validation': 'when_supported'
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Create S3Client with the config file
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                # Verify boto3.client was called
                assert mock_boto_client.called

                # Get the Config object passed to boto3.client
                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']

                # Verify all optional parameters were set
                assert boto_config.s3 is not None
                assert boto_config.s3['us_east_1_regional_endpoint'] == 'regional'
                assert boto_config.request_min_compression_size_bytes == 1024
                assert boto_config.disable_request_compression is False
                assert boto_config.request_checksum_calculation == 'when_supported'
                assert boto_config.response_checksum_validation == 'when_supported'
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_us_east_1_regional_endpoint_parameter(self, mock_boto_client):
        """Test us_east_1_regional_endpoint parameter - covers line 65"""
        from quantdl.storage.s3_client import S3Client

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-east-1',
                    'us_east_1_regional_endpoint': 'legacy'
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']
                assert boto_config.s3['us_east_1_regional_endpoint'] == 'legacy'
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_request_min_compression_size_bytes_parameter(self, mock_boto_client):
        """Test request_min_compression_size_bytes parameter - covers line 71"""
        from quantdl.storage.s3_client import S3Client

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-west-2',
                    'request_min_compression_size_bytes': 2048
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']
                assert boto_config.request_min_compression_size_bytes == 2048
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_disable_request_compression_parameter(self, mock_boto_client):
        """Test disable_request_compression parameter - covers line 74"""
        from quantdl.storage.s3_client import S3Client

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-west-2',
                    'disable_request_compression': True
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']
                assert boto_config.disable_request_compression is True
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_request_checksum_calculation_parameter(self, mock_boto_client):
        """Test request_checksum_calculation parameter - covers line 77"""
        from quantdl.storage.s3_client import S3Client

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-west-2',
                    'request_checksum_calculation': 'when_required'
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']
                assert boto_config.request_checksum_calculation == 'when_required'
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_response_checksum_validation_parameter(self, mock_boto_client):
        """Test response_checksum_validation parameter - covers line 80"""
        from quantdl.storage.s3_client import S3Client

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                's3': {
                    'region_name': 'us-west-2',
                    'response_checksum_validation': 'when_required'
                }
            }
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client_wrapper = S3Client(config_path=config_path)

                # Access the client property to trigger boto3.client call
                _ = s3_client_wrapper.client

                call_kwargs = mock_boto_client.call_args[1]
                boto_config = call_kwargs['config']
                assert boto_config.response_checksum_validation == 'when_required'
        finally:
            import os
            os.unlink(config_path)

    @patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'test_key', 'AWS_SECRET_ACCESS_KEY': 'test_secret'})
    @patch('quantdl.storage.s3_client.boto3.client')
    def test_client_property_returns_boto_client(self, mock_boto_client):
        """Test client property returns boto3 S3 client"""
        from quantdl.storage.s3_client import S3Client

        mock_client_instance = Mock()
        mock_boto_client.return_value = mock_client_instance

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {'s3': {'region_name': 'us-west-2'}}
            yaml.dump(config, f)
            config_path = f.name

        try:
            with patch('quantdl.storage.s3_client.UploadConfig') as mock_config_class:
                mock_config_instance = Mock()
                mock_config_instance.client = config['s3']
                mock_config_instance.load.return_value = config
                mock_config_class.return_value = mock_config_instance

                s3_client = S3Client(config_path=config_path)

                # Access the client property
                result_client = s3_client.client

                assert result_client == mock_client_instance
        finally:
            import os
            os.unlink(config_path)
