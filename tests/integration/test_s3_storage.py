"""
Integration tests for S3 storage modules
Tests s3_client and validation modules with mocked S3 service
"""
import pytest
import tempfile
import yaml
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError
from quantdl.storage.clients import S3Client
from quantdl.storage.pipeline import Validator


@pytest.mark.integration
class TestS3ClientIntegration:
    """Integration tests for S3Client"""

    @pytest.fixture
    def mock_config_file(self):
        """Create a temporary config file for S3Client"""
        config_data = {
            'client': {
                'region_name': 'us-east-2',
                'max_pool_connections': 50,
                'connect_timeout': 60,
                'read_timeout': 60,
                'retries': {
                    'mode': 'adaptive',
                    'total_max_attempts': 3
                },
                's3': {
                    'addressing_style': 'virtual',
                    'payload_signing_enabled': True
                },
                'tcp_keepalive': True
            },
            'transfer': {
                'multipart_threshold': 8388608,
                'max_concurrency': 10
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock AWS environment variables"""
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test_access_key')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'test_secret_key')
        monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
        monkeypatch.setenv('STORAGE_BACKEND', 's3')

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_client_initialization(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test S3Client initialization with config file"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        client = S3Client(config_path=mock_config_file)

        # Verify config loaded and client accessible
        assert client.config is not None
        assert client.is_local is False
        _ = client.client  # Trigger client creation
        assert mock_boto_client.called

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_boto_config_creation(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test boto3 Config object creation"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        client = S3Client(config_path=mock_config_file)
        _ = client.client  # Trigger client creation

        # Verify boto3.client was called with config
        call_kwargs = mock_boto_client.call_args[1]
        config = call_kwargs['config']
        assert config.region_name == 'us-east-2'
        assert config.max_pool_connections == 50

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_boto_config_with_retries(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test boto3 Config with retry settings"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        client = S3Client(config_path=mock_config_file)
        _ = client.client  # Trigger client creation

        # Verify retries config passed to boto3
        call_kwargs = mock_boto_client.call_args[1]
        config = call_kwargs['config']
        assert config.retries is not None
        assert config.retries['mode'] == 'adaptive'
        assert config.retries['total_max_attempts'] == 3

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_boto_config_with_s3_settings(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test boto3 Config with S3-specific settings"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        client = S3Client(config_path=mock_config_file)
        _ = client.client  # Trigger client creation

        # Verify S3 config passed to boto3
        call_kwargs = mock_boto_client.call_args[1]
        config = call_kwargs['config']
        assert config.s3 is not None
        assert config.s3['addressing_style'] == 'virtual'
        assert config.s3['payload_signing_enabled'] is True

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_client_property_creates_boto_client(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test that client property creates boto3 S3 client"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        client = S3Client(config_path=mock_config_file)
        s3_client = client.client

        # Verify boto3.client was called with correct params
        mock_boto_client.assert_called_once()
        call_args = mock_boto_client.call_args
        assert call_args[0][0] == 's3'
        assert call_args[1]['aws_access_key_id'] == 'test_access_key'
        assert call_args[1]['aws_secret_access_key'] == 'test_secret_key'

    def test_missing_client_config_raises_error(self, mock_env_vars):
        """Test that missing client config raises ValueError"""
        # Create config with empty client section
        config_data = {'client': {}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Creating the client should raise because boto config is built in __init__
            with pytest.raises(ValueError, match="Client configuration is empty"):
                S3Client(config_path=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


@pytest.mark.integration
@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestValidatorIntegration:
    """Integration tests for Validator with mocked S3"""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client"""
        return MagicMock()

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock environment variables"""
        monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
        monkeypatch.setenv('STORAGE_BACKEND', 's3')

    def test_validator_initialization(self, mock_s3_client, mock_env_vars):
        """Test Validator initialization"""
        validator = Validator(s3_client=mock_s3_client)

        assert validator.s3_client == mock_s3_client
        assert validator.bucket_name == 'test-bucket'

    def test_validator_with_custom_bucket(self, mock_s3_client):
        """Test Validator with custom bucket name"""
        validator = Validator(s3_client=mock_s3_client, bucket_name='custom-bucket')

        assert validator.bucket_name == 'custom-bucket'

    def test_list_files_under_prefix(self, mock_s3_client, mock_env_vars):
        """Test listing files under S3 prefix"""
        # Mock S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'data/raw/fundamental/AAPL/fundamental.parquet'},
                {'Key': 'data/raw/fundamental/MSFT/fundamental.parquet'},
            ],
            'IsTruncated': False
        }

        validator = Validator(s3_client=mock_s3_client)
        files = validator.list_files_under_prefix('data/raw/fundamental')

        assert len(files) == 2
        assert 'data/raw/fundamental/AAPL/fundamental.parquet' in files
        assert 'data/raw/fundamental/MSFT/fundamental.parquet' in files

        # Verify S3 call
        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket='test-bucket',
            Prefix='data/raw/fundamental'
        )

    def test_list_files_with_pagination(self, mock_s3_client):
        """Test listing files with S3 pagination"""
        # Mock paginated responses
        mock_s3_client.list_objects_v2.side_effect = [
            {
                'Contents': [
                    {'Key': 'file1.parquet'},
                    {'Key': 'file2.parquet'},
                ],
                'IsTruncated': True,
                'NextContinuationToken': 'token123'
            },
            {
                'Contents': [
                    {'Key': 'file3.parquet'},
                ],
                'IsTruncated': False
            }
        ]

        validator = Validator(s3_client=mock_s3_client)
        files = validator.list_files_under_prefix('data/')

        assert len(files) == 3
        assert mock_s3_client.list_objects_v2.call_count == 2

    def test_list_files_empty_prefix(self, mock_s3_client):
        """Test listing files when prefix has no objects"""
        mock_s3_client.list_objects_v2.return_value = {
            'IsTruncated': False
        }

        validator = Validator(s3_client=mock_s3_client)
        files = validator.list_files_under_prefix('data/empty/')

        assert len(files) == 0

    def test_data_exists_daily_ticks(self, mock_s3_client, mock_env_vars):
        """Test checking existence of daily ticks data"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'ticks', year=2024)

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/raw/ticks/daily/AAPL/2024/ticks.parquet'
        )

    def test_data_exists_minute_ticks(self, mock_s3_client, mock_env_vars):
        """Test checking existence of minute ticks data"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'ticks', day='2024-06-15')

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/raw/ticks/minute/AAPL/2024/06/15/ticks.parquet'
        )

    def test_data_exists_fundamental(self, mock_s3_client, mock_env_vars):
        """Test checking existence of fundamental data"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'fundamental')

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/raw/fundamental/AAPL/fundamental.parquet'
        )

    def test_data_exists_derived_fundamental(self, mock_s3_client, mock_env_vars):
        """Test checking existence of derived fundamental data"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'fundamental', data_tier='derived')

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/derived/features/fundamental/AAPL/metrics.parquet'
        )

    def test_data_exists_ttm(self, mock_s3_client, mock_env_vars):
        """Test checking existence of TTM data"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'ttm')

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/derived/features/fundamental/AAPL/ttm.parquet'
        )

    def test_data_not_exists(self, mock_s3_client, mock_env_vars):
        """Test checking non-existent data"""
        # Mock 404 error
        error_response = {'Error': {'Code': '404'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'ticks', year=2024)

        assert exists is False

    def test_data_exists_with_s3_error(self, mock_s3_client, mock_env_vars):
        """Test handling of S3 errors other than 404"""
        # Mock non-404 error
        error_response = {'Error': {'Code': '500'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.data_exists('AAPL', 'ticks', year=2024)

        # Should return False and log error
        assert exists is False

    def test_data_exists_invalid_parameters(self, mock_s3_client, mock_env_vars):
        """Test data_exists with invalid parameters"""
        validator = Validator(s3_client=mock_s3_client)

        # Both year and day specified
        with pytest.raises(ValueError, match="Specify year OR day"):
            validator.data_exists('AAPL', 'ticks', year=2024, day='2024-06-15')

        # Invalid data_tier
        with pytest.raises(ValueError, match="Expected data_tier is raw or derived"):
            validator.data_exists('AAPL', 'ticks', year=2024, data_tier='invalid')

        # Invalid data_type
        with pytest.raises(ValueError, match="Expected data_type"):
            validator.data_exists('AAPL', 'invalid_type', year=2024)

        # Ticks without year or day
        with pytest.raises(ValueError, match="Must provide either year or day"):
            validator.data_exists('AAPL', 'ticks')

    def test_top_3000_exists(self, mock_s3_client, mock_env_vars):
        """Test checking existence of top 3000 symbols file"""
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.top_3000_exists(2024, 6)

        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/symbols/2024/06/top3000.txt'
        )

    def test_top_3000_not_exists(self, mock_s3_client, mock_env_vars):
        """Test checking non-existent top 3000 file"""
        error_response = {'Error': {'Code': '404'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        validator = Validator(s3_client=mock_s3_client)
        exists = validator.top_3000_exists(2024, 6)

        assert exists is False


@pytest.mark.integration
@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestS3ClientValidatorWorkflow:
    """Integration tests for S3Client + Validator workflow"""

    @pytest.fixture
    def mock_config_file(self):
        """Create a minimal config file"""
        config_data = {
            'client': {
                'region_name': 'us-east-2',
                'max_pool_connections': 10
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Mock all environment variables"""
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test_key')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'test_secret')
        monkeypatch.setenv('S3_BUCKET_NAME', 'test-bucket')
        monkeypatch.setenv('STORAGE_BACKEND', 's3')

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_end_to_end_workflow(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test end-to-end workflow: S3Client → Validator → S3 operations"""
        # Setup mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock head_object to return success
        mock_s3.head_object.return_value = {}

        # Create S3Client and get boto client
        s3_client = S3Client(config_path=mock_config_file)
        boto_client = s3_client.client

        # Create Validator with the boto client
        validator = Validator(s3_client=boto_client)

        # Test data existence check
        exists = validator.data_exists('AAPL', 'ticks', year=2024)

        assert exists is True
        mock_s3.head_object.assert_called_once()

    @patch('quantdl.storage.clients.s3.boto3.client')
    def test_workflow_with_multiple_operations(self, mock_boto_client, mock_config_file, mock_env_vars):
        """Test workflow with multiple validator operations"""
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock different responses for different operations
        mock_s3.head_object.return_value = {}
        mock_s3.list_objects_v2.return_value = {
            'Contents': [{'Key': 'data/raw/fundamental/AAPL/fundamental.parquet'}],
            'IsTruncated': False
        }

        # Create clients
        s3_client = S3Client(config_path=mock_config_file)
        validator = Validator(s3_client=s3_client.client)

        # Perform multiple operations
        exists_check = validator.data_exists('AAPL', 'fundamental')
        files = validator.list_files_under_prefix('data/raw/fundamental')

        assert exists_check is True
        assert len(files) == 1
        assert mock_s3.head_object.called
        assert mock_s3.list_objects_v2.called
