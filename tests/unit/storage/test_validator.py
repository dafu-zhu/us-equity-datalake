"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking
"""
import os
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestValidator:
    """Test Validator class"""

    def test_data_exists_monthly_partition(self):
        """Test data_exists with monthly partition - covers line 145."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, month=6)

        # Should construct monthly partition path
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/AAPL/2024/06/ticks.parquet'
        )
        assert result is True

    def test_data_exists_monthly_partition_with_security_id(self):
        """Test data_exists with security_id uses security_id-based path."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, month=6, security_id=12345)

        # Should use security_id in path, not symbol
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/12345/2024/06/ticks.parquet'
        )
        assert result is True

    def test_data_exists_yearly_with_security_id_uses_history_path(self):
        """Test data_exists with security_id and year (no month) uses history.parquet path."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, security_id=12345)

        # With security_id, yearly check uses history.parquet
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/12345/history.parquet'
        )
        assert result is True

    def test_data_exists_yearly_without_security_id_uses_legacy_path(self):
        """Test data_exists without security_id uses legacy yearly partition path."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024)

        # Without security_id, falls back to legacy yearly path
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/daily/AAPL/2024/ticks.parquet'
        )
        assert result is True

    def test_data_exists_minute_ticks_with_security_id(self):
        """Test data_exists for minute ticks with security_id uses security_id in path."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", day="2024-06-15", security_id=12345)

        # Minute ticks with security_id
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/minute/12345/2024/06/15/ticks.parquet'
        )
        assert result is True

    def test_data_exists_minute_ticks_without_security_id(self):
        """Test data_exists for minute ticks without security_id uses symbol in path."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.head_object.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", day="2024-06-15")

        # Minute ticks without security_id uses symbol
        mock_s3_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key='data/raw/ticks/minute/AAPL/2024/06/15/ticks.parquet'
        )
        assert result is True

    def test_data_exists_not_found(self):
        """Test data_exists returns False when object not found (404)."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.data_exists("AAPL", "ticks", year=2024, month=6, security_id=12345)

        assert result is False

    def test_get_existing_minute_days_returns_day_set(self):
        """Test batch listing of existing minute tick days."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            'CommonPrefixes': [
                {'Prefix': 'data/raw/ticks/minute/12345/2024/06/03/'},
                {'Prefix': 'data/raw/ticks/minute/12345/2024/06/04/'},
                {'Prefix': 'data/raw/ticks/minute/12345/2024/06/15/'},
            ]
        }

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.get_existing_minute_days(12345, 2024, 6)

        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket",
            Prefix='data/raw/ticks/minute/12345/2024/06/',
            Delimiter='/'
        )
        assert result == {'03', '04', '15'}

    def test_get_existing_minute_days_empty(self):
        """Test batch listing returns empty set when no days exist."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {}

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        result = validator.get_existing_minute_days(12345, 2024, 6)

        assert result == set()

    def test_top_3000_exists_error_logging(self):
        """Test top_3000_exists logs error on non-404 errors - covers lines 191-192."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock non-404 error
        error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')

        # Patch setup_logger to return our mock logger
        with patch('quantdl.storage.validation.setup_logger', return_value=mock_logger):
            validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
            result = validator.top_3000_exists(2024, 6)

        assert result is False
        mock_logger.error.assert_called()
        # Check that the error message contains the key
        error_call = str(mock_logger.error.call_args)
        assert "data/symbols/2024/06/top3000.txt" in error_call
