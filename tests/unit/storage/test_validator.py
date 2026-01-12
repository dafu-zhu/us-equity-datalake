"""
Unit tests for storage.validation.Validator class
Tests data validation and existence checking
"""
import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError


class TestValidator:
    """Test Validator class"""

    def test_list_available_years_fundamental_type(self):
        """Test list_available_years with fundamental data_type - covers line 72."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock S3 response with fundamental data
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'data/raw/fundamental/AAPL/2023/fundamental.parquet'},
                {'Key': 'data/raw/fundamental/AAPL/2024/fundamental.parquet'}
            ],
            'IsTruncated': False
        }

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        years = validator.list_available_years("AAPL", "fundamental")

        # For fundamental type, the function continues (line 100), so years should be empty
        assert years == []

    def test_list_available_years_ttm_type(self):
        """Test list_available_years with ttm data_type - covers line 74."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock S3 response
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'data/derived/features/fundamental/AAPL/2023/ttm.parquet'},
                {'Key': 'data/derived/features/fundamental/AAPL/2024/ttm.parquet'}
            ],
            'IsTruncated': False
        }

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        years = validator.list_available_years("AAPL", "ttm")

        # Should build prefix with ttm path
        mock_s3_client.list_objects_v2.assert_called_with(
            Bucket="test-bucket",
            Prefix='data/derived/features/fundamental/AAPL/'
        )

    def test_list_available_years_with_continuation_token(self):
        """Test pagination with continuation token - covers lines 90, 108."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock paginated responses
        mock_s3_client.list_objects_v2.side_effect = [
            {
                'Contents': [
                    {'Key': 'data/raw/ticks/daily/AAPL/2023/01/ticks.parquet'}
                ],
                'IsTruncated': True,
                'NextContinuationToken': 'token123'
            },
            {
                'Contents': [
                    {'Key': 'data/raw/ticks/daily/AAPL/2024/01/ticks.parquet'}
                ],
                'IsTruncated': False
            }
        ]

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        years = validator.list_available_years("AAPL", "ticks")

        # Should make two calls - second with continuation token
        assert mock_s3_client.list_objects_v2.call_count == 2
        second_call_kwargs = mock_s3_client.list_objects_v2.call_args_list[1][1]
        assert second_call_kwargs['ContinuationToken'] == 'token123'
        assert 2024 in years and 2023 in years

    def test_list_available_years_fundamental_continues(self):
        """Test that fundamental data type continues processing - covers line 100."""
        from quantdl.storage.validation import Validator

        mock_s3_client = Mock()
        mock_logger = Mock()

        # Mock response with fundamental data
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'data/raw/fundamental/AAPL/fundamental.parquet'}
            ],
            'IsTruncated': False
        }

        validator = Validator(s3_client=mock_s3_client, bucket_name="test-bucket")
        years = validator.list_available_years("AAPL", "fundamental")

        # Should return empty list as fundamental continues without extracting years
        assert years == []

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
