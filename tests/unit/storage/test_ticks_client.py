"""Unit tests for TicksClient."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import polars as pl
import datetime as dt
from io import BytesIO


class TestTicksClient:
    """Tests for TicksClient class."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        return Mock()

    @pytest.fixture
    def mock_security_master(self):
        """Create mock SecurityMaster."""
        mock = Mock()
        mock.get_security_id.return_value = 12345
        return mock

    @pytest.fixture
    def ticks_client(self, mock_s3_client, mock_security_master):
        """Create TicksClient with mocked dependencies."""
        from quantdl.storage.clients.ticks import TicksClient
        return TicksClient(
            s3_client=mock_s3_client,
            bucket_name='test-bucket',
            security_master=mock_security_master
        )

    def test_init_sets_attributes(self, mock_s3_client, mock_security_master):
        """Test that __init__ sets all attributes correctly."""
        from quantdl.storage.clients.ticks import TicksClient

        client = TicksClient(
            s3_client=mock_s3_client,
            bucket_name='my-bucket',
            security_master=mock_security_master
        )

        assert client.s3_client is mock_s3_client
        assert client.bucket_name == 'my-bucket'
        assert client.security_master is mock_security_master
        assert client._cache == {}

    def test_resolve_symbol_caches_result(self, ticks_client, mock_security_master):
        """Test that _resolve_symbol caches the result."""
        # First call
        result1 = ticks_client._resolve_symbol('AAPL', 2024)
        assert result1 == 12345
        mock_security_master.get_security_id.assert_called_once_with('AAPL', '2024-12-31')

        # Second call should use cache
        mock_security_master.reset_mock()
        result2 = ticks_client._resolve_symbol('AAPL', 2024)
        assert result2 == 12345
        mock_security_master.get_security_id.assert_not_called()

    def test_resolve_symbol_different_years(self, ticks_client, mock_security_master):
        """Test that different years get different cache entries."""
        mock_security_master.get_security_id.side_effect = [100, 200]

        result_2023 = ticks_client._resolve_symbol('AAPL', 2023)
        result_2024 = ticks_client._resolve_symbol('AAPL', 2024)

        assert result_2023 == 100
        assert result_2024 == 200
        assert mock_security_master.get_security_id.call_count == 2

    def test_clear_cache(self, ticks_client, mock_security_master):
        """Test that clear_cache() empties the cache."""
        # Populate cache
        ticks_client._resolve_symbol('AAPL', 2024)
        assert len(ticks_client._cache) == 1

        # Clear cache
        ticks_client.clear_cache()
        assert len(ticks_client._cache) == 0

        # Next call should hit security_master again
        mock_security_master.reset_mock()
        ticks_client._resolve_symbol('AAPL', 2024)
        mock_security_master.get_security_id.assert_called_once()

    def test_determine_months_no_filters(self, ticks_client):
        """Test _determine_months with no date filters."""
        start, end = ticks_client._determine_months(None, None)
        assert start == 1
        assert end == 12

    def test_determine_months_with_start_date(self, ticks_client):
        """Test _determine_months with start_date filter."""
        start, end = ticks_client._determine_months('2024-03-15', None)
        assert start == 3
        assert end == 12

    def test_determine_months_with_end_date(self, ticks_client):
        """Test _determine_months with end_date filter."""
        start, end = ticks_client._determine_months(None, '2024-09-30')
        assert start == 1
        assert end == 9

    def test_determine_months_with_both_dates(self, ticks_client):
        """Test _determine_months with both date filters."""
        start, end = ticks_client._determine_months('2024-03-01', '2024-08-31')
        assert start == 3
        assert end == 8

    def test_apply_date_filter_start_only(self, ticks_client):
        """Test _apply_date_filter with start_date only."""
        df = pl.DataFrame({
            'timestamp': ['2024-01-15', '2024-03-15', '2024-06-15'],
            'value': [1, 2, 3]
        })

        result = ticks_client._apply_date_filter(df, '2024-02-01', None)

        assert len(result) == 2
        assert result['timestamp'].to_list() == ['2024-03-15', '2024-06-15']

    def test_apply_date_filter_end_only(self, ticks_client):
        """Test _apply_date_filter with end_date only."""
        df = pl.DataFrame({
            'timestamp': ['2024-01-15', '2024-03-15', '2024-06-15'],
            'value': [1, 2, 3]
        })

        result = ticks_client._apply_date_filter(df, None, '2024-04-01')

        assert len(result) == 2
        assert result['timestamp'].to_list() == ['2024-01-15', '2024-03-15']

    def test_apply_date_filter_both(self, ticks_client):
        """Test _apply_date_filter with both dates."""
        df = pl.DataFrame({
            'timestamp': ['2024-01-15', '2024-03-15', '2024-06-15'],
            'value': [1, 2, 3]
        })

        result = ticks_client._apply_date_filter(df, '2024-02-01', '2024-05-01')

        assert len(result) == 1
        assert result['timestamp'].to_list() == ['2024-03-15']

    def test_apply_date_filter_no_filter(self, ticks_client):
        """Test _apply_date_filter with no filters returns original."""
        df = pl.DataFrame({
            'timestamp': ['2024-01-15', '2024-03-15', '2024-06-15'],
            'value': [1, 2, 3]
        })

        result = ticks_client._apply_date_filter(df, None, None)

        assert len(result) == 3

    @patch('quantdl.storage.clients.ticks.dt')
    def test_fetch_by_security_id_routes_to_history(self, mock_dt, ticks_client):
        """Test that _fetch_by_security_id routes completed years to history."""
        mock_dt.date.today.return_value = dt.date(2024, 6, 15)

        with patch.object(ticks_client, '_fetch_history_year') as mock_fetch:
            mock_fetch.return_value = pl.DataFrame()
            ticks_client._fetch_by_security_id(123, 2023, None, None)
            mock_fetch.assert_called_once_with(123, 2023, None, None)

    @patch('quantdl.storage.clients.ticks.dt')
    def test_fetch_by_security_id_routes_to_monthly(self, mock_dt, ticks_client):
        """Test that _fetch_by_security_id routes current year to monthly."""
        mock_dt.date.today.return_value = dt.date(2024, 6, 15)

        with patch.object(ticks_client, '_fetch_monthly') as mock_fetch:
            mock_fetch.return_value = pl.DataFrame()
            ticks_client._fetch_by_security_id(123, 2024, None, None)
            mock_fetch.assert_called_once_with(123, 2024, None, None)

    def test_get_daily_ticks_resolves_and_fetches(self, ticks_client):
        """Test that get_daily_ticks resolves symbol and fetches data."""
        with patch.object(ticks_client, '_resolve_symbol') as mock_resolve, \
             patch.object(ticks_client, '_fetch_by_security_id') as mock_fetch:
            mock_resolve.return_value = 123
            mock_fetch.return_value = pl.DataFrame({'timestamp': ['2024-01-01']})

            result = ticks_client.get_daily_ticks('AAPL', 2024)

            mock_resolve.assert_called_once_with('AAPL', 2024)
            mock_fetch.assert_called_once_with(123, 2024, None, None)

    def test_get_daily_ticks_history_uses_end_date_year(self, ticks_client, mock_s3_client):
        """Test that get_daily_ticks_history uses end_date year for resolution."""
        # Create parquet data in memory
        df = pl.DataFrame({
            'timestamp': ['2023-01-15', '2023-06-15'],
            'close': [100.0, 105.0]
        })
        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        mock_s3_client.get_object.return_value = {'Body': buffer}

        with patch.object(ticks_client, '_resolve_symbol') as mock_resolve:
            mock_resolve.return_value = 123

            result = ticks_client.get_daily_ticks_history('AAPL', end_date='2023-12-31')

            mock_resolve.assert_called_once_with('AAPL', 2023)


class TestTicksClientErrors:
    """Tests for TicksClient error handling."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        return Mock()

    @pytest.fixture
    def mock_security_master(self):
        """Create mock SecurityMaster."""
        mock = Mock()
        mock.get_security_id.return_value = 12345
        return mock

    @pytest.fixture
    def ticks_client(self, mock_s3_client, mock_security_master):
        """Create TicksClient with mocked dependencies."""
        from quantdl.storage.clients.ticks import TicksClient
        return TicksClient(
            s3_client=mock_s3_client,
            bucket_name='test-bucket',
            security_master=mock_security_master
        )

    def test_fetch_history_year_raises_on_no_such_key(self, ticks_client, mock_s3_client):
        """Test that _fetch_history_year raises ValueError on missing data."""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        with pytest.raises(ValueError, match="No historical data found"):
            ticks_client._fetch_history_year(123, 2023, None, None)

    def test_fetch_monthly_raises_when_no_months_found(self, ticks_client, mock_s3_client):
        """Test that _fetch_monthly raises ValueError when no month files exist."""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        with pytest.raises(ValueError, match="No data found"):
            ticks_client._fetch_monthly(123, 2024, None, None)

    def test_get_daily_ticks_history_raises_on_missing(self, ticks_client, mock_s3_client):
        """Test that get_daily_ticks_history raises ValueError on missing data."""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not found'}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        with pytest.raises(ValueError, match="No historical data found"):
            ticks_client.get_daily_ticks_history('AAPL')
