"""
Unit tests for WRDS-free daily update app (app_no_wrds.py).

Tests SimpleCIKResolver and DailyUpdateAppNoWRDS classes.
"""

import datetime as dt
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pytest


class TestSimpleCIKResolver:
    """Tests for SimpleCIKResolver class"""

    def test_init(self):
        """Test SimpleCIKResolver initialization"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        assert resolver.logger is logger
        assert resolver._sec_cache is None

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_fetch_sec_mapping_success(self, mock_get):
        """Test fetching SEC CIK mapping successfully"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
            "1": {"ticker": "MSFT", "cik_str": 789019},
            "2": {"ticker": "BRK.B", "cik_str": 1067983}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        result = resolver._fetch_sec_mapping()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert 'ticker' in result.columns
        assert 'cik' in result.columns
        assert len(result) == 3

        # Check ticker normalization (BRK.B -> BRKB)
        tickers = result['ticker'].to_list()
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert 'BRKB' in tickers  # Normalized

        # Check CIK zero-padding
        ciks = result['cik'].to_list()
        assert '0000320193' in ciks
        assert '0000789019' in ciks

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_fetch_sec_mapping_caches_result(self, mock_get):
        """Test that SEC mapping is cached after first fetch"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        # First call
        result1 = resolver._fetch_sec_mapping()
        # Second call
        result2 = resolver._fetch_sec_mapping()

        # Should only call API once
        assert mock_get.call_count == 1
        # Should return same cached object
        assert result1 is result2

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_fetch_sec_mapping_handles_error(self, mock_get):
        """Test SEC mapping fetch handles errors gracefully"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_get.side_effect = Exception("Network error")

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        result = resolver._fetch_sec_mapping()

        # Should return empty DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert 'ticker' in result.columns
        assert 'cik' in result.columns
        logger.error.assert_called_once()

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_cik_success(self, mock_get):
        """Test getting CIK for a symbol"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        cik = resolver.get_cik('AAPL')

        assert cik == '0000320193'

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_cik_normalizes_symbol(self, mock_get):
        """Test that get_cik normalizes symbols (BRK-B, BRK.B, BRKB all work)"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "BRK.B", "cik_str": 1067983}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        # All variations should work
        assert resolver.get_cik('BRK-B') == '0001067983'
        assert resolver.get_cik('BRK.B') == '0001067983'
        assert resolver.get_cik('BRKB') == '0001067983'

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_cik_returns_none_for_missing(self, mock_get):
        """Test get_cik returns None for unknown symbol"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        cik = resolver.get_cik('UNKNOWN')

        assert cik is None

    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_batch_prefetch_ciks(self, mock_get):
        """Test batch CIK resolution"""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
            "1": {"ticker": "MSFT", "cik_str": 789019}
        }
        mock_get.return_value = mock_response

        logger = Mock()
        resolver = SimpleCIKResolver(logger=logger)

        symbols = ['AAPL', 'MSFT', 'UNKNOWN']
        result = resolver.batch_prefetch_ciks(symbols)

        assert isinstance(result, dict)
        assert result['AAPL'] == '0000320193'
        assert result['MSFT'] == '0000789019'
        assert result['UNKNOWN'] is None


class TestDailyUpdateAppNoWRDSInitialization:
    """Tests for DailyUpdateAppNoWRDS initialization"""

    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_init(self, mock_logger, mock_ticks, mock_config, mock_s3):
        """Test DailyUpdateAppNoWRDS initialization"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Check components initialized
        assert app.config is not None
        assert app.s3_client is not None
        assert app.logger is not None
        assert app.calendar is not None
        assert app.alpaca_ticks is not None
        assert app.cik_resolver is not None
        assert app.data_collectors is not None
        assert app.data_publishers is not None
        assert app.sec_client is not None

    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_init_uses_simple_cik_resolver(self, mock_logger, mock_ticks, mock_config, mock_s3):
        """Test that DailyUpdateAppNoWRDS uses SimpleCIKResolver"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS, SimpleCIKResolver

        app = DailyUpdateAppNoWRDS()

        # Should use SimpleCIKResolver, not SecurityMaster-based resolver
        assert isinstance(app.cik_resolver, SimpleCIKResolver)


class TestDailyUpdateAppNoWRDSGetSymbols:
    """Tests for _get_symbols method"""

    @patch('quantdl.update.app_no_wrds.fetch_all_stocks')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_fetches_from_nasdaq(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_fetch
    ):
        """Test that _get_symbols fetches from Nasdaq FTP"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        import pandas as pd

        # Mock Nasdaq response
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Name': ['Apple Inc.', 'Microsoft Corp', 'Alphabet Inc']
        })
        mock_fetch.return_value = mock_df

        app = DailyUpdateAppNoWRDS()
        symbols = app._get_symbols()

        # Check Nasdaq was called
        mock_fetch.assert_called_once_with(with_filter=True, refresh=True, logger=app.logger)

        # Check result
        assert symbols == ['AAPL', 'MSFT', 'GOOGL']

    @patch('quantdl.update.app_no_wrds.fetch_all_stocks')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_caches_result(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_fetch
    ):
        """Test that _get_symbols caches the result"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        import pandas as pd

        mock_df = pd.DataFrame({'Ticker': ['AAPL'], 'Name': ['Apple']})
        mock_fetch.return_value = mock_df

        app = DailyUpdateAppNoWRDS()

        # Call twice
        symbols1 = app._get_symbols()
        symbols2 = app._get_symbols()

        # Should only fetch once
        assert mock_fetch.call_count == 1
        # Should return same result
        assert symbols1 == symbols2


class TestDailyUpdateAppNoWRDSCheckMarketOpen:
    """Tests for check_market_open method"""

    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_check_market_open(self, mock_logger, mock_ticks, mock_config, mock_s3):
        """Test check_market_open method"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock calendar
        app.calendar = Mock()
        app.calendar.is_trading_day.return_value = True

        date = dt.date(2024, 6, 3)
        result = app.check_market_open(date)

        assert result is True
        app.calendar.is_trading_day.assert_called_once_with(date)


# Additional tests for other methods would follow the same pattern as test_app.py
# but with WRDS-free mocking (e.g., mocking fetch_all_stocks instead of CRSP)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
