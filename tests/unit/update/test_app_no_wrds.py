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

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_init(self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar):
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

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_init_uses_simple_cik_resolver(self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar):
        """Test that DailyUpdateAppNoWRDS uses SimpleCIKResolver"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS, SimpleCIKResolver

        app = DailyUpdateAppNoWRDS()

        # Should use SimpleCIKResolver, not SecurityMaster-based resolver
        assert isinstance(app.cik_resolver, SimpleCIKResolver)


class TestDailyUpdateAppNoWRDSGetSymbols:
    """Tests for _get_symbols method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.fetch_all_stocks')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_fetches_from_nasdaq(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_fetch, mock_security_master, mock_calendar
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

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.fetch_all_stocks')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_caches_result(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_fetch, mock_security_master, mock_calendar
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

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_check_market_open(self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar):
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


class TestDailyUpdateAppNoWRDSGetRecentEdgarFilings:
    """Tests for get_recent_edgar_filings method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_recent_edgar_filings_success(
        self, mock_get, mock_s3, mock_config, mock_ticks, mock_logger, mock_security_master, mock_calendar
    ):
        """Test get_recent_edgar_filings returns recent filings"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        # Use fixed date for deterministic testing
        fixed_today = dt.date(2025, 1, 13)

        # Calculate dates based on fixed today
        recent_date1 = (fixed_today - dt.timedelta(days=2)).isoformat()
        recent_date2 = (fixed_today - dt.timedelta(days=5)).isoformat()
        old_date = (fixed_today - dt.timedelta(days=20)).isoformat()

        # Mock EDGAR API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'filings': {
                'recent': {
                    'filingDate': [recent_date1, recent_date2, old_date],
                    'form': ['10-K', '10-Q', '8-K'],
                    'accessionNumber': ['0001234567-25-000001', '0001234567-25-000002', '0001234567-24-000010']
                }
            }
        }
        mock_get.return_value = mock_response

        app = DailyUpdateAppNoWRDS()

        # Mock dt.date.today() to return fixed date
        with patch('quantdl.update.app_no_wrds.dt.date') as mock_date:
            mock_date.today.return_value = fixed_today
            mock_date.side_effect = lambda *args, **kw: dt.date(*args, **kw)

            result = app.get_recent_edgar_filings('0000320193', lookback_days=7)

            # Should return only filings within lookback (recent_date1 and recent_date2)
            assert len(result) == 2
            assert result[0]['form'] == '10-K'
            assert result[0]['filingDate'] == recent_date1
            assert result[1]['form'] == '10-Q'
            assert result[1]['filingDate'] == recent_date2

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_recent_edgar_filings_no_recent(
        self, mock_get, mock_s3, mock_config, mock_ticks, mock_logger, mock_security_master, mock_calendar
    ):
        """Test get_recent_edgar_filings with no recent filings"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        # Use fixed date for deterministic testing
        fixed_today = dt.date(2025, 1, 13)

        # Use date from over a year ago
        old_date = (fixed_today - dt.timedelta(days=365)).isoformat()

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'filings': {
                'recent': {
                    'filingDate': [old_date],
                    'form': ['10-K'],
                    'accessionNumber': ['0001234567-24-000001']
                }
            }
        }
        mock_get.return_value = mock_response

        app = DailyUpdateAppNoWRDS()

        # Mock dt.date.today() to return fixed date
        with patch('quantdl.update.app_no_wrds.dt.date') as mock_date:
            mock_date.today.return_value = fixed_today
            mock_date.side_effect = lambda *args, **kw: dt.date(*args, **kw)

            result = app.get_recent_edgar_filings('0000320193', lookback_days=7)

            assert len(result) == 0

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.requests.get')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_recent_edgar_filings_handles_error(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_get, mock_security_master, mock_calendar
    ):
        """Test get_recent_edgar_filings handles API errors gracefully"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        mock_get.side_effect = Exception("Network error")

        app = DailyUpdateAppNoWRDS()
        result = app.get_recent_edgar_filings('0000320193')

        assert result == []

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.requests.get')
    def test_get_recent_edgar_filings_filters_relevant_forms(
        self, mock_get, mock_s3, mock_config, mock_ticks, mock_logger, mock_security_master, mock_calendar
    ):
        """Test that only relevant forms (10-K, 10-Q, 8-K) are returned"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        # Use fixed date for deterministic testing
        fixed_today = dt.date(2025, 1, 13)

        # Calculate recent dates
        date1 = (fixed_today - dt.timedelta(days=2)).isoformat()
        date2 = (fixed_today - dt.timedelta(days=3)).isoformat()
        date3 = (fixed_today - dt.timedelta(days=4)).isoformat()

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'filings': {
                'recent': {
                    'filingDate': [date1, date2, date3],
                    'form': ['10-K', 'S-1', '8-K'],  # S-1 should be filtered out
                    'accessionNumber': ['0001-25-001', '0001-25-002', '0001-25-003']
                }
            }
        }
        mock_get.return_value = mock_response

        app = DailyUpdateAppNoWRDS()

        # Mock dt.date.today() to return fixed date
        with patch('quantdl.update.app_no_wrds.dt.date') as mock_date:
            mock_date.today.return_value = fixed_today
            mock_date.side_effect = lambda *args, **kw: dt.date(*args, **kw)

            result = app.get_recent_edgar_filings('0000320193', lookback_days=7)

            # Only 10-K and 8-K should be included
            assert len(result) == 2
            forms = [f['form'] for f in result]
            assert '10-K' in forms
            assert '8-K' in forms
            assert 'S-1' not in forms


class TestDailyUpdateAppNoWRDSCheckFiling:
    """Tests for _check_filing method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_check_filing_with_recent_filings(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test _check_filing when symbol has recent filings"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        from threading import Semaphore

        app = DailyUpdateAppNoWRDS()

        # Mock get_recent_edgar_filings
        app.get_recent_edgar_filings = Mock(return_value=[
            {'form': '10-K', 'filingDate': '2025-01-10', 'accessionNumber': '0001-25-001'}
        ])

        semaphore = Semaphore(10)
        result = app._check_filing('AAPL', '0000320193', lookback_days=7, semaphore=semaphore)

        assert result['symbol'] == 'AAPL'
        assert result['cik'] == '0000320193'
        assert result['has_recent_filing'] is True

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_check_filing_no_recent_filings(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test _check_filing when symbol has no recent filings"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        from threading import Semaphore

        app = DailyUpdateAppNoWRDS()
        app.get_recent_edgar_filings = Mock(return_value=[])

        semaphore = Semaphore(10)
        result = app._check_filing('AAPL', '0000320193', lookback_days=7, semaphore=semaphore)

        assert result['symbol'] == 'AAPL'
        assert result['has_recent_filing'] is False


class TestDailyUpdateAppNoWRDSGetSymbolsWithRecentFilings:
    """Tests for get_symbols_with_recent_filings method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_with_recent_filings(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test get_symbols_with_recent_filings identifies symbols correctly"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock CIK resolver
        app.cik_resolver.batch_prefetch_ciks = Mock(return_value={
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044'
        })

        # Mock _check_filing to simulate different results
        def mock_check_filing(symbol, cik, lookback_days, semaphore):
            if symbol == 'AAPL':
                return {'symbol': symbol, 'cik': cik, 'has_recent_filing': True, 'filing_types': ['10-K']}
            return {'symbol': symbol, 'cik': cik, 'has_recent_filing': False, 'filing_types': []}

        app._check_filing = mock_check_filing

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        update_date = dt.date(2025, 1, 12)

        result, filing_stats = app.get_symbols_with_recent_filings(symbols, update_date, lookback_days=7)

        # Only AAPL should be returned
        assert result == {'AAPL'}
        assert filing_stats == {'10-K': 1}

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_symbols_with_recent_filings_filters_none_ciks(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test that symbols without CIKs are filtered out"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock CIK resolver with some None values
        app.cik_resolver.batch_prefetch_ciks = Mock(return_value={
            'AAPL': '0000320193',
            'UNKNOWN': None,
            'MSFT': '0000789019'
        })

        app._check_filing = Mock(return_value={
            'symbol': 'AAPL',
            'cik': '0000320193',
            'has_recent_filing': False,
            'filing_types': []
        })

        symbols = ['AAPL', 'UNKNOWN', 'MSFT']
        update_date = dt.date(2025, 1, 12)

        result, filing_stats = app.get_symbols_with_recent_filings(symbols, update_date, lookback_days=7)

        # _check_filing should only be called for symbols with CIKs
        assert app._check_filing.call_count == 2  # AAPL and MSFT only
        assert filing_stats == {}


class TestDailyUpdateAppNoWRDSUpdateDailyTicks:
    """Tests for update_daily_ticks and process_symbol methods"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_daily_ticks_success(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_daily_ticks successfully updates ticks"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        from dataclasses import dataclass
        import io

        @dataclass
        class MockDataPoint:
            timestamp: str
            open: float
            high: float
            low: float
            close: float
            volume: int

        app = DailyUpdateAppNoWRDS()

        # Mock Alpaca fetch
        mock_tick_data = [{
            'timestamp': '2025-01-10T09:30:00',
            'open': 150.0,
            'high': 151.0,
            'low': 149.5,
            'close': 150.5,
            'volume': 1000000
        }]

        app.alpaca_ticks.fetch_daily_day_bulk = Mock(return_value={
            'AAPL': mock_tick_data
        })
        app.alpaca_ticks.parse_ticks = Mock(return_value=[
            MockDataPoint(timestamp='2025-01-10T09:30:00', open=150.0, high=151.0,
                         low=149.5, close=150.5, volume=1000000)
        ])

        # Mock S3 client (no existing file) - create a proper exception type
        no_such_key_exception = type('NoSuchKey', (Exception,), {})
        app.s3_client.exceptions = type('Exceptions', (), {'NoSuchKey': no_such_key_exception})()
        app.s3_client.get_object = Mock(side_effect=no_such_key_exception())
        app.data_publishers.upload_fileobj = Mock()
        app.data_publishers.bucket_name = 'test-bucket'

        update_date = dt.date(2025, 1, 10)
        stats = app.update_daily_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['success'] == 1
        assert stats['failed'] == 0
        assert stats['skipped'] == 0
        app.data_publishers.upload_fileobj.assert_called_once()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_daily_ticks_skips_no_data(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_daily_ticks skips symbols with no data"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock Alpaca returning no data
        app.alpaca_ticks.fetch_daily_day_bulk = Mock(return_value={'AAPL': []})

        update_date = dt.date(2025, 1, 10)
        stats = app.update_daily_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['skipped'] == 1
        assert stats['success'] == 0

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_daily_ticks_appends_to_existing(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_daily_ticks appends to existing monthly file"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        from dataclasses import dataclass
        import io

        @dataclass
        class MockDataPoint:
            timestamp: str
            open: float
            high: float
            low: float
            close: float
            volume: int

        app = DailyUpdateAppNoWRDS()

        # Mock new tick data
        mock_tick_data = [{
            'timestamp': '2025-01-10T09:30:00',
            'open': 150.0,
            'high': 151.0,
            'low': 149.5,
            'close': 150.5,
            'volume': 1000000
        }]

        app.alpaca_ticks.fetch_daily_day_bulk = Mock(return_value={'AAPL': mock_tick_data})
        app.alpaca_ticks.parse_ticks = Mock(return_value=[
            MockDataPoint(timestamp='2025-01-10T09:30:00', open=150.0, high=151.0,
                         low=149.5, close=150.5, volume=1000000)
        ])

        # Mock existing S3 file
        existing_df = pl.DataFrame({
            'timestamp': ['2025-01-09'],
            'open': [148.0],
            'high': [149.0],
            'low': [147.5],
            'close': [148.5],
            'volume': [900000]
        })

        buffer = io.BytesIO()
        existing_df.write_parquet(buffer)
        buffer.seek(0)

        mock_response = {'Body': buffer}
        app.s3_client.get_object = Mock(return_value=mock_response)
        app.data_publishers.upload_fileobj = Mock()
        app.data_publishers.bucket_name = 'test-bucket'

        update_date = dt.date(2025, 1, 10)
        stats = app.update_daily_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['success'] == 1
        # Verify upload was called
        app.data_publishers.upload_fileobj.assert_called_once()


class TestDailyUpdateAppNoWRDSUpdateMinuteTicks:
    """Tests for update_minute_ticks method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_minute_ticks_success(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_minute_ticks successfully updates minute data"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()
        app.security_master = Mock()
        app.security_master.get_security_id = Mock(return_value=12345)

        # Mock minute data fetch and parse
        mock_minute_df = pl.DataFrame({
            'timestamp': ['2025-01-10 09:30:00'],
            'open': [150.0],
            'high': [150.5],
            'low': [149.8],
            'close': [150.2],
            'volume': [1000]
        })

        app.data_collectors.fetch_minute_day = Mock(return_value={'AAPL': 'mock_bars'})
        app.data_collectors.parse_minute_bars_to_daily = Mock(return_value={
            ('AAPL', '2025-01-10'): mock_minute_df
        })
        app.data_publishers.upload_fileobj = Mock()

        update_date = dt.date(2025, 1, 10)
        stats = app.update_minute_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['success'] == 1
        assert stats['failed'] == 0
        assert stats['skipped'] == 0
        app.data_publishers.upload_fileobj.assert_called_once()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_minute_ticks_skips_empty_data(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_minute_ticks skips empty DataFrames"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock empty DataFrame
        empty_df = pl.DataFrame({
            'timestamp': [], 'open': [], 'high': [],
            'low': [], 'close': [], 'volume': []
        })

        app.data_collectors.fetch_minute_day = Mock(return_value={})
        app.data_collectors.parse_minute_bars_to_daily = Mock(return_value={
            ('AAPL', '2025-01-10'): empty_df
        })

        update_date = dt.date(2025, 1, 10)
        stats = app.update_minute_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['skipped'] == 1
        assert stats['success'] == 0

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_minute_ticks_handles_error(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_minute_ticks handles upload errors"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        mock_minute_df = pl.DataFrame({
            'timestamp': ['2025-01-10 09:30:00'],
            'open': [150.0], 'high': [150.5],
            'low': [149.8], 'close': [150.2], 'volume': [1000]
        })

        app.data_collectors.fetch_minute_day = Mock(return_value={})
        app.data_collectors.parse_minute_bars_to_daily = Mock(return_value={
            ('AAPL', '2025-01-10'): mock_minute_df
        })
        app.data_publishers.upload_fileobj = Mock(side_effect=Exception("Upload failed"))

        update_date = dt.date(2025, 1, 10)
        stats = app.update_minute_ticks(update_date, symbols=['AAPL'], max_workers=1)

        assert stats['failed'] == 1
        assert stats['success'] == 0


class TestDailyUpdateAppNoWRDSUpdateFundamental:
    """Tests for update_fundamental method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_fundamental_success(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_fundamental successfully updates fundamental data"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock CIK resolver
        app.cik_resolver.get_cik = Mock(return_value='0000320193')

        # Mock publishers
        app.data_publishers.publish_fundamental = Mock(return_value={'status': 'success'})
        app.data_publishers.publish_ttm_fundamental = Mock(return_value={'status': 'success'})
        app.data_publishers.publish_derived_fundamental = Mock()

        # Mock derived data
        mock_derived_df = pl.DataFrame({'metric': ['roa'], 'value': [0.15]})
        app.data_collectors.collect_derived_long = Mock(return_value=(mock_derived_df, None))

        symbols = ['AAPL']
        start_date = '2025-01-05'
        end_date = '2025-01-12'

        stats = app.update_fundamental(symbols, start_date, end_date, max_workers=1)

        assert stats['success'] == 1
        assert stats['failed'] == 0
        assert stats['skipped'] == 0

        app.data_publishers.publish_fundamental.assert_called_once()
        app.data_publishers.publish_ttm_fundamental.assert_called_once()
        app.data_publishers.publish_derived_fundamental.assert_called_once()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_fundamental_skips_no_cik(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_fundamental skips symbols without CIK"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock CIK resolver returning None
        app.cik_resolver.get_cik = Mock(return_value=None)

        symbols = ['UNKNOWN']
        start_date = '2025-01-05'
        end_date = '2025-01-12'

        stats = app.update_fundamental(symbols, start_date, end_date, max_workers=1)

        # No symbols processed
        assert stats['success'] == 0
        assert stats['failed'] == 0
        assert stats['skipped'] == 0

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_fundamental_handles_skipped_status(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_fundamental handles skipped status from publisher"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        app.cik_resolver.get_cik = Mock(return_value='0000320193')
        app.data_publishers.publish_fundamental = Mock(return_value={'status': 'skipped'})

        symbols = ['AAPL']
        start_date = '2025-01-05'
        end_date = '2025-01-12'

        stats = app.update_fundamental(symbols, start_date, end_date, max_workers=1)

        assert stats['skipped'] == 1
        assert stats['success'] == 0


class TestDailyUpdateAppNoWRDSRunDailyUpdate:
    """Tests for run_daily_update method"""

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_run_daily_update_market_open(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test run_daily_update on a trading day"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        # Mock all dependencies
        app.security_master.update_from_sec = Mock(return_value={'extended': 0, 'added': 0, 'unchanged': 100})
        app._get_symbols = Mock(return_value=['AAPL', 'MSFT'])
        app.check_market_open = Mock(return_value=True)
        app.update_daily_ticks = Mock(return_value={'success': 2, 'failed': 0, 'skipped': 0})
        app.update_minute_ticks = Mock(return_value={'success': 2, 'failed': 0, 'skipped': 0})
        app.get_symbols_with_recent_filings = Mock(return_value=({'AAPL'}, {'10-K': 1}))
        app.update_fundamental = Mock(return_value={'success': 1, 'failed': 0, 'skipped': 0})

        target_date = dt.date(2025, 1, 10)
        app.run_daily_update(target_date=target_date, update_ticks=True, update_fundamentals=True)

        # Verify all steps were called
        app.security_master.update_from_sec.assert_called_once()
        app.check_market_open.assert_called_once_with(target_date)
        app.update_daily_ticks.assert_called_once()
        app.update_minute_ticks.assert_called_once()
        app.get_symbols_with_recent_filings.assert_called_once()
        app.update_fundamental.assert_called_once()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_run_daily_update_market_closed(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test run_daily_update on non-trading day"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        app.security_master.update_from_sec = Mock(return_value={'extended': 0, 'added': 0, 'unchanged': 100})
        app._get_symbols = Mock(return_value=['AAPL', 'MSFT'])
        app.check_market_open = Mock(return_value=False)
        app.update_daily_ticks = Mock()
        app.update_minute_ticks = Mock()
        app.get_symbols_with_recent_filings = Mock(return_value=(set(), {}))
        app.update_fundamental = Mock()

        target_date = dt.date(2025, 1, 11)  # Weekend
        app.run_daily_update(target_date=target_date, update_ticks=True, update_fundamentals=True)

        # Ticks updates should NOT be called
        app.update_daily_ticks.assert_not_called()
        app.update_minute_ticks.assert_not_called()

        # Fundamentals should still be checked
        app.get_symbols_with_recent_filings.assert_called_once()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_run_daily_update_ticks_only(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test run_daily_update with update_fundamentals=False"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        app.security_master.update_from_sec = Mock(return_value={'extended': 0, 'added': 0, 'unchanged': 100})
        app._get_symbols = Mock(return_value=['AAPL'])
        app.check_market_open = Mock(return_value=True)
        app.update_daily_ticks = Mock(return_value={'success': 1, 'failed': 0, 'skipped': 0})
        app.update_minute_ticks = Mock(return_value={'success': 1, 'failed': 0, 'skipped': 0})
        app.get_symbols_with_recent_filings = Mock()
        app.update_fundamental = Mock()

        target_date = dt.date(2025, 1, 10)
        app.run_daily_update(
            target_date=target_date,
            update_ticks=True,
            update_fundamentals=False
        )

        # Ticks should be updated
        app.update_daily_ticks.assert_called_once()
        app.update_minute_ticks.assert_called_once()

        # Fundamentals should NOT be updated
        app.get_symbols_with_recent_filings.assert_not_called()
        app.update_fundamental.assert_not_called()

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.dt.timedelta')
    @patch('quantdl.update.app_no_wrds.dt.date')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_run_daily_update_defaults_to_yesterday(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_date_class, mock_timedelta, mock_security_master, mock_calendar
    ):
        """Test run_daily_update defaults to yesterday's date"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        app.security_master.update_from_sec = Mock(return_value={'extended': 0, 'added': 0, 'unchanged': 100})
        app._get_symbols = Mock(return_value=[])
        app.check_market_open = Mock(return_value=False)
        app.get_symbols_with_recent_filings = Mock(return_value=(set(), {}))

        # Mock date and timedelta properly
        today = dt.date(2025, 1, 12)
        yesterday = dt.date(2025, 1, 11)

        mock_date_class.today.return_value = today
        mock_timedelta.return_value = dt.timedelta(days=1)

        # Mock subtraction to return yesterday
        with patch.object(type(today), '__sub__', return_value=yesterday):
            app.run_daily_update(target_date=None)

            # Should use yesterday (2025-01-11)
            app.check_market_open.assert_called_once_with(yesterday)

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_run_daily_update_no_recent_filings(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test run_daily_update when no symbols have recent filings"""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()

        app.security_master.update_from_sec = Mock(return_value={'extended': 0, 'added': 0, 'unchanged': 100})
        app._get_symbols = Mock(return_value=['AAPL', 'MSFT'])
        app.check_market_open = Mock(return_value=True)
        app.update_daily_ticks = Mock(return_value={'success': 2, 'failed': 0, 'skipped': 0})
        app.update_minute_ticks = Mock(return_value={'success': 2, 'failed': 0, 'skipped': 0})
        app.get_symbols_with_recent_filings = Mock(return_value=(set(), {}))  # Empty set
        app.update_fundamental = Mock()

        target_date = dt.date(2025, 1, 10)
        app.run_daily_update(target_date=target_date, update_ticks=True, update_fundamentals=True)

        # Fundamental update should NOT be called
        app.update_fundamental.assert_not_called()

    def test_get_cik_returns_none_for_empty_sec_map(self):
        """Test SimpleCIKResolver returns None when SEC map empty (line 101)."""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        resolver = SimpleCIKResolver(logger=Mock())
        resolver._fetch_sec_mapping = Mock(return_value=pl.DataFrame())

        result = resolver.get_cik('AAPL')
        assert result is None

    def test_batch_prefetch_ciks_returns_none_for_empty_sec_map(self):
        """Test batch_prefetch_ciks returns None for all symbols when SEC map empty (line 119)."""
        from quantdl.update.app_no_wrds import SimpleCIKResolver

        resolver = SimpleCIKResolver(logger=Mock())
        resolver._fetch_sec_mapping = Mock(return_value=pl.DataFrame())

        result = resolver.batch_prefetch_ciks(['AAPL', 'MSFT'])
        assert result == {'AAPL': None, 'MSFT': None}

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_get_recent_edgar_filings_request_exception(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test get_recent_edgar_filings handles RequestException (lines 264,265)."""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        import requests

        app = DailyUpdateAppNoWRDS()
        app.sec_rate_limiter = Mock()

        with patch('quantdl.update.app_no_wrds.requests.get', side_effect=requests.RequestException("Connection error")):
            result = app.get_recent_edgar_filings('0000320193', lookback_days=7)

        assert result == []

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_daily_ticks_skipped_no_data(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_daily_ticks skipped when no data (lines 403,404)."""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()
        app.security_master.get_security_id = Mock(return_value=1001)
        app.alpaca_ticks.fetch_daily_day_bulk = Mock(return_value={'AAPL': []})
        app._get_symbols = Mock(return_value=['AAPL'])

        stats = app.update_daily_ticks(update_date=dt.date(2025, 1, 10))

        assert stats['skipped'] >= 1

    @patch('quantdl.update.app_no_wrds.TradingCalendar')
    @patch('quantdl.update.app_no_wrds.SecurityMaster')
    @patch('quantdl.update.app_no_wrds.S3Client')
    @patch('quantdl.update.app_no_wrds.UploadConfig')
    @patch('quantdl.update.app_no_wrds.Ticks')
    @patch('quantdl.update.app_no_wrds.setup_logger')
    def test_update_fundamental_failed_result(
        self, mock_logger, mock_ticks, mock_config, mock_s3, mock_security_master, mock_calendar
    ):
        """Test update_fundamental counts failed results (lines 610,611)."""
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS

        app = DailyUpdateAppNoWRDS()
        app.cik_resolver = Mock()
        app.cik_resolver.get_cik.return_value = '0000320193'
        app.data_publishers.publish_fundamental = Mock(return_value={'status': 'failed', 'error': 'API error'})

        stats = app.update_fundamental(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )

        assert stats['failed'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
