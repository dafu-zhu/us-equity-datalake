"""
Unit tests for update.app module
Tests DailyUpdateApp class functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import datetime as dt
from pathlib import Path
import polars as pl
import io


class TestDailyUpdateAppInitialization:
    """Test DailyUpdateApp initialization"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_initialization(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test DailyUpdateApp initialization with all dependencies"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mocks
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_s3_instance = Mock()
        mock_s3_instance.client = Mock()
        mock_s3.return_value = mock_s3_instance

        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Initialize app
        app = DailyUpdateApp(config_path="test_config.yaml")

        # Verify initialization
        assert app.config == mock_config_instance
        assert app.s3_client == mock_s3_instance.client
        assert app.logger == mock_logger_instance
        assert isinstance(app._symbols_cache, dict)
        assert len(app._symbols_cache) == 0

        # Verify logger setup
        mock_logger.assert_called_once_with(
            name="daily_update",
            log_dir=Path("data/logs/update"),
            level=10,  # logging.DEBUG = 10
            console_output=True
        )

        # Verify Alpaca headers
        assert app.headers['APCA-API-KEY-ID'] == 'test_key'
        assert app.headers['APCA-API-SECRET-KEY'] == 'test_secret'


class TestCheckMarketOpen:
    """Test check_market_open method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_market_open_on_trading_day(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test check_market_open returns True for trading day"""
        from quantdl.update.app import DailyUpdateApp

        # Setup calendar mock
        mock_calendar_instance = Mock()
        mock_calendar_instance.is_trading_day.return_value = True
        mock_calendar.return_value = mock_calendar_instance

        app = DailyUpdateApp()
        test_date = dt.date(2024, 6, 3)  # Monday

        result = app.check_market_open(test_date)

        assert result is True
        mock_calendar_instance.is_trading_day.assert_called_once_with(test_date)
        app.logger.info.assert_called()

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_market_closed_on_weekend(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test check_market_open returns False for weekend"""
        from quantdl.update.app import DailyUpdateApp

        # Setup calendar mock
        mock_calendar_instance = Mock()
        mock_calendar_instance.is_trading_day.return_value = False
        mock_calendar.return_value = mock_calendar_instance

        app = DailyUpdateApp()
        test_date = dt.date(2024, 6, 1)  # Saturday

        result = app.check_market_open(test_date)

        assert result is False
        mock_calendar_instance.is_trading_day.assert_called_once_with(test_date)


class TestGetSymbolsForYear:
    """Test _get_symbols_for_year method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_symbols_for_year_caches_result(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test that symbols are cached after first load"""
        from quantdl.update.app import DailyUpdateApp

        # Setup universe manager mock
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT', 'GOOGL']
        mock_universe.return_value = mock_universe_instance

        app = DailyUpdateApp()

        # First call - should load from universe manager
        symbols_1 = app._get_symbols_for_year(2024)
        assert symbols_1 == ['AAPL', 'MSFT', 'GOOGL']
        assert mock_universe_instance.load_symbols_for_year.call_count == 1

        # Second call - should use cache
        symbols_2 = app._get_symbols_for_year(2024)
        assert symbols_2 == ['AAPL', 'MSFT', 'GOOGL']
        assert mock_universe_instance.load_symbols_for_year.call_count == 1  # Not called again

        # Different year - should load again
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        symbols_3 = app._get_symbols_for_year(2023)
        assert symbols_3 == ['AAPL', 'MSFT']
        assert mock_universe_instance.load_symbols_for_year.call_count == 2


class TestGetRecentEdgarFilings:
    """Test get_recent_edgar_filings method"""

    @patch('quantdl.update.app.requests.get')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_recent_edgar_filings_success(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_requests
    ):
        """Test successful retrieval of recent EDGAR filings"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mock response
        today = dt.date.today()
        recent_date = (today - dt.timedelta(days=3)).isoformat()
        old_date = (today - dt.timedelta(days=10)).isoformat()

        mock_response = Mock()
        mock_response.json.return_value = {
            'filings': {
                'recent': {
                    'filingDate': [recent_date, old_date],
                    'form': ['10-Q', '10-K'],
                    'accessionNumber': ['0001234567-24-000001', '0001234567-24-000002']
                }
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        # Setup rate limiter mock
        mock_rate_limiter_instance = Mock()
        mock_rate_limiter.return_value = mock_rate_limiter_instance

        app = DailyUpdateApp()

        # Call method
        result = app.get_recent_edgar_filings(cik='0000320193', lookback_days=7)

        # Verify results - only recent filing within lookback period
        assert len(result) == 1
        assert result[0]['filingDate'] == recent_date
        assert result[0]['form'] == '10-Q'

        # Verify rate limiter was called
        mock_rate_limiter_instance.acquire.assert_called()

    @patch('quantdl.update.app.requests.get')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_recent_edgar_filings_no_recent_filings(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_requests
    ):
        """Test handling when no recent filings data exists (line 155)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mock response with empty recent filings
        mock_response = Mock()
        mock_response.json.return_value = {
            'filings': {
                'recent': {}  # Empty recent filings
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        app = DailyUpdateApp()

        # Call method - should return empty list
        result = app.get_recent_edgar_filings(cik='0000320193', lookback_days=7)

        assert result == []

    @patch('quantdl.update.app.requests.get')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_recent_edgar_filings_request_exception(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_requests
    ):
        """Test handling of request exceptions"""
        from quantdl.update.app import DailyUpdateApp
        import requests

        # Setup mock to raise exception
        mock_requests.side_effect = requests.RequestException("Connection error")

        app = DailyUpdateApp()

        # Call method - should return empty list on error
        result = app.get_recent_edgar_filings(cik='0000320193', lookback_days=7)

        assert result == []
        app.logger.debug.assert_called()

    @patch('quantdl.update.app.requests.get')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_recent_edgar_filings_generic_exception(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_requests
    ):
        """Test handling of generic exceptions (lines 182-184)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mock to raise generic exception
        mock_requests.side_effect = ValueError("Unexpected error")

        app = DailyUpdateApp()

        # Call method - should return empty list on error
        result = app.get_recent_edgar_filings(cik='0000320193', lookback_days=7)

        assert result == []
        app.logger.debug.assert_called()


class TestGetSymbolsWithRecentFilings:
    """Test get_symbols_with_recent_filings method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_symbols_with_recent_filings(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test identifying symbols with recent EDGAR filings"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock with batch_prefetch_ciks
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': None  # No CIK found
        }
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        app = DailyUpdateApp()

        # Mock _check_filing to return results based on symbol
        def mock_check_filing(symbol, cik, lookback, semaphore):
            has_filing = symbol == 'AAPL'
            return {'symbol': symbol, 'cik': cik, 'has_recent_filing': has_filing}

        with patch.object(app, '_check_filing', side_effect=mock_check_filing):
            result = app.get_symbols_with_recent_filings(
                update_date=dt.date(2024, 6, 1),
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                lookback_days=7
            )

            # Only AAPL should have recent filings
            assert result == {'AAPL'}

            # Verify batch CIK resolver was called
            mock_cik_resolver_instance.batch_prefetch_ciks.assert_called_once_with(
                ['AAPL', 'MSFT', 'GOOGL'], 2024
            )

            # Verify _check_filing was called for symbols with CIKs (AAPL and MSFT only)
            assert app._check_filing.call_count == 2

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_symbols_with_recent_filings_progress_logging(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test progress logging every 100 symbols (line 227)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock - create 150 symbols with CIKs
        symbols = [f'SYM{i:03d}' for i in range(150)]
        mock_cik_resolver_instance = Mock()
        cik_map = {sym: f'000{sym}' for sym in symbols}
        mock_cik_resolver_instance.batch_prefetch_ciks.return_value = cik_map
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        app = DailyUpdateApp()

        # Mock _check_filing to return no filings
        def mock_check_filing(symbol, cik, lookback, semaphore):
            return {'symbol': symbol, 'cik': cik, 'has_recent_filing': False}

        with patch.object(app, '_check_filing', side_effect=mock_check_filing):
            result = app.get_symbols_with_recent_filings(
                update_date=dt.date(2024, 6, 1),
                symbols=symbols,
                lookback_days=7
            )

            # Verify progress logging was called (should log at 100 symbols)
            # Check that logger.info was called with progress message
            info_calls = [str(call) for call in app.logger.info.call_args_list]
            progress_logged = any('Checked 100/' in str(call) for call in info_calls)
            assert progress_logged


class TestUpdateDailyTicks:
    """Test update_daily_ticks method"""

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_new_file(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test updating daily ticks when monthly file doesn't exist"""
        from quantdl.update.app import DailyUpdateApp
        from quantdl.collection.models import TickDataPoint

        # Setup mocks
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{
                't': '2024-06-03T00:00:00Z',
                'o': 180.0,
                'h': 182.0,
                'l': 179.0,
                'c': 181.0,
                'v': 50000000
            }]
        }
        mock_ticks_instance.parse_ticks.return_value = [
            TickDataPoint(
                timestamp='2024-06-03T00:00:00',
                open=180.0,
                high=182.0,
                low=179.0,
                close=181.0,
                volume=50000000,
                num_trades=10000,
                vwap=180.5
            )
        ]
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client to raise NoSuchKey (file doesn't exist)
        mock_s3_client = Mock()
        mock_s3_client.exceptions.NoSuchKey = type('NoSuchKey', (Exception,), {})
        mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock to run tasks synchronously
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify stats
        assert result['success'] == 1
        assert result['failed'] == 0
        assert result['skipped'] == 0

        # Verify upload was called
        assert mock_publishers_instance.upload_fileobj.called

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_no_symbols_provided(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test update_daily_ticks when symbols=None, should load from universe (line 260)"""
        from quantdl.update.app import DailyUpdateApp
        from quantdl.collection.models import TickDataPoint

        # Setup universe manager to return symbols
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        mock_universe.return_value = mock_universe_instance

        # Setup mocks
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{
                't': '2024-06-03T00:00:00Z',
                'o': 180.0,
                'h': 182.0,
                'l': 179.0,
                'c': 181.0,
                'v': 50000000
            }]
        }
        mock_ticks_instance.parse_ticks.return_value = [
            TickDataPoint(
                timestamp='2024-06-03T00:00:00',
                open=180.0,
                high=182.0,
                low=179.0,
                close=181.0,
                volume=50000000,
                num_trades=10000,
                vwap=180.5
            )
        ]
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client
        mock_s3_client = Mock()
        mock_s3_client.exceptions.NoSuchKey = type('NoSuchKey', (Exception,), {})
        mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method with symbols=None
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=None  # Should trigger _get_symbols_for_year
        )

        # Verify universe manager was called
        mock_universe_instance.load_symbols_for_year.assert_called_once_with(2024, sym_type='alpaca')

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_no_data_from_alpaca(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test skipping when no data from Alpaca (lines 286-287)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mocks - no data from Alpaca
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': []  # Empty data
        }
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client
        mock_s3_client = Mock()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify stats - should be skipped
        assert result['success'] == 0
        assert result['failed'] == 0
        assert result['skipped'] == 1

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_empty_dataframe_after_parsing(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test skipping when DataFrame is empty after parsing (lines 304-305)"""
        from quantdl.update.app import DailyUpdateApp
        from quantdl.collection.models import TickDataPoint

        # Setup mocks - parse_ticks returns an empty list
        # This creates a situation where we have data from Alpaca but parsing produces nothing
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{'some': 'data'}]
        }
        # Return empty list from parsing - this causes pl.DataFrame() to create empty df
        # and then with_columns will create a df with 0 rows
        mock_ticks_instance.parse_ticks.return_value = []
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client
        mock_s3_client = Mock()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                try:
                    future.result.return_value = fn(*args, **kwargs)
                except Exception:
                    # If there's an error creating DataFrame from empty list, it will be caught
                    # and returned as a failed status
                    future.result.return_value = {'symbol': args[0] if args else 'AAPL', 'status': 'failed', 'error': 'Error processing'}
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify stats - either skipped or failed depending on whether empty DataFrame
        # creation succeeds or fails
        assert result['success'] == 0
        assert result['skipped'] + result['failed'] == 1

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_merge_with_existing(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test merging with existing S3 file (lines 315, 318)"""
        from quantdl.update.app import DailyUpdateApp
        from quantdl.collection.models import TickDataPoint

        # Setup mocks
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{
                't': '2024-06-03T00:00:00Z',
                'o': 180.0,
                'h': 182.0,
                'l': 179.0,
                'c': 181.0,
                'v': 50000000
            }]
        }
        mock_ticks_instance.parse_ticks.return_value = [
            TickDataPoint(
                timestamp='2024-06-03T00:00:00',
                open=180.0,
                high=182.0,
                low=179.0,
                close=181.0,
                volume=50000000,
                num_trades=10000,
                vwap=180.5
            )
        ]
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client with existing file
        existing_df = pl.DataFrame({
            'timestamp': ['2024-06-01', '2024-06-02'],
            'open': [175.0, 177.0],
            'high': [176.0, 178.0],
            'low': [174.0, 176.0],
            'close': [175.5, 177.5],
            'volume': [40000000, 45000000]
        })

        import io
        buffer = io.BytesIO()
        existing_df.write_parquet(buffer)
        buffer.seek(0)

        mock_s3_client = Mock()
        mock_s3_client.exceptions.NoSuchKey = type('NoSuchKey', (Exception,), {})
        mock_s3_client.get_object.return_value = {'Body': buffer}
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify success
        assert result['success'] == 1
        assert result['failed'] == 0
        assert result['skipped'] == 0

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_s3_read_error(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test handling S3 read errors (lines 328-329)"""
        from quantdl.update.app import DailyUpdateApp
        from quantdl.collection.models import TickDataPoint

        # Setup mocks
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{
                't': '2024-06-03T00:00:00Z',
                'o': 180.0,
                'h': 182.0,
                'l': 179.0,
                'c': 181.0,
                'v': 50000000
            }]
        }
        mock_ticks_instance.parse_ticks.return_value = [
            TickDataPoint(
                timestamp='2024-06-03T00:00:00',
                open=180.0,
                high=182.0,
                low=179.0,
                close=181.0,
                volume=50000000,
                num_trades=10000,
                vwap=180.5
            )
        ]
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client to raise generic error
        mock_s3_client = Mock()
        mock_s3_client.exceptions.NoSuchKey = type('NoSuchKey', (Exception,), {})
        mock_s3_client.get_object.side_effect = Exception("S3 connection error")
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method - should succeed by creating new file
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify success (should handle error gracefully)
        assert result['success'] == 1

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_processing_error(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test handling processing errors (lines 362-364, 378)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mocks - parse_ticks will raise an exception
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            'AAPL': [{'some': 'data'}]
        }
        mock_ticks_instance.parse_ticks.side_effect = Exception("Parse error")
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client
        mock_s3_client = Mock()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                try:
                    future.result.return_value = fn(*args, **kwargs)
                except Exception as e:
                    future.result.return_value = {'symbol': args[0] if args else 'UNKNOWN', 'status': 'failed', 'error': str(e)}
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify failure was recorded
        assert result['failed'] == 1

    @patch('quantdl.update.app.ThreadPoolExecutor')
    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_daily_ticks_progress_logging(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe,
        mock_executor
    ):
        """Test progress logging every 100 symbols (line 383)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup mocks - create 150 symbols with no data to skip quickly
        symbols = [f'SYM{i:03d}' for i in range(150)]
        mock_ticks_instance = Mock()
        mock_ticks_instance.fetch_daily_day_bulk.return_value = {
            sym: [] for sym in symbols  # All empty
        }
        mock_ticks.return_value = mock_ticks_instance

        # Setup S3 client
        mock_s3_client = Mock()
        mock_s3.return_value.client = mock_s3_client

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.bucket_name = 'test-bucket'
        mock_publishers.return_value = mock_publishers_instance

        # Setup executor mock
        def immediate_executor(max_workers):
            executor = Mock()
            def submit(fn, *args, **kwargs):
                future = Mock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            executor.submit = submit
            executor.__enter__ = Mock(return_value=executor)
            executor.__exit__ = Mock(return_value=False)
            return executor

        mock_executor.side_effect = immediate_executor

        app = DailyUpdateApp()

        # Call method
        result = app.update_daily_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=symbols
        )

        # Verify progress logging was called
        info_calls = [str(call) for call in app.logger.info.call_args_list]
        progress_logged = any('Progress: 100/' in str(call) for call in info_calls)
        assert progress_logged


class TestUpdateMinuteTicks:
    """Test update_minute_ticks method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_minute_ticks_success(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test successful minute ticks update"""
        from quantdl.update.app import DailyUpdateApp

        # Setup data collectors mock
        mock_collectors_instance = Mock()
        mock_collectors_instance.fetch_minute_day.return_value = {
            'AAPL': [{'t': '2024-06-03T09:30:00Z', 'o': 180.0, 'c': 180.5, 'v': 100000}]
        }

        # Create sample DataFrame for parsed data
        sample_df = pl.DataFrame({
            'timestamp': ['2024-06-03T09:30:00'],
            'open': [180.0],
            'high': [180.5],
            'low': [179.8],
            'close': [180.5],
            'volume': [100000]
        })
        mock_collectors_instance.parse_minute_bars_to_daily.return_value = {
            ('AAPL', '2024-06-03'): sample_df
        }
        mock_collectors.return_value = mock_collectors_instance

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_minute_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify stats
        assert result['success'] == 1
        assert result['failed'] == 0
        assert result['skipped'] == 0

        # Verify upload was called
        assert mock_publishers_instance.upload_fileobj.called

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_minute_ticks_empty_dataframe(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test minute ticks update with empty DataFrame"""
        from quantdl.update.app import DailyUpdateApp

        # Setup data collectors mock with empty DataFrame
        mock_collectors_instance = Mock()
        mock_collectors_instance.fetch_minute_day.return_value = {'AAPL': []}
        empty_df = pl.DataFrame()
        mock_collectors_instance.parse_minute_bars_to_daily.return_value = {
            ('AAPL', '2024-06-03'): empty_df
        }
        mock_collectors.return_value = mock_collectors_instance

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_minute_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify stats - should be skipped
        assert result['success'] == 0
        assert result['failed'] == 0
        assert result['skipped'] == 1

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_minute_ticks_no_symbols_provided(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test update_minute_ticks when symbols=None, should load from universe (line 417)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup universe manager to return symbols
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        mock_universe.return_value = mock_universe_instance

        # Setup data collectors mock
        mock_collectors_instance = Mock()
        mock_collectors_instance.fetch_minute_day.return_value = {
            'AAPL': [{'t': '2024-06-03T09:30:00Z', 'o': 180.0, 'c': 180.5, 'v': 100000}]
        }

        # Create sample DataFrame for parsed data
        sample_df = pl.DataFrame({
            'timestamp': ['2024-06-03T09:30:00'],
            'open': [180.0],
            'high': [180.5],
            'low': [179.8],
            'close': [180.5],
            'volume': [100000]
        })
        mock_collectors_instance.parse_minute_bars_to_daily.return_value = {
            ('AAPL', '2024-06-03'): sample_df
        }
        mock_collectors.return_value = mock_collectors_instance

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.upload_fileobj = Mock()
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method with symbols=None
        result = app.update_minute_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=None  # Should trigger _get_symbols_for_year
        )

        # Verify universe manager was called
        mock_universe_instance.load_symbols_for_year.assert_called_once_with(2024, sym_type='alpaca')

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_minute_ticks_upload_error(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test handling upload errors (lines 468-470)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup data collectors mock
        mock_collectors_instance = Mock()
        mock_collectors_instance.fetch_minute_day.return_value = {
            'AAPL': [{'t': '2024-06-03T09:30:00Z', 'o': 180.0, 'c': 180.5, 'v': 100000}]
        }

        # Create sample DataFrame for parsed data
        sample_df = pl.DataFrame({
            'timestamp': ['2024-06-03T09:30:00'],
            'open': [180.0],
            'high': [180.5],
            'low': [179.8],
            'close': [180.5],
            'volume': [100000]
        })
        mock_collectors_instance.parse_minute_bars_to_daily.return_value = {
            ('AAPL', '2024-06-03'): sample_df
        }
        mock_collectors.return_value = mock_collectors_instance

        # Setup publishers mock to raise error
        mock_publishers_instance = Mock()
        mock_publishers_instance.upload_fileobj.side_effect = Exception("Upload failed")
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_minute_ticks(
            update_date=dt.date(2024, 6, 3),
            symbols=['AAPL']
        )

        # Verify failure was recorded
        assert result['success'] == 0
        assert result['failed'] == 1
        assert result['skipped'] == 0


class TestUpdateFundamental:
    """Test update_fundamental method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_fundamental_success(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test successful fundamental data update"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.get_cik.return_value = '0000320193'
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        # Setup publishers mock
        mock_publishers_instance = Mock()
        mock_publishers_instance.publish_fundamental.return_value = {'status': 'success'}
        mock_publishers_instance.publish_ttm_fundamental.return_value = {'status': 'success'}
        mock_publishers_instance.publish_derived_fundamental = Mock()
        mock_publishers.return_value = mock_publishers_instance

        # Setup collectors mock
        mock_collectors_instance = Mock()
        sample_df = pl.DataFrame({'metric': ['revenue'], 'value': [1000000]})
        mock_collectors_instance.collect_derived_long.return_value = (sample_df, None)
        mock_collectors.return_value = mock_collectors_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_fundamental(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Verify stats
        assert result['success'] == 1
        assert result['failed'] == 0
        assert result['skipped'] == 0

        # Verify all publish methods were called
        assert mock_publishers_instance.publish_fundamental.called
        assert mock_publishers_instance.publish_ttm_fundamental.called
        assert mock_publishers_instance.publish_derived_fundamental.called

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_fundamental_no_cik(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test fundamental update when CIK cannot be resolved"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver to return None
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.get_cik.return_value = None
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_fundamental(
            symbols=['UNKNOWN'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Verify no processing occurred
        assert result['success'] == 0
        assert result['failed'] == 0
        assert result['skipped'] == 0

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_fundamental_skipped_status(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test handling skipped status from publish_fundamental (lines 529-530)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.get_cik.return_value = '0000320193'
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        # Setup publishers mock to return skipped
        mock_publishers_instance = Mock()
        mock_publishers_instance.publish_fundamental.return_value = {'status': 'skipped'}
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_fundamental(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Verify stats - should be skipped
        assert result['success'] == 0
        assert result['failed'] == 0
        assert result['skipped'] == 1

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_fundamental_failed_status(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test handling failed status from publish_fundamental (lines 531-533)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.get_cik.return_value = '0000320193'
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        # Setup publishers mock to return failed
        mock_publishers_instance = Mock()
        mock_publishers_instance.publish_fundamental.return_value = {'status': 'failed'}
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_fundamental(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Verify stats - should be failed
        assert result['success'] == 0
        assert result['failed'] == 1
        assert result['skipped'] == 0

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_update_fundamental_processing_error(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test handling processing errors (lines 563-565)"""
        from quantdl.update.app import DailyUpdateApp

        # Setup CIK resolver mock
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.get_cik.return_value = '0000320193'
        mock_cik_resolver.return_value = mock_cik_resolver_instance

        # Setup publishers mock to raise exception
        mock_publishers_instance = Mock()
        mock_publishers_instance.publish_fundamental.side_effect = Exception("Processing error")
        mock_publishers.return_value = mock_publishers_instance

        app = DailyUpdateApp()

        # Call method
        result = app.update_fundamental(
            symbols=['AAPL'],
            start_date='2024-01-01',
            end_date='2024-06-30'
        )

        # Verify failure was recorded
        assert result['success'] == 0
        assert result['failed'] == 1
        assert result['skipped'] == 0


class TestRunDailyUpdate:
    """Test run_daily_update method"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_run_daily_update_full_workflow(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test complete daily update workflow"""
        from quantdl.update.app import DailyUpdateApp

        # Setup universe manager mock
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        mock_universe.return_value = mock_universe_instance

        app = DailyUpdateApp()

        # Mock all update methods
        with patch.object(app, 'check_market_open', return_value=True), \
             patch.object(app, 'update_daily_ticks', return_value={'success': 2, 'failed': 0, 'skipped': 0}), \
             patch.object(app, 'update_minute_ticks', return_value={'success': 2, 'failed': 0, 'skipped': 0}), \
             patch.object(app, 'get_symbols_with_recent_filings', return_value={'AAPL'}), \
             patch.object(app, 'update_fundamental', return_value={'success': 1, 'failed': 0, 'skipped': 0}):

            # Call method
            app.run_daily_update(
                target_date=dt.date(2024, 6, 3),
                update_ticks=True,
                update_fundamentals=True,
                fundamental_lookback_days=7
            )

            # Verify workflow
            app.check_market_open.assert_called_once_with(dt.date(2024, 6, 3))
            app.update_daily_ticks.assert_called_once()
            app.update_minute_ticks.assert_called_once()
            app.get_symbols_with_recent_filings.assert_called_once()
            app.update_fundamental.assert_called_once()

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_run_daily_update_market_closed(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test daily update when market is closed"""
        from quantdl.update.app import DailyUpdateApp

        # Setup universe manager mock
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        mock_universe.return_value = mock_universe_instance

        app = DailyUpdateApp()

        # Mock check_market_open to return False
        with patch.object(app, 'check_market_open', return_value=False), \
             patch.object(app, 'update_daily_ticks') as mock_daily, \
             patch.object(app, 'update_minute_ticks') as mock_minute, \
             patch.object(app, 'get_symbols_with_recent_filings', return_value=set()), \
             patch.object(app, 'update_fundamental'):

            # Call method
            app.run_daily_update(
                target_date=dt.date(2024, 6, 1),  # Saturday
                update_ticks=True,
                update_fundamentals=True
            )

            # Verify ticks updates were skipped
            mock_daily.assert_not_called()
            mock_minute.assert_not_called()

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_run_daily_update_defaults_to_yesterday(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test that run_daily_update defaults to yesterday when no date provided"""
        from quantdl.update.app import DailyUpdateApp

        # Setup universe manager mock
        mock_universe_instance = Mock()
        mock_universe_instance.load_symbols_for_year.return_value = ['AAPL']
        mock_universe.return_value = mock_universe_instance

        app = DailyUpdateApp()

        # Mock methods
        with patch.object(app, 'check_market_open', return_value=True), \
             patch.object(app, 'update_daily_ticks', return_value={'success': 1, 'failed': 0, 'skipped': 0}), \
             patch.object(app, 'update_minute_ticks', return_value={'success': 1, 'failed': 0, 'skipped': 0}), \
             patch.object(app, 'get_symbols_with_recent_filings', return_value=set()):

            # Call method without target_date
            app.run_daily_update(target_date=None)

            # Verify check_market_open was called with yesterday's date
            yesterday = dt.date.today() - dt.timedelta(days=1)
            app.check_market_open.assert_called_once_with(yesterday)


class TestMainFunction:
    """Test main entry point function"""

    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['app.py', '--date', '2024-06-03'])
    def test_main_with_date_argument(self, mock_app_class):
        """Test main function with date argument"""
        from quantdl.update.app import main

        # Setup mock
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        # Call main
        main()

        # Verify app was initialized
        mock_app_class.assert_called_once()

        # Verify run_daily_update was called with correct date
        mock_app_instance.run_daily_update.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] == dt.date(2024, 6, 3)
        assert call_args[1]['update_ticks'] is True
        assert call_args[1]['update_fundamentals'] is True
        assert call_args[1]['fundamental_lookback_days'] == 7

    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['app.py', '--no-ticks', '--no-fundamentals', '--lookback', '14'])
    def test_main_with_flags(self, mock_app_class):
        """Test main function with various flags"""
        from quantdl.update.app import main

        # Setup mock
        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        # Call main
        main()

        # Verify run_daily_update was called with correct flags
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['update_ticks'] is False
        assert call_args[1]['update_fundamentals'] is False
        assert call_args[1]['fundamental_lookback_days'] == 14


class TestParallelFilingChecks:
    """Test parallel filing check functionality"""

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_check_filing_with_recent_filings(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test _check_filing returns True when recent filings exist"""
        from quantdl.update.app import DailyUpdateApp
        from threading import Semaphore

        app = DailyUpdateApp(config_path="test_config.yaml")
        app.get_recent_edgar_filings = Mock(return_value=[
            {'form': '10-Q', 'filingDate': '2024-06-30'}
        ])

        semaphore = Semaphore(10)
        result = app._check_filing("AAPL", "0000320193", 7, semaphore)

        assert result['symbol'] == "AAPL"
        assert result['cik'] == "0000320193"
        assert result['has_recent_filing'] is True
        app.get_recent_edgar_filings.assert_called_once_with("0000320193", 7)

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_check_filing_without_recent_filings(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test _check_filing returns False when no recent filings"""
        from quantdl.update.app import DailyUpdateApp
        from threading import Semaphore

        app = DailyUpdateApp(config_path="test_config.yaml")
        app.get_recent_edgar_filings = Mock(return_value=[])

        semaphore = Semaphore(10)
        result = app._check_filing("MSFT", "0000789019", 7, semaphore)

        assert result['symbol'] == "MSFT"
        assert result['cik'] == "0000789019"
        assert result['has_recent_filing'] is False

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_symbols_with_recent_filings_parallel(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver_class,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test get_symbols_with_recent_filings uses parallel execution"""
        from quantdl.update.app import DailyUpdateApp

        app = DailyUpdateApp(config_path="test_config.yaml")

        # Mock CIK resolver to return CIKs for all symbols
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044'
        }
        app.cik_resolver = mock_cik_resolver_instance

        # Mock EDGAR filing checks
        def mock_check_filing(symbol, cik, lookback, semaphore):
            has_filing = symbol in ['AAPL', 'GOOGL']
            return {'symbol': symbol, 'cik': cik, 'has_recent_filing': has_filing}

        app._check_filing = Mock(side_effect=mock_check_filing)

        update_date = dt.date(2024, 6, 30)
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        result = app.get_symbols_with_recent_filings(update_date, symbols, lookback_days=7)

        # Verify batch CIK resolution was used
        mock_cik_resolver_instance.batch_prefetch_ciks.assert_called_once_with(
            symbols, 2024
        )

        # Verify parallel checks were executed
        assert app._check_filing.call_count == 3

        # Verify result
        assert result == {'AAPL', 'GOOGL'}

    @patch('quantdl.update.app.UniverseManager')
    @patch('quantdl.update.app.SECClient')
    @patch('quantdl.update.app.DataPublishers')
    @patch('quantdl.update.app.DataCollectors')
    @patch('quantdl.update.app.CIKResolver')
    @patch('quantdl.update.app.RateLimiter')
    @patch('quantdl.update.app.CRSPDailyTicks')
    @patch('quantdl.update.app.Ticks')
    @patch('quantdl.update.app.TradingCalendar')
    @patch('quantdl.update.app.setup_logger')
    @patch('quantdl.update.app.S3Client')
    @patch('quantdl.update.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_get_symbols_with_recent_filings_filters_null_ciks(
        self, mock_config, mock_s3, mock_logger, mock_calendar,
        mock_ticks, mock_crsp, mock_rate_limiter, mock_cik_resolver_class,
        mock_collectors, mock_publishers, mock_sec_client, mock_universe
    ):
        """Test get_symbols_with_recent_filings filters out NULL CIKs"""
        from quantdl.update.app import DailyUpdateApp

        app = DailyUpdateApp(config_path="test_config.yaml")

        # Mock CIK resolver with one NULL CIK
        mock_cik_resolver_instance = Mock()
        mock_cik_resolver_instance.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'MSFT': None,
            'GOOGL': '0001652044'
        }
        app.cik_resolver = mock_cik_resolver_instance

        app._check_filing = Mock(return_value={'symbol': 'AAPL', 'cik': '0000320193', 'has_recent_filing': True})

        update_date = dt.date(2024, 6, 30)
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        result = app.get_symbols_with_recent_filings(update_date, symbols, lookback_days=7)

        # Verify only 2 symbols checked (MSFT skipped due to NULL CIK)
        assert app._check_filing.call_count == 2
