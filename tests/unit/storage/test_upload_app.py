"""
Unit tests for storage.app module
Focus on UploadApp initialization and handler delegation
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import datetime as dt
import logging
import os
import polars as pl
from botocore.exceptions import ClientError


def _make_app():
    """Create a minimal mock UploadApp for testing."""
    from quantdl.storage.app import UploadApp

    app = UploadApp.__new__(UploadApp)
    app.logger = Mock()

    # Mock S3 client - simulate no progress file exists
    app.client = Mock()
    error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
    app.client.get_object.side_effect = ClientError(error_response, 'GetObject')

    # Mock config
    app.config = Mock()
    app.config.bucket = 'test-bucket'

    # Mock dependencies
    app.validator = Mock()
    app.data_collectors = Mock()
    app.data_publishers = Mock()
    app.data_collectors.ticks_collector = Mock(alpaca_start_year=2025)
    app.universe_manager = Mock()
    app.calendar = Mock()
    app.cik_resolver = Mock()
    app.sec_rate_limiter = Mock()
    app.crsp_ticks = Mock()
    app.security_master = Mock()
    app._wrds_available = True
    app._alpaca_start_year = 2025
    app.headers = {
        "APCA-API-KEY-ID": "test_key",
        "APCA-API-SECRET-KEY": "test_secret"
    }
    app.alpaca_ticks = Mock()

    return app


class TestUploadAppInitialization:
    """Test UploadApp constructor and initialization."""

    @patch('quantdl.storage.app.DataPublishers')
    @patch('quantdl.storage.app.DataCollectors')
    @patch('quantdl.storage.app.CIKResolver')
    @patch('quantdl.storage.app.RateLimiter')
    @patch('quantdl.storage.app.TradingCalendar')
    @patch('quantdl.storage.app.UniverseManager')
    @patch('quantdl.storage.app.CRSPDailyTicks')
    @patch('quantdl.storage.app.Ticks')
    @patch('quantdl.storage.app.Validator')
    @patch('quantdl.storage.app.setup_logger')
    @patch('quantdl.storage.app.S3Client')
    @patch('quantdl.storage.app.UploadConfig')
    @patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_API_SECRET': 'test_secret'
    })
    def test_initialization(
        self,
        mock_config,
        mock_s3_client,
        mock_logger,
        mock_validator,
        mock_ticks,
        mock_crsp,
        mock_universe,
        mock_calendar,
        mock_rate_limiter,
        mock_cik_resolver,
        mock_collectors,
        mock_publishers
    ):
        """Test UploadApp constructor wiring and defaults."""
        from quantdl.storage.app import UploadApp

        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_s3_instance = Mock()
        mock_s3_instance.client = Mock()
        mock_s3_client.return_value = mock_s3_instance

        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        app = UploadApp(alpaca_start_year=2025)

        assert app.config == mock_config_instance
        assert app.client == mock_s3_instance.client
        assert app.logger == mock_logger_instance
        assert app.validator == mock_validator.return_value
        assert app.alpaca_ticks == mock_ticks.return_value
        assert app.crsp_ticks == mock_crsp.return_value
        assert app.universe_manager == mock_universe.return_value
        assert app.calendar == mock_calendar.return_value
        assert app.sec_rate_limiter == mock_rate_limiter.return_value
        assert app.cik_resolver == mock_cik_resolver.return_value
        assert app.data_collectors == mock_collectors.return_value
        assert app.data_publishers == mock_publishers.return_value

        mock_logger.assert_called_once_with(
            name="uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.DEBUG,
            console_output=True
        )
        assert app.headers["APCA-API-KEY-ID"] == "test_key"
        assert app.headers["APCA-API-SECRET-KEY"] == "test_secret"


class TestUploadAppDailyTicks:
    """Test daily ticks upload delegation to handler."""

    def test_upload_daily_ticks_delegates_to_handler(self):
        """Test upload_daily_ticks creates handler and calls upload_year."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload_year.return_value = None

        with patch.object(app, '_get_daily_ticks_handler', return_value=mock_handler):
            result = app.upload_daily_ticks(
                year=2024,
                overwrite=True,
                chunk_size=100,
                sleep_time=0.5,
                current_year=2024
            )

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            overwrite=True,
            chunk_size=100,
            sleep_time=0.5,
            current_year=2024
        )
        assert result is None

    def test_upload_daily_ticks_default_params(self):
        """Test upload_daily_ticks uses default parameters."""
        app = _make_app()
        mock_handler = Mock()

        with patch.object(app, '_get_daily_ticks_handler', return_value=mock_handler):
            app.upload_daily_ticks(year=2024)

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            overwrite=False,
            chunk_size=200,
            sleep_time=0.2,
            current_year=None
        )


class TestUploadAppMinuteTicks:
    """Test minute ticks upload delegation to handler."""

    def test_upload_minute_ticks_delegates_to_handler(self):
        """Test upload_minute_ticks creates handler and calls upload_year."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload_year.return_value = None

        with patch.object(app, '_get_minute_ticks_handler', return_value=mock_handler):
            app.upload_minute_ticks(
                year=2024,
                month=6,
                overwrite=True,
                resume=True,
                num_workers=25
            )

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            months=[6],
            overwrite=True,
            resume=True,
            num_workers=25,
            chunk_size=500,
            sleep_time=0.0
        )

    def test_upload_minute_ticks_year_multiple_months(self):
        """Test upload_minute_ticks_year with multiple months."""
        app = _make_app()
        mock_handler = Mock()

        with patch.object(app, '_get_minute_ticks_handler', return_value=mock_handler):
            app.upload_minute_ticks_year(
                year=2024,
                months=[1, 2, 3],
                overwrite=False
            )

        mock_handler.upload_year.assert_called_once_with(
            year=2024,
            months=[1, 2, 3],
            overwrite=False,
            resume=False,
            num_workers=50,
            chunk_size=500,
            sleep_time=0.0
        )


class TestUploadAppFundamental:
    """Test fundamental data upload delegation to handlers."""

    def test_upload_fundamental_delegates_to_handler(self):
        """Test upload_fundamental creates handler and calls upload."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload.return_value = {'success': 10, 'failed': 0}

        with patch.object(app, '_get_fundamental_handler', return_value=mock_handler):
            result = app.upload_fundamental(
                start_date='2024-01-01',
                end_date='2024-12-31',
                max_workers=25,
                overwrite=True
            )

        mock_handler.upload.assert_called_once_with(
            '2024-01-01', '2024-12-31', 25, True
        )
        assert result == {'success': 10, 'failed': 0}

    def test_upload_ttm_fundamental_delegates_to_handler(self):
        """Test upload_ttm_fundamental creates handler and calls upload."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload.return_value = None

        with patch.object(app, '_get_ttm_handler', return_value=mock_handler):
            app.upload_ttm_fundamental(
                start_date='2024-01-01',
                end_date='2024-12-31'
            )

        mock_handler.upload.assert_called_once_with(
            '2024-01-01', '2024-12-31', 50, False
        )

    def test_upload_derived_fundamental_delegates_to_handler(self):
        """Test upload_derived_fundamental creates handler and calls upload."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload.return_value = None

        with patch.object(app, '_get_derived_handler', return_value=mock_handler):
            app.upload_derived_fundamental(
                start_date='2024-01-01',
                end_date='2024-12-31',
                max_workers=100
            )

        mock_handler.upload.assert_called_once_with(
            '2024-01-01', '2024-12-31', 100, False
        )


class TestUploadAppTop3000:
    """Test top 3000 upload delegation to handler."""

    def test_upload_top_3000_monthly_delegates_to_handler(self):
        """Test upload_top_3000_monthly creates handler and calls upload_year."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload_year.return_value = None

        with patch.object(app, '_get_top3000_handler', return_value=mock_handler):
            app.upload_top_3000_monthly(
                year=2024,
                overwrite=True,
                auto_resolve=False
            )

        mock_handler.upload_year.assert_called_once_with(2024, True, False)


class TestUploadAppSentiment:
    """Test sentiment upload delegation to handler."""

    def test_upload_sentiment_delegates_to_handler(self):
        """Test upload_sentiment creates handler and calls upload."""
        app = _make_app()
        mock_handler = Mock()
        mock_handler.upload.return_value = None

        with patch.object(app, '_get_sentiment_handler', return_value=mock_handler):
            app.upload_sentiment(
                start_date='2024-01-01',
                end_date='2024-12-31',
                overwrite=True
            )

        mock_handler.upload.assert_called_once_with(
            '2024-01-01', '2024-12-31', True
        )


class TestUploadAppRun:
    """Test UploadApp.run() orchestration."""

    def test_run_invokes_selected_flows(self):
        """Test run() invokes only the selected upload methods."""
        app = _make_app()

        # Mock the internal methods that run() calls
        app._run_daily_ticks = Mock()
        app.upload_minute_ticks_year = Mock()
        app.upload_fundamental = Mock()
        app.upload_ttm_fundamental = Mock()
        app.upload_derived_fundamental = Mock()
        app.upload_top_3000_monthly = Mock()
        app.upload_sentiment = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            run_fundamental=True,
            run_daily_ticks=True,
            run_minute_ticks=False,
            run_derived_fundamental=False,
            run_ttm_fundamental=False,
            run_top_3000=False
        )

        # Fundamental and daily ticks should be called
        assert app.upload_fundamental.called
        assert app._run_daily_ticks.called

        # Others should not be called
        assert not app.upload_minute_ticks_year.called
        assert not app.upload_derived_fundamental.called
        assert not app.upload_ttm_fundamental.called
        assert not app.upload_top_3000_monthly.called

    def test_run_all_enables_all_flows(self):
        """Test run_all=True enables all upload methods except minute ticks."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_minute_ticks_year = Mock()
        app.upload_fundamental = Mock()
        app.upload_ttm_fundamental = Mock()
        app.upload_derived_fundamental = Mock()
        app.upload_top_3000_monthly = Mock()
        app.upload_sentiment = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            run_all=True,
            minute_ticks_start_year=2024
        )

        assert app._run_daily_ticks.called
        assert not app.upload_minute_ticks_year.called  # minute ticks excluded from run_all by design
        assert app.upload_fundamental.called
        assert app.upload_ttm_fundamental.called
        assert app.upload_derived_fundamental.called
        assert app.upload_top_3000_monthly.called
        assert app.upload_sentiment.called

    def test_run_skips_minute_ticks_before_start_year(self):
        """Test run() skips minute ticks for years before minute_ticks_start_year."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_minute_ticks_year = Mock()
        app.upload_fundamental = Mock()

        app.run(
            start_year=2015,
            end_year=2016,
            run_minute_ticks=True,
            minute_ticks_start_year=2017
        )

        # Minute ticks should not be called (years are before 2017)
        assert not app.upload_minute_ticks_year.called

    def test_run_minute_ticks_runs_all_months(self):
        """Test run() calls minute ticks for valid years."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_minute_ticks_year = Mock()
        app.upload_fundamental = Mock()

        app.run(
            start_year=2020,
            end_year=2020,
            run_minute_ticks=True,
            minute_ticks_start_year=2017
        )

        app.upload_minute_ticks_year.assert_called()
        # Check that 2020 was passed
        call_args = app.upload_minute_ticks_year.call_args
        assert call_args[0][0] == 2020 or call_args[1].get('year') == 2020

    def test_run_passes_daily_chunk_and_sleep(self):
        """Test run() passes daily_chunk_size and daily_sleep_time to _run_daily_ticks."""
        app = _make_app()

        app._run_daily_ticks = Mock()
        app.upload_fundamental = Mock()

        app.run(
            start_year=2024,
            end_year=2024,
            run_daily_ticks=True,
            daily_chunk_size=100,
            daily_sleep_time=0.5
        )

        app._run_daily_ticks.assert_called_once_with(2024, 2024, False, 100, 0.5)


class TestUploadAppClose:
    """Test UploadApp resource cleanup."""

    def test_close_closes_wrds_connection(self):
        """Test close() closes WRDS connection."""
        app = _make_app()
        app.crsp_ticks.conn = Mock()

        app.close()

        app.crsp_ticks.conn.close.assert_called_once()

    def test_close_without_crsp_conn(self):
        """Test close() handles missing CRSP connection."""
        app = _make_app()
        app.crsp_ticks.conn = None

        # Should not raise
        app.close()

    def test_close_without_universe_manager(self):
        """Test close() handles missing universe manager."""
        app = _make_app()
        app.crsp_ticks.conn = None
        app.universe_manager = None

        # Should not raise
        app.close()


class TestUploadAppHandlerFactories:
    """Test handler factory methods create correct handlers."""

    def test_get_daily_ticks_handler_creates_handler(self):
        """Test _get_daily_ticks_handler creates DailyTicksHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.ticks.DailyTicksHandler') as MockHandler:
            handler = app._get_daily_ticks_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['s3_client'] == app.client
        assert call_kwargs['bucket_name'] == 'test-bucket'
        assert call_kwargs['data_publishers'] == app.data_publishers
        assert call_kwargs['logger'] == app.logger

    def test_get_minute_ticks_handler_creates_handler(self):
        """Test _get_minute_ticks_handler creates MinuteTicksHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.ticks.MinuteTicksHandler') as MockHandler:
            handler = app._get_minute_ticks_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['s3_client'] == app.client
        assert call_kwargs['calendar'] == app.calendar

    def test_get_fundamental_handler_creates_handler(self):
        """Test _get_fundamental_handler creates FundamentalHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.fundamental.FundamentalHandler') as MockHandler:
            handler = app._get_fundamental_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['cik_resolver'] == app.cik_resolver
        assert call_kwargs['sec_rate_limiter'] == app.sec_rate_limiter

    def test_get_ttm_handler_creates_handler(self):
        """Test _get_ttm_handler creates TTMHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.fundamental.TTMHandler') as MockHandler:
            handler = app._get_ttm_handler()

        MockHandler.assert_called_once()

    def test_get_derived_handler_creates_handler(self):
        """Test _get_derived_handler creates DerivedHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.fundamental.DerivedHandler') as MockHandler:
            handler = app._get_derived_handler()

        MockHandler.assert_called_once()

    def test_get_top3000_handler_creates_handler(self):
        """Test _get_top3000_handler creates Top3000Handler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.top3000.Top3000Handler') as MockHandler:
            handler = app._get_top3000_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['alpaca_start_year'] == 2025

    def test_get_sentiment_handler_creates_handler(self):
        """Test _get_sentiment_handler creates SentimentHandler."""
        app = _make_app()

        with patch('quantdl.storage.handlers.sentiment.SentimentHandler') as MockHandler:
            handler = app._get_sentiment_handler()

        MockHandler.assert_called_once()
        call_kwargs = MockHandler.call_args[1]
        assert call_kwargs['bucket'] == 'test-bucket'
