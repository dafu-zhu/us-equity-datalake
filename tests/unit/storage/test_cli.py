"""Unit tests for storage CLI."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import argparse
import datetime as dt


class TestStorageCLI:
    """Tests for storage/cli.py main function."""

    @patch('quantdl.storage.cli.UploadApp')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    def test_main_with_fundamental_flag(self, mock_parse_args, mock_upload_app):
        """Test CLI with --run-fundamental flag."""
        from quantdl.storage.cli import main

        mock_args = Mock()
        mock_args.start_year = 2020
        mock_args.end_year = 2024
        mock_args.overwrite = False
        mock_args.resume = False
        mock_args.run_fundamental = True
        mock_args.run_derived_fundamental = False
        mock_args.run_ttm_fundamental = False
        mock_args.run_daily_ticks = False
        mock_args.run_minute_ticks = False
        mock_args.run_top_3000 = False
        mock_args.run_sentiment = False
        mock_args.run_all = False
        mock_args.alpaca_start_year = 2025
        mock_args.minute_start_year = 2017
        mock_args.daily_chunk_size = 200
        mock_args.daily_sleep_time = 0.2
        mock_args.max_workers = 50
        mock_args.minute_workers = 50
        mock_args.minute_chunk_size = 500
        mock_args.minute_sleep_time = 0.0
        mock_parse_args.return_value = mock_args

        mock_app = Mock()
        mock_upload_app.return_value = mock_app

        main()

        mock_upload_app.assert_called_once_with(alpaca_start_year=2025)
        mock_app.run.assert_called_once()
        mock_app.close.assert_called_once()

    @patch('quantdl.storage.cli.UploadApp')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    def test_main_with_run_all_flag(self, mock_parse_args, mock_upload_app):
        """Test CLI with --run-all flag."""
        from quantdl.storage.cli import main

        mock_args = Mock()
        mock_args.start_year = 2009
        mock_args.end_year = 2024
        mock_args.overwrite = True
        mock_args.resume = False
        mock_args.run_fundamental = False
        mock_args.run_derived_fundamental = False
        mock_args.run_ttm_fundamental = False
        mock_args.run_daily_ticks = False
        mock_args.run_minute_ticks = False
        mock_args.run_top_3000 = False
        mock_args.run_sentiment = False
        mock_args.run_all = True
        mock_args.alpaca_start_year = 2025
        mock_args.minute_start_year = 2017
        mock_args.daily_chunk_size = 200
        mock_args.daily_sleep_time = 0.2
        mock_args.max_workers = 50
        mock_args.minute_workers = 50
        mock_args.minute_chunk_size = 500
        mock_args.minute_sleep_time = 0.0
        mock_parse_args.return_value = mock_args

        mock_app = Mock()
        mock_upload_app.return_value = mock_app

        main()

        call_kwargs = mock_app.run.call_args[1]
        assert call_kwargs['run_all'] is True
        assert call_kwargs['overwrite'] is True

    @patch('quantdl.storage.cli.UploadApp')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    def test_main_calls_close_on_exception(self, mock_parse_args, mock_upload_app):
        """Test that close() is called even when run() raises exception."""
        from quantdl.storage.cli import main

        mock_args = Mock()
        mock_args.start_year = 2020
        mock_args.end_year = 2024
        mock_args.overwrite = False
        mock_args.resume = False
        mock_args.run_fundamental = True
        mock_args.run_derived_fundamental = False
        mock_args.run_ttm_fundamental = False
        mock_args.run_daily_ticks = False
        mock_args.run_minute_ticks = False
        mock_args.run_top_3000 = False
        mock_args.run_sentiment = False
        mock_args.run_all = False
        mock_args.alpaca_start_year = 2025
        mock_args.minute_start_year = 2017
        mock_args.daily_chunk_size = 200
        mock_args.daily_sleep_time = 0.2
        mock_args.max_workers = 50
        mock_args.minute_workers = 50
        mock_args.minute_chunk_size = 500
        mock_args.minute_sleep_time = 0.0
        mock_parse_args.return_value = mock_args

        mock_app = Mock()
        mock_app.run.side_effect = RuntimeError("Upload failed")
        mock_upload_app.return_value = mock_app

        with pytest.raises(RuntimeError, match="Upload failed"):
            main()

        # close() should still be called
        mock_app.close.assert_called_once()

    @patch('quantdl.storage.cli.dt')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.error')
    def test_main_rejects_future_end_year(self, mock_error, mock_parse_args, mock_dt):
        """Test that end_year in the future raises error."""
        from quantdl.storage.cli import main

        # Mock today as 2024-06-15
        mock_dt.date.today.return_value = dt.date(2024, 6, 15)

        mock_args = Mock()
        mock_args.start_year = 2020
        mock_args.end_year = 2025  # Future year
        mock_parse_args.return_value = mock_args

        mock_error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            main()

        mock_error.assert_called_once()
        assert "2025" in str(mock_error.call_args)

    @patch('quantdl.storage.cli.UploadApp')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    def test_main_with_sentiment_flag(self, mock_parse_args, mock_upload_app):
        """Test CLI with --run-sentiment flag."""
        from quantdl.storage.cli import main

        mock_args = Mock()
        mock_args.start_year = 2020
        mock_args.end_year = 2024
        mock_args.overwrite = False
        mock_args.resume = False
        mock_args.run_fundamental = False
        mock_args.run_derived_fundamental = False
        mock_args.run_ttm_fundamental = False
        mock_args.run_daily_ticks = False
        mock_args.run_minute_ticks = False
        mock_args.run_top_3000 = False
        mock_args.run_sentiment = True
        mock_args.run_all = False
        mock_args.alpaca_start_year = 2025
        mock_args.minute_start_year = 2017
        mock_args.daily_chunk_size = 200
        mock_args.daily_sleep_time = 0.2
        mock_args.max_workers = 50
        mock_args.minute_workers = 50
        mock_args.minute_chunk_size = 500
        mock_args.minute_sleep_time = 0.0
        mock_parse_args.return_value = mock_args

        mock_app = Mock()
        mock_upload_app.return_value = mock_app

        main()

        call_kwargs = mock_app.run.call_args[1]
        assert call_kwargs['run_sentiment'] is True

    @patch('quantdl.storage.cli.UploadApp')
    @patch('quantdl.storage.cli.argparse.ArgumentParser.parse_args')
    def test_main_with_resume_flag(self, mock_parse_args, mock_upload_app):
        """Test CLI with --resume flag."""
        from quantdl.storage.cli import main

        mock_args = Mock()
        mock_args.start_year = 2020
        mock_args.end_year = 2024
        mock_args.overwrite = False
        mock_args.resume = True
        mock_args.run_fundamental = False
        mock_args.run_derived_fundamental = False
        mock_args.run_ttm_fundamental = False
        mock_args.run_daily_ticks = False
        mock_args.run_minute_ticks = True
        mock_args.run_top_3000 = False
        mock_args.run_sentiment = False
        mock_args.run_all = False
        mock_args.alpaca_start_year = 2025
        mock_args.minute_start_year = 2017
        mock_args.daily_chunk_size = 200
        mock_args.daily_sleep_time = 0.2
        mock_args.max_workers = 50
        mock_args.minute_workers = 50
        mock_args.minute_chunk_size = 500
        mock_args.minute_sleep_time = 0.0
        mock_parse_args.return_value = mock_args

        mock_app = Mock()
        mock_upload_app.return_value = mock_app

        main()

        call_kwargs = mock_app.run.call_args[1]
        assert call_kwargs['resume'] is True
