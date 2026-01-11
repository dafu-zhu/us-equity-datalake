"""
Unit tests for cli module
Tests command-line interface functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from quantdl.cli import main


class TestCLI:
    """Test CLI main function"""

    @patch('quantdl.cli.UploadApp')
    def test_main_creates_app(self, mock_app_class):
        """Test that main creates an UploadApp instance"""
        mock_app = Mock()
        mock_app_class.return_value = mock_app

        main()

        # Verify UploadApp was instantiated
        mock_app_class.assert_called_once()

    @patch('quantdl.cli.UploadApp')
    def test_main_runs_with_correct_parameters(self, mock_app_class):
        """Test that main runs UploadApp with correct parameters"""
        mock_app = Mock()
        mock_app_class.return_value = mock_app

        main()

        # Verify run was called with expected parameters
        mock_app.run.assert_called_once_with(
            start_year=2009,
            end_year=2025,
            max_workers=50,
            sleep_time=0.03,
            overwrite=True,
            daily_chunk_size=200,
            daily_sleep_time=0.2,
            minute_ticks_start_year=2017,
            run_daily_ticks=True,
            run_minute_ticks=False,
            run_fundamental=False,
            run_derived_fundamental=False,
            run_ttm_fundamental=False,
        )

    @patch('quantdl.cli.UploadApp')
    def test_main_closes_app_on_success(self, mock_app_class):
        """Test that main closes app after successful run"""
        mock_app = Mock()
        mock_app_class.return_value = mock_app

        main()

        # Verify close was called
        mock_app.close.assert_called_once()

    @patch('quantdl.cli.UploadApp')
    def test_main_closes_app_on_exception(self, mock_app_class):
        """Test that main closes app even if run raises exception"""
        mock_app = Mock()
        mock_app.run.side_effect = RuntimeError("Test error")
        mock_app_class.return_value = mock_app

        with pytest.raises(RuntimeError, match="Test error"):
            main()

        # Verify close was still called despite exception
        mock_app.close.assert_called_once()

    @patch('quantdl.cli.UploadApp')
    def test_main_closes_app_on_keyboard_interrupt(self, mock_app_class):
        """Test that main closes app on keyboard interrupt"""
        mock_app = Mock()
        mock_app.run.side_effect = KeyboardInterrupt()
        mock_app_class.return_value = mock_app

        with pytest.raises(KeyboardInterrupt):
            main()

        # Verify close was called
        mock_app.close.assert_called_once()
