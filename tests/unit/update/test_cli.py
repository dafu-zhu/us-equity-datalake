import datetime as dt
from unittest.mock import Mock, patch
import os


class TestUpdateCLI:
    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--date', '2024-06-03'])
    def test_main_with_date_argument(self, mock_app_class):
        """Test CLI with date argument (uses WRDS mode when credentials available)"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        mock_app_class.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] == dt.date(2024, 6, 3)
        assert call_args[1]['update_ticks'] is True
        assert call_args[1]['update_fundamentals'] is True
        assert call_args[1]['fundamental_lookback_days'] == 7

    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--no-ticks', '--no-fundamentals', '--lookback', '14'])
    def test_main_with_flags(self, mock_app_class):
        """Test CLI with flags (uses WRDS mode when credentials available)"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] is None
        assert call_args[1]['update_ticks'] is False
        assert call_args[1]['update_fundamentals'] is False
        assert call_args[1]['fundamental_lookback_days'] == 14

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--date', '2024-06-03'])
    def test_main_auto_detects_wrds_free_mode(self, mock_app_class):
        """Test CLI auto-detects WRDS-free mode when credentials missing"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        # Should use WRDS-free app
        mock_app_class.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] == dt.date(2024, 6, 3)

    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds'])
    def test_main_with_no_wrds_flag(self, mock_app_class):
        """Test CLI respects --no-wrds flag even when credentials available"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        # Should use WRDS-free app despite credentials being available
        mock_app_class.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] is None  # Default to yesterday

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--no-ticks'])
    def test_main_wrds_free_with_flags(self, mock_app_class):
        """Test WRDS-free mode with additional flags"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        mock_app_class.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['update_ticks'] is False
        assert call_args[1]['update_fundamentals'] is True
