import datetime as dt
from unittest.mock import Mock, patch


class TestUpdateCLI:
    @patch('quantdl.update.cli.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--date', '2024-06-03'])
    def test_main_with_date_argument(self, mock_app_class):
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

    @patch('quantdl.update.cli.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--no-ticks', '--no-fundamentals', '--lookback', '14'])
    def test_main_with_flags(self, mock_app_class):
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['target_date'] is None
        assert call_args[1]['update_ticks'] is False
        assert call_args[1]['update_fundamentals'] is False
        assert call_args[1]['fundamental_lookback_days'] == 14
