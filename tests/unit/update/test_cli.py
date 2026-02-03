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
        assert call_args[1]['update_daily_ticks'] is True
        assert call_args[1]['update_minute_ticks'] is True
        assert call_args[1]['update_fundamental'] is True
        assert call_args[1]['update_ttm'] is True
        assert call_args[1]['update_derived'] is True
        assert call_args[1]['fundamental_lookback_days'] == 7

    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--no-daily-ticks', '--no-minute-ticks', '--no-fundamental', '--no-ttm', '--no-derived', '--lookback', '14'])
    def test_main_with_flags(self, mock_app_class):
        """Test CLI with flags (uses WRDS mode when credentials available)"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        call_args = mock_app_instance.run_daily_update.call_args
        # Default to yesterday when no --date specified
        yesterday = dt.date.today() - dt.timedelta(days=1)
        assert call_args[1]['target_date'] == yesterday
        assert call_args[1]['update_daily_ticks'] is False
        assert call_args[1]['update_minute_ticks'] is False
        assert call_args[1]['update_fundamental'] is False
        assert call_args[1]['update_ttm'] is False
        assert call_args[1]['update_derived'] is False
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
        # Default to yesterday when no --date specified
        yesterday = dt.date.today() - dt.timedelta(days=1)
        assert call_args[1]['target_date'] == yesterday

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--no-daily-ticks', '--no-minute-ticks'])
    def test_main_wrds_free_with_flags(self, mock_app_class):
        """Test WRDS-free mode with additional flags"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        mock_app_class.assert_called_once()
        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['update_daily_ticks'] is False
        assert call_args[1]['update_minute_ticks'] is False
        assert call_args[1]['update_fundamental'] is True

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--no-ticks'])
    def test_main_no_ticks_shorthand(self, mock_app_class):
        """Test --no-ticks shorthand skips both daily and minute ticks"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        call_args = mock_app_instance.run_daily_update.call_args
        assert call_args[1]['update_daily_ticks'] is False
        assert call_args[1]['update_minute_ticks'] is False
        assert call_args[1]['update_fundamental'] is True

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--date', '2024-06-05', '--backfill-from', '2024-06-03'])
    def test_main_backfill_from(self, mock_app_class):
        """Test --backfill-from processes multiple days, fundamental at end"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        # Should be called 4 times: June 3, 4, 5 (ticks) + June 5 (fundamental)
        assert mock_app_instance.run_daily_update.call_count == 4

        # Check dates processed
        calls = mock_app_instance.run_daily_update.call_args_list
        dates = [call[1]['target_date'] for call in calls]
        assert dates == [
            dt.date(2024, 6, 3), dt.date(2024, 6, 4), dt.date(2024, 6, 5),  # ticks
            dt.date(2024, 6, 5)  # fundamental at end
        ]

        # Verify ticks calls skip fundamental
        for call in calls[:3]:
            assert call[1]['update_fundamental'] is False
            assert call[1]['update_ttm'] is False
            assert call[1]['update_derived'] is False

        # Verify final call is fundamental only with full lookback
        final_call = calls[3][1]
        assert final_call['update_daily_ticks'] is False
        assert final_call['update_minute_ticks'] is False
        assert final_call['update_fundamental'] is True
        assert final_call['fundamental_lookback_days'] == 3  # 3 days backfill

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--date', '2024-06-01', '--backfill-from', '2024-06-05'])
    def test_main_backfill_invalid_range(self, mock_app_class, capsys):
        """Test --backfill-from with end date before start date fails"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        # Should not call run_daily_update
        mock_app_instance.run_daily_update.assert_not_called()

        # Should print error
        captured = capsys.readouterr()
        assert 'Error' in captured.out
        assert 'must be before' in captured.out

    @patch.dict(os.environ, {}, clear=True)
    @patch('quantdl.update.app_no_wrds.DailyUpdateAppNoWRDS')
    @patch('sys.argv', ['cli.py', '--no-wrds', '--date', '2024-07-02', '--backfill-from', '2024-06-01'])
    def test_main_backfill_exceeds_max_days(self, mock_app_class, capsys):
        """Test --backfill-from with range > 30 days fails (June 1 to July 2 = 31 days)"""
        from quantdl.update.cli import main

        mock_app_instance = Mock()
        mock_app_class.return_value = mock_app_instance

        main()

        # Should not call run_daily_update
        mock_app_instance.run_daily_update.assert_not_called()

        # Should print error
        captured = capsys.readouterr()
        assert 'Error' in captured.out
        assert 'exceeds max' in captured.out
