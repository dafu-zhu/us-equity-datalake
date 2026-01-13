"""Tests for consolidate CLI."""
from unittest.mock import Mock, patch


class TestConsolidateCLI:
    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--year', '2024'])
    def test_main_with_year_argument(self, mock_app_class):
        """Test CLI with required year argument"""
        from quantdl.consolidate.cli import main

        mock_app_instance = Mock()
        mock_app_instance.consolidate_year.return_value = {
            'success': 100,
            'failed': 2,
            'skipped': 5
        }
        mock_app_class.return_value = mock_app_instance

        main()

        # Verify app initialized with default config
        mock_app_class.assert_called_once_with(config_path='configs/storage.yaml')

        # Verify consolidate_year called with correct args
        mock_app_instance.consolidate_year.assert_called_once_with(
            year=2024,
            force=False
        )

    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--year', '2025', '--force'])
    def test_main_with_force_flag(self, mock_app_class):
        """Test CLI with force flag enabled"""
        from quantdl.consolidate.cli import main

        mock_app_instance = Mock()
        mock_app_instance.consolidate_year.return_value = {
            'success': 50,
            'failed': 0,
            'skipped': 0
        }
        mock_app_class.return_value = mock_app_instance

        main()

        # Verify consolidate_year called with force=True
        mock_app_instance.consolidate_year.assert_called_once_with(
            year=2025,
            force=True
        )

    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--year', '2023', '--config', 'custom/config.yaml'])
    def test_main_with_custom_config(self, mock_app_class):
        """Test CLI with custom config path"""
        from quantdl.consolidate.cli import main

        mock_app_instance = Mock()
        mock_app_instance.consolidate_year.return_value = {
            'success': 10,
            'failed': 0,
            'skipped': 1
        }
        mock_app_class.return_value = mock_app_instance

        main()

        # Verify app initialized with custom config
        mock_app_class.assert_called_once_with(config_path='custom/config.yaml')

        # Verify consolidate_year called with correct year
        mock_app_instance.consolidate_year.assert_called_once_with(
            year=2023,
            force=False
        )

    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--year', '2024', '--force', '--config', 'test.yaml'])
    def test_main_with_all_arguments(self, mock_app_class):
        """Test CLI with all arguments provided"""
        from quantdl.consolidate.cli import main

        mock_app_instance = Mock()
        mock_app_instance.consolidate_year.return_value = {
            'success': 1000,
            'failed': 10,
            'skipped': 50
        }
        mock_app_class.return_value = mock_app_instance

        main()

        # Verify app initialized with custom config
        mock_app_class.assert_called_once_with(config_path='test.yaml')

        # Verify consolidate_year called with all args
        mock_app_instance.consolidate_year.assert_called_once_with(
            year=2024,
            force=True
        )

    @patch('builtins.print')
    @patch('quantdl.update.app.DailyUpdateApp')
    @patch('sys.argv', ['cli.py', '--year', '2024'])
    def test_main_prints_stats(self, mock_app_class, mock_print):
        """Test CLI prints consolidation statistics"""
        from quantdl.consolidate.cli import main

        mock_app_instance = Mock()
        mock_app_instance.consolidate_year.return_value = {
            'success': 100,
            'failed': 2,
            'skipped': 5
        }
        mock_app_class.return_value = mock_app_instance

        main()

        # Verify print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]

        # Check consolidation start message
        assert any('Consolidating year 2024' in call for call in print_calls)
        assert any('force=False' in call for call in print_calls)

        # Check stats output
        assert any('Consolidation completed' in call for call in print_calls)
        assert any('Success: 100' in call for call in print_calls)
        assert any('Failed: 2' in call for call in print_calls)
        assert any('Skipped: 5' in call for call in print_calls)
