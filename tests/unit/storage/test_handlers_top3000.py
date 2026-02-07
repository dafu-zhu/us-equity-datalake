"""Unit tests for Top3000Handler."""

import datetime as dt

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

from quantdl.storage.handlers.top3000 import Top3000Handler


@pytest.fixture
def deps():
    return {
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'universe_manager': Mock(),
        'validator': Mock(),
        'calendar': Mock(),
        'logger': Mock(spec=logging.Logger),
        'alpaca_start_year': 2025,
    }


@pytest.fixture
def handler(deps):
    return Top3000Handler(**deps)


class TestTop3000Handler:

    def test_init(self, handler, deps):
        assert handler.publishers is deps['data_publishers']
        assert handler.collectors is deps['data_collectors']
        assert handler.universe_manager is deps['universe_manager']
        assert handler.validator is deps['validator']
        assert handler.calendar is deps['calendar']
        assert handler.alpaca_start_year == 2025

    def test_upload_year_no_symbols(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = []

        result = handler.upload_year(2020)

        deps['logger'].warning.assert_called_once()
        assert result == handler.stats

    def test_upload_year_skips_existing(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = True

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020, overwrite=False)

        assert result['skipped'] == 12  # All 12 months skipped

    def test_upload_year_no_trading_days(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = False
        deps['calendar'].load_trading_days.return_value = []

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020)

        assert result['skipped'] == 12
        deps['logger'].warning.assert_any_call('No trading days for 2020-01')

    def test_upload_year_stops_at_future(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = False

        # Only month 1 has data, month 2 is in the future
        def load_trading_days(year, month):
            if month == 1:
                return ['2025-01-31']
            elif month == 2:
                return ['2025-02-28']
            return []

        deps['calendar'].load_trading_days.side_effect = load_trading_days
        deps['universe_manager'].get_top_3000.return_value = ['AAPL']
        deps['data_publishers'].publish_top_3000.return_value = {'status': 'success'}

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 1, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2025)

        assert result['success'] == 1
        # Should have stopped before month 2's future date

    def test_upload_year_success(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL', 'MSFT']
        deps['validator'].top_3000_exists.return_value = False
        deps['calendar'].load_trading_days.return_value = ['2020-01-31']
        deps['universe_manager'].get_top_3000.return_value = ['AAPL', 'MSFT']
        deps['data_publishers'].publish_top_3000.return_value = {'status': 'success'}

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020)

        assert result['success'] == 12

    def test_upload_year_skipped_result(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = False
        deps['calendar'].load_trading_days.return_value = ['2020-01-31']
        deps['universe_manager'].get_top_3000.return_value = ['AAPL']
        deps['data_publishers'].publish_top_3000.return_value = {'status': 'skipped', 'error': 'No data'}

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020)

        assert result['skipped'] == 12

    def test_upload_year_failed_result(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = False
        deps['calendar'].load_trading_days.return_value = ['2020-01-31']
        deps['universe_manager'].get_top_3000.return_value = ['AAPL']
        deps['data_publishers'].publish_top_3000.return_value = {'status': 'failed', 'error': 'S3 err'}

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020)

        assert result['failed'] == 12

    def test_upload_year_crsp_source(self, handler, deps):
        """Year < alpaca_start_year → source='crsp'."""
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = True

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            handler.upload_year(2020)

        info_calls = [str(c) for c in deps['logger'].info.call_args_list]
        assert any('source=crsp' in c for c in info_calls)

    def test_upload_year_overwrite(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']
        deps['validator'].top_3000_exists.return_value = True  # exists but overwrite=True
        deps['calendar'].load_trading_days.return_value = ['2020-01-31']
        deps['universe_manager'].get_top_3000.return_value = ['AAPL']
        deps['data_publishers'].publish_top_3000.return_value = {'status': 'success'}

        with patch('quantdl.storage.handlers.top3000.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2020, 12, 31)
            mock_dt.datetime = dt.datetime
            result = handler.upload_year(2020, overwrite=True)

        assert result['success'] == 12
        # validator.top_3000_exists should not have been checked
        deps['validator'].top_3000_exists.assert_not_called()
