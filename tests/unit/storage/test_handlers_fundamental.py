"""Unit tests for FundamentalHandler, TTMHandler, DerivedHandler."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import logging
import polars as pl

from quantdl.storage.handlers.fundamental import (
    FundamentalHandler,
    TTMHandler,
    DerivedHandler,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_deps():
    """Common mocked dependencies for fundamental handlers."""
    return {
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'cik_resolver': Mock(),
        'universe_manager': Mock(),
        'validator': Mock(),
        'sec_rate_limiter': Mock(),
        'logger': Mock(spec=logging.Logger),
    }


# ── FundamentalHandler ───────────────────────────────────────────────────────


class TestFundamentalHandler:

    @pytest.fixture
    def handler(self, mock_deps):
        return FundamentalHandler(**mock_deps)

    def test_init(self, handler, mock_deps):
        assert handler.publishers is mock_deps['data_publishers']
        assert handler.collectors is mock_deps['data_collectors']
        assert handler.cik_resolver is mock_deps['cik_resolver']
        assert handler.universe_manager is mock_deps['universe_manager']
        assert handler.validator is mock_deps['validator']
        assert handler.sec_rate_limiter is mock_deps['sec_rate_limiter']
        assert handler.stats == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def test_upload_no_symbols_returns_early(self, handler):
        with patch.object(handler, '_prepare_symbols', return_value=([], {}, 0.1)):
            result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 0
        assert result['failed'] == 0
        handler.logger.warning.assert_called_once()

    @patch('quantdl.storage.handlers.fundamental.tqdm')
    @patch('quantdl.storage.handlers.fundamental.ThreadPoolExecutor')
    def test_upload_processes_symbols(self, mock_executor_cls, mock_tqdm, handler):
        with patch.object(handler, '_prepare_symbols', return_value=(['AAPL'], {'AAPL': '123'}, 0.1)):
            mock_future = Mock()
            mock_future.result.return_value = {'status': 'success'}
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = mock_future
            mock_executor_cls.return_value = mock_executor

            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = Mock(return_value=iter([mock_future]))

            result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 1

    def test_process_symbol_with_cik(self, handler, mock_deps):
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '0000320193')

        mock_deps['data_publishers'].publish_fundamental.assert_called_once_with(
            sym='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            cik='0000320193',
            sec_rate_limiter=mock_deps['sec_rate_limiter'],
        )
        assert result['status'] == 'success'

    def test_process_symbol_without_cik(self, handler, mock_deps):
        mock_deps['cik_resolver'].get_cik.return_value = '0000320193'
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, None)

        mock_deps['cik_resolver'].get_cik.assert_called_once_with('AAPL', '2020-06-30', year=2020)
        assert result['status'] == 'success'

    def test_process_symbol_skips_existing(self, handler, mock_deps):
        mock_deps['validator'].data_exists.return_value = True
        mock_deps['data_publishers'].publish_fundamental.return_value = {'status': 'success'}

        handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        mock_deps['validator'].data_exists.assert_called_once()
        handler.logger.debug.assert_called()

    def test_prepare_symbols_builds_list(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.side_effect = [
            ['AAPL', 'MSFT'],
            ['AAPL', 'GOOGL'],
        ]
        mock_deps['cik_resolver'].batch_prefetch_ciks.side_effect = [
            {'AAPL': '1', 'MSFT': '2'},
            {'GOOGL': '3'},
        ]

        symbols, cik_map, _ = handler._prepare_symbols('2020-01-01', '2021-12-31')

        # AAPL appears in both years but should be deduplicated
        assert set(symbols) == {'AAPL', 'MSFT', 'GOOGL'}
        assert cik_map == {'AAPL': '1', 'MSFT': '2', 'GOOGL': '3'}

    def test_prepare_symbols_filters_without_cik(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL', 'NOCIK']
        mock_deps['cik_resolver'].batch_prefetch_ciks.return_value = {
            'AAPL': '1',
            'NOCIK': None,
        }

        symbols, _, _ = handler._prepare_symbols('2020-01-01', '2020-12-31')

        assert 'AAPL' in symbols
        assert 'NOCIK' not in symbols

    def test_prepare_symbols_logs_non_sec_filers(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL', 'XYZ']
        mock_deps['cik_resolver'].batch_prefetch_ciks.return_value = {
            'AAPL': '1',
            'XYZ': None,
        }

        handler._prepare_symbols('2020-01-01', '2020-12-31')

        # <=30 non-filers → logged individually
        info_calls = [str(c) for c in handler.logger.info.call_args_list]
        assert any('Non-SEC filers' in c for c in info_calls)

    def test_update_stats_all_statuses(self, handler):
        handler._update_stats({'status': 'success'})
        handler._update_stats({'status': 'canceled'})
        handler._update_stats({'status': 'skipped'})
        handler._update_stats({'status': 'failed'})
        handler._update_stats({'status': 'unknown'})  # defaults to failed
        handler._update_stats({})  # no status → failed

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 3}

    def test_build_result(self, handler):
        import time
        handler.stats = {'success': 5, 'failed': 1, 'skipped': 2, 'canceled': 0}
        start = time.time() - 10

        result = handler._build_result(start, 1.0, 5.0, 10)

        assert result['success'] == 5
        assert result['failed'] == 1
        assert result['prefetch_time'] == 1.0
        assert result['fetch_time'] == 5.0
        assert result['avg_rate'] == 2.0  # 10 / 5.0

    def test_build_result_zero_fetch_time(self, handler):
        import time
        start = time.time()

        result = handler._build_result(start, 0.0, 0.0, 0)

        assert result['avg_rate'] == 0


# ── TTMHandler ───────────────────────────────────────────────────────────────


class TestTTMHandler:

    @pytest.fixture
    def handler(self, mock_deps):
        return TTMHandler(**mock_deps)

    def test_init(self, handler, mock_deps):
        assert handler.publishers is mock_deps['data_publishers']
        assert handler.cik_resolver is mock_deps['cik_resolver']

    def test_upload_no_symbols(self, handler):
        with patch.object(handler, '_prepare_symbols', return_value=([], {}, 0.1)):
            result = handler.upload('2020-01-01', '2020-12-31')

        assert result == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}
        handler.logger.warning.assert_called_once()

    @patch('quantdl.storage.handlers.fundamental.tqdm')
    @patch('quantdl.storage.handlers.fundamental.ThreadPoolExecutor')
    def test_upload_processes_symbols(self, mock_executor_cls, mock_tqdm, handler):
        with patch.object(handler, '_prepare_symbols', return_value=(['AAPL'], {'AAPL': '1'}, 0.1)):
            mock_future = Mock()
            mock_future.result.return_value = {'status': 'success'}
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = mock_future
            mock_executor_cls.return_value = mock_executor

            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = Mock(return_value=iter([mock_future]))

            result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 1

    def test_process_symbol_with_cik(self, handler, mock_deps):
        mock_deps['data_publishers'].publish_ttm_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        mock_deps['data_publishers'].publish_ttm_fundamental.assert_called_once()
        assert result['status'] == 'success'

    def test_process_symbol_without_cik(self, handler, mock_deps):
        mock_deps['cik_resolver'].get_cik.return_value = '123'
        mock_deps['data_publishers'].publish_ttm_fundamental.return_value = {'status': 'success'}

        handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, None)

        mock_deps['cik_resolver'].get_cik.assert_called_once_with('AAPL', '2020-06-30', year=2020)

    def test_prepare_symbols(self, handler, mock_deps):
        mock_deps['universe_manager'].load_symbols_for_year.side_effect = [
            ['AAPL', 'MSFT'],
            ['AAPL', 'GOOGL'],
        ]
        mock_deps['cik_resolver'].batch_prefetch_ciks.side_effect = [
            {'AAPL': '1', 'MSFT': '2'},
            {'GOOGL': '3'},
        ]

        symbols, cik_map, _ = handler._prepare_symbols('2020-01-01', '2021-12-31')

        assert set(symbols) == {'AAPL', 'MSFT', 'GOOGL'}

    def test_update_stats_all_statuses(self, handler):
        handler._update_stats({'status': 'success'})
        handler._update_stats({'status': 'canceled'})
        handler._update_stats({'status': 'skipped'})
        handler._update_stats({'status': 'failed'})
        handler._update_stats({})

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 2}


# ── DerivedHandler ───────────────────────────────────────────────────────────


class TestDerivedHandler:

    @pytest.fixture
    def deps(self, mock_deps):
        """DerivedHandler doesn't take sec_rate_limiter."""
        d = {k: v for k, v in mock_deps.items() if k != 'sec_rate_limiter'}
        return d

    @pytest.fixture
    def handler(self, deps):
        return DerivedHandler(**deps)

    def test_init(self, handler, deps):
        assert handler.publishers is deps['data_publishers']
        assert handler.cik_resolver is deps['cik_resolver']

    def test_upload_no_symbols(self, handler):
        with patch.object(handler, '_prepare_symbols', return_value=([], {}, 0.1)):
            result = handler.upload('2020-01-01', '2020-12-31')

        assert result == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    @patch('quantdl.storage.handlers.fundamental.tqdm')
    @patch('quantdl.storage.handlers.fundamental.ThreadPoolExecutor')
    def test_upload_processes_symbols(self, mock_executor_cls, mock_tqdm, handler):
        with patch.object(handler, '_prepare_symbols', return_value=(['AAPL'], {'AAPL': '1'}, 0.1)):
            mock_future = Mock()
            mock_future.result.return_value = {'status': 'success'}
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = mock_future
            mock_executor_cls.return_value = mock_executor

            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            mock_pbar.__iter__ = Mock(return_value=iter([mock_future]))

            result = handler.upload('2020-01-01', '2020-12-31')

        assert result['success'] == 1

    def test_process_symbol_with_cik(self, handler, deps):
        deps['data_collectors'].collect_derived_long.return_value = (
            pl.DataFrame({'a': [1]}), None
        )
        deps['data_publishers'].publish_derived_fundamental.return_value = {'status': 'success'}

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        deps['data_collectors'].collect_derived_long.assert_called_once()
        assert result['status'] == 'success'

    def test_process_symbol_without_cik_resolves(self, handler, deps):
        deps['cik_resolver'].get_cik.return_value = '123'
        deps['data_collectors'].collect_derived_long.return_value = (
            pl.DataFrame({'a': [1]}), None
        )
        deps['data_publishers'].publish_derived_fundamental.return_value = {'status': 'success'}

        handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, None)

        deps['cik_resolver'].get_cik.assert_called_once()

    def test_process_symbol_cik_none_after_resolve(self, handler, deps):
        """Second None check at line 433."""
        deps['cik_resolver'].get_cik.return_value = None

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, None)

        assert result['status'] == 'skipped'
        assert 'No CIK' in result['error']

    def test_process_symbol_empty_derived_df(self, handler, deps):
        deps['data_collectors'].collect_derived_long.return_value = (
            pl.DataFrame(), 'No data'
        )

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        assert result['status'] == 'skipped'
        assert result['error'] == 'No data'

    def test_process_symbol_empty_derived_df_no_reason(self, handler, deps):
        deps['data_collectors'].collect_derived_long.return_value = (
            pl.DataFrame(), None
        )

        result = handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        assert result['status'] == 'skipped'
        assert result['error'] == 'No derived data'

    def test_process_symbol_skips_existing(self, handler, deps):
        deps['validator'].data_exists.return_value = True
        deps['data_collectors'].collect_derived_long.return_value = (
            pl.DataFrame({'a': [1]}), None
        )
        deps['data_publishers'].publish_derived_fundamental.return_value = {'status': 'success'}

        handler._process_symbol('AAPL', '2020-01-01', '2020-12-31', False, '123')

        deps['validator'].data_exists.assert_called_once()
        handler.logger.debug.assert_called()

    def test_prepare_symbols_empty_year(self, handler, deps):
        deps['universe_manager'].load_symbols_for_year.side_effect = [
            [],        # 2020: no symbols
            ['AAPL'],  # 2021: one symbol
        ]
        deps['cik_resolver'].batch_prefetch_ciks.return_value = {'AAPL': '1'}

        symbols, _, _ = handler._prepare_symbols('2020-01-01', '2021-12-31')

        handler.logger.warning.assert_called_once()
        assert 'AAPL' in symbols

    def test_update_stats_all_statuses(self, handler):
        handler._update_stats({'status': 'success'})
        handler._update_stats({'status': 'canceled'})
        handler._update_stats({'status': 'skipped'})
        handler._update_stats({'status': 'failed'})
        handler._update_stats({})

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 2}
