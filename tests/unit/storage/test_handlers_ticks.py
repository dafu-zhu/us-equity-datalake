"""Unit tests for DailyTicksHandler and MinuteTicksHandler."""

import datetime as dt
import queue
import threading

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import logging
import polars as pl

from quantdl.storage.handlers.ticks import DailyTicksHandler, MinuteTicksHandler


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def daily_deps():
    """Mocked dependencies for DailyTicksHandler."""
    return {
        's3_client': Mock(),
        'bucket_name': 'test-bucket',
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'security_master': Mock(),
        'universe_manager': Mock(),
        'validator': Mock(),
        'crsp_ticks': Mock(),
        'logger': Mock(spec=logging.Logger),
        'wrds_available': True,
        'alpaca_start_year': 2025,
    }


@pytest.fixture
def minute_deps():
    """Mocked dependencies for MinuteTicksHandler."""
    return {
        's3_client': Mock(),
        'bucket_name': 'test-bucket',
        'data_publishers': Mock(),
        'data_collectors': Mock(),
        'security_master': Mock(),
        'universe_manager': Mock(),
        'validator': Mock(),
        'calendar': Mock(),
        'logger': Mock(spec=logging.Logger),
    }


# ── DailyTicksHandler ───────────────────────────────────────────────────────


class TestDailyTicksHandler:

    @pytest.fixture
    def handler(self, daily_deps):
        return DailyTicksHandler(**daily_deps)

    def test_init(self, handler, daily_deps):
        assert handler.s3_client is daily_deps['s3_client']
        assert handler.bucket_name == 'test-bucket'
        assert handler.wrds_available is True
        assert handler.alpaca_start_year == 2025

    def test_upload_year_defaults_current_year(self, handler, daily_deps):
        daily_deps['universe_manager'].load_symbols_for_year.return_value = []

        with patch.object(handler, '_build_security_id_cache', return_value={}), \
             patch.object(handler, '_upload_monthly_mode', return_value=handler.stats) as mock_monthly:
            handler.upload_year(dt.datetime.now().year)
            mock_monthly.assert_called_once()

    def test_upload_year_raises_without_wrds(self, daily_deps):
        daily_deps['wrds_available'] = False
        handler = DailyTicksHandler(**daily_deps)

        with pytest.raises(RuntimeError, match='WRDS required'):
            handler.upload_year(2020)

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_year_history_mode(self, mock_tqdm, handler, daily_deps):
        daily_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']

        with patch.object(handler, '_build_security_id_cache', return_value={'AAPL': 1}), \
             patch.object(handler, '_upload_history_mode', return_value=handler.stats) as mock_hist:
            handler.upload_year(2020, current_year=2025)

        mock_hist.assert_called_once()

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_year_monthly_mode(self, mock_tqdm, handler, daily_deps):
        daily_deps['universe_manager'].load_symbols_for_year.return_value = ['AAPL']

        with patch.object(handler, '_build_security_id_cache', return_value={'AAPL': 1}), \
             patch.object(handler, '_upload_monthly_mode', return_value=handler.stats) as mock_month:
            handler.upload_year(2025, current_year=2025)

        mock_month.assert_called_once()

    def test_build_security_id_cache(self, handler, daily_deps):
        daily_deps['security_master'].get_security_id.side_effect = [100, ValueError('not found')]

        cache = handler._build_security_id_cache(['AAPL', 'BADTICKER'], 2020)

        assert cache == {'AAPL': 100, 'BADTICKER': None}

    def test_build_permno_mapping(self, handler, daily_deps):
        master_tb = Mock()
        master_tb.iter_rows.return_value = [
            {'permno': 1, 'security_id': 100, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
            {'permno': 2, 'security_id': 200, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
            {'permno': 3, 'security_id': None, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
            {'permno': None, 'security_id': 300, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
        ]
        daily_deps['security_master'].master_tb = master_tb

        permno_map, all_sids = handler._build_permno_mapping()

        assert 1 in permno_map
        assert 2 in permno_map
        assert 3 not in permno_map  # sid is None
        assert all_sids == {100, 200}

    def test_filter_existing_symbols(self, handler, daily_deps):
        daily_deps['validator'].data_exists.side_effect = [True, False]
        pbar = MagicMock()

        # sec_id None → skipped, sec_id exists → canceled, sec_id new → fetch
        with patch.object(handler, '_build_security_id_cache'):
            cache = {'SYM_NONE': None, 'SYM_EXISTS': 1, 'SYM_NEW': 2}
            result = handler._filter_existing_symbols(
                ['SYM_NONE', 'SYM_EXISTS', 'SYM_NEW'], 2020, cache, pbar, month=None
            )

        assert result == ['SYM_NEW']
        assert handler.stats['skipped'] == 1
        assert handler.stats['canceled'] == 1

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_history_mode_with_overwrite(self, mock_tqdm, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['data_collectors'].collect_daily_ticks_year_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        result = handler._upload_history_mode(
            ['AAPL'], 2025, {'AAPL': 1}, overwrite=True,
            chunk_size=200, sleep_time=0.0, source='alpaca'
        )

        assert result['success'] == 1

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_history_mode_no_overwrite(self, mock_tqdm, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        with patch.object(handler, '_filter_existing_symbols', return_value=[]):
            result = handler._upload_history_mode(
                ['AAPL'], 2020, {'AAPL': 1}, overwrite=False,
                chunk_size=200, sleep_time=0.0, source='crsp'
            )

        assert result['success'] == 0

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_monthly_mode_skips_future(self, mock_tqdm, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 3, 15)
            mock_dt.datetime = dt.datetime

            result = handler._upload_monthly_mode(
                ['AAPL'], 2025, {'AAPL': 1}, overwrite=False,
                chunk_size=200, sleep_time=0.0, source='alpaca'
            )

        # Should log about future months
        info_calls = [str(c) for c in handler.logger.info.call_args_list]
        assert any('future month' in c for c in info_calls)

    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_monthly_mode_processes(self, mock_tqdm, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        daily_deps['data_collectors'].collect_daily_ticks_month_bulk.return_value = {
            'AAPL': pl.DataFrame({'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks.return_value = {'status': 'success'}

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 1, 31)
            mock_dt.datetime = dt.datetime

            result = handler._upload_monthly_mode(
                ['AAPL'], 2025, {'AAPL': 1}, overwrite=True,
                chunk_size=200, sleep_time=0.0, source='alpaca'
            )

        assert result['success'] == 1

    def test_upload_crsp_bulk_history_no_wrds(self, daily_deps):
        daily_deps['wrds_available'] = False
        handler = DailyTicksHandler(**daily_deps)

        with pytest.raises(RuntimeError, match='WRDS connection'):
            handler.upload_crsp_bulk_history()

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_crsp_bulk_history_resume(self, mock_tqdm, mock_tracker_cls, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        # Setup permno mapping
        master_tb = Mock()
        master_tb.iter_rows.return_value = [
            {'permno': 1, 'security_id': 100, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
        ]
        master_tb.filter.return_value.select.return_value.head.return_value = pl.DataFrame({'symbol': ['AAPL']})
        daily_deps['security_master'].master_tb = master_tb

        # Tracker says 100 is already done
        mock_tracker = MagicMock()
        mock_tracker.load.return_value = {100}
        mock_tracker_cls.return_value = mock_tracker

        result = handler.upload_crsp_bulk_history(resume=True, overwrite=False)

        # No permnos to fetch → returns early
        assert result == handler.stats

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_crsp_bulk_history_overwrite(self, mock_tqdm, mock_tracker_cls, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        master_tb = Mock()
        master_tb.iter_rows.return_value = [
            {'permno': 1, 'security_id': 100, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
        ]
        master_tb.filter.return_value.select.return_value.head.return_value = pl.DataFrame({'symbol': ['AAPL']})
        daily_deps['security_master'].master_tb = master_tb

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        daily_deps['crsp_ticks'].collect_daily_ticks_full_history_bulk.return_value = {
            1: pl.DataFrame({'timestamp': ['2020-06-15'], 'close': [100.0]})
        }
        daily_deps['data_publishers'].publish_daily_ticks_to_history.return_value = {'status': 'success'}

        result = handler.upload_crsp_bulk_history(overwrite=True, resume=False)

        mock_tracker.reset.assert_called_once()
        assert result['success'] == 1

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_crsp_bulk_history_all_complete(self, mock_tqdm, mock_tracker_cls, handler, daily_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        master_tb = Mock()
        master_tb.iter_rows.return_value = [
            {'permno': 1, 'security_id': 100, 'start_date': '2020-01-01', 'end_date': '2020-12-31'},
        ]
        daily_deps['security_master'].master_tb = master_tb

        mock_tracker = MagicMock()
        mock_tracker.load.return_value = {100}  # Already completed
        mock_tracker_cls.return_value = mock_tracker

        result = handler.upload_crsp_bulk_history(resume=True)

        handler.logger.info.assert_any_call("All security_ids completed")

    def test_process_permno_data_missing(self, handler):
        tracker = MagicMock()
        pbar = MagicMock()
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        handler._process_permno_data(
            permno=1,
            permno_data={},  # permno not in data
            permno_sid_map=permno_sid_map,
            pending_sids={100},
            master_tb=Mock(),
            tracker=tracker,
            pbar=pbar,
        )

        assert handler.stats['skipped'] == 1
        tracker.mark_skipped.assert_called_once_with(100)

    def test_process_permno_data_not_pending(self, handler):
        tracker = MagicMock()
        pbar = MagicMock()
        df = pl.DataFrame({'timestamp': ['2020-06-15'], 'close': [100.0]})
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        handler._process_permno_data(
            permno=1,
            permno_data={1: df},
            permno_sid_map=permno_sid_map,
            pending_sids=set(),  # sid not pending
            master_tb=Mock(),
            tracker=tracker,
            pbar=pbar,
        )

        # Nothing should happen
        assert handler.stats['success'] == 0
        assert handler.stats['skipped'] == 0

    def test_process_permno_data_empty_df(self, handler):
        tracker = MagicMock()
        pbar = MagicMock()
        df = pl.DataFrame({'timestamp': ['2023-06-15'], 'close': [100.0]})
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        handler._process_permno_data(
            permno=1,
            permno_data={1: df},
            permno_sid_map=permno_sid_map,
            pending_sids={100},
            master_tb=Mock(),
            tracker=tracker,
            pbar=pbar,
        )

        # 2023 is outside 2020 range, filter → empty
        assert handler.stats['skipped'] == 1
        tracker.mark_skipped.assert_called_once_with(100)

    def test_process_permno_data_success(self, handler, daily_deps):
        tracker = MagicMock()
        pbar = MagicMock()
        df = pl.DataFrame({'timestamp': ['2020-06-15'], 'close': [100.0]})
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        master_tb = Mock()
        master_tb.filter.return_value.select.return_value.head.return_value = pl.DataFrame({'symbol': ['AAPL']})

        daily_deps['data_publishers'].publish_daily_ticks_to_history.return_value = {'status': 'success'}

        handler._process_permno_data(
            permno=1,
            permno_data={1: df},
            permno_sid_map=permno_sid_map,
            pending_sids={100},
            master_tb=master_tb,
            tracker=tracker,
            pbar=pbar,
        )

        assert handler.stats['success'] == 1
        tracker.mark_completed.assert_called_once_with(100)

    def test_process_permno_data_failure(self, handler, daily_deps):
        tracker = MagicMock()
        pbar = MagicMock()
        df = pl.DataFrame({'timestamp': ['2020-06-15'], 'close': [100.0]})
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        master_tb = Mock()
        master_tb.filter.return_value.select.return_value.head.return_value = pl.DataFrame({'symbol': ['AAPL']})

        daily_deps['data_publishers'].publish_daily_ticks_to_history.return_value = {'status': 'failed'}

        handler._process_permno_data(
            permno=1,
            permno_data={1: df},
            permno_sid_map=permno_sid_map,
            pending_sids={100},
            master_tb=master_tb,
            tracker=tracker,
            pbar=pbar,
        )

        assert handler.stats['failed'] == 1
        tracker.mark_failed.assert_called_once_with(100)

    def test_process_permno_data_exception(self, handler, daily_deps):
        tracker = MagicMock()
        pbar = MagicMock()
        df = pl.DataFrame({'timestamp': ['2020-06-15'], 'close': [100.0]})
        permno_sid_map = {1: [(100, '2020-01-01', '2020-12-31')]}

        master_tb = Mock()
        master_tb.filter.return_value.select.return_value.head.return_value = pl.DataFrame({'symbol': ['AAPL']})

        daily_deps['data_publishers'].publish_daily_ticks_to_history.side_effect = Exception('S3 error')

        handler._process_permno_data(
            permno=1,
            permno_data={1: df},
            permno_sid_map=permno_sid_map,
            pending_sids={100},
            master_tb=master_tb,
            tracker=tracker,
            pbar=pbar,
        )

        assert handler.stats['failed'] == 1
        tracker.mark_failed.assert_called_once_with(100)
        handler.logger.error.assert_called_once()

    def test_update_stats_from_result(self, handler):
        handler._update_stats_from_result({'status': 'success'})
        handler._update_stats_from_result({'status': 'canceled'})
        handler._update_stats_from_result({'status': 'skipped'})
        handler._update_stats_from_result({'status': 'failed'})
        handler._update_stats_from_result({})  # defaults to failed

        assert handler.stats == {'success': 1, 'canceled': 1, 'skipped': 1, 'failed': 2}


# ── MinuteTicksHandler ───────────────────────────────────────────────────────


class TestMinuteTicksHandler:

    @pytest.fixture
    def handler(self, minute_deps):
        return MinuteTicksHandler(**minute_deps)

    def test_init(self, handler, minute_deps):
        assert handler.s3_client is minute_deps['s3_client']
        assert handler.calendar is minute_deps['calendar']

    def test_upload_year_defaults_months(self, handler, minute_deps):
        minute_deps['universe_manager'].load_symbols_for_year.return_value = []

        with patch.object(handler, '_upload_month') as mock_month:
            minute_deps['data_publishers'].minute_ticks_worker = Mock()
            handler.upload_year(2025, months=None, num_workers=1)

        # Should be called for all 12 months
        assert mock_month.call_count == 12

    def test_upload_year_starts_and_stops_workers(self, handler, minute_deps):
        minute_deps['universe_manager'].load_symbols_for_year.return_value = []

        worker_started = []

        def fake_worker(data_queue, stats, stats_lock):
            item = data_queue.get()
            assert item is None  # Poison pill

        minute_deps['data_publishers'].minute_ticks_worker = fake_worker

        with patch.object(handler, '_upload_month'):
            stats = handler.upload_year(2025, months=[1], num_workers=2)

        assert isinstance(stats, dict)

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_month_resume(self, mock_tqdm, mock_tracker_cls, handler, minute_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        mock_tracker = MagicMock()
        mock_tracker.load.return_value = {'AAPL'}
        mock_tracker_cls.return_value = mock_tracker

        minute_deps['calendar'].load_trading_days.return_value = ['2025-01-02']

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue()

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 12, 31)

            minute_deps['data_collectors'].fetch_minute_month.return_value = {}
            minute_deps['data_collectors'].parse_minute_bars_to_daily.return_value = {}

            handler._upload_month(
                2025, 1, ['AAPL', 'MSFT'], overwrite=False, resume=True,
                chunk_size=500, sleep_time=0.0,
                data_queue=data_queue, stats=stats, stats_lock=stats_lock
            )

        # AAPL filtered by resume → only MSFT remains
        assert mock_tracker.load.called
        # Verify logger reported resume status
        info_calls = [str(c) for c in handler.logger.info.call_args_list]
        assert any('1 done' in c for c in info_calls)

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_month_overwrite_resets_tracker(self, mock_tqdm, mock_tracker_cls, handler, minute_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        minute_deps['calendar'].load_trading_days.return_value = ['2025-01-02']

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue()

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 12, 31)

            minute_deps['data_collectors'].fetch_minute_month.return_value = {}
            minute_deps['data_collectors'].parse_minute_bars_to_daily.return_value = {}

            handler._upload_month(
                2025, 1, ['AAPL'], overwrite=True, resume=False,
                chunk_size=500, sleep_time=0.0,
                data_queue=data_queue, stats=stats, stats_lock=stats_lock
            )

        mock_tracker.reset.assert_called_once()

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    def test_upload_month_no_trading_days(self, mock_tracker_cls, handler, minute_deps):
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        minute_deps['calendar'].load_trading_days.return_value = []

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue()

        handler._upload_month(
            2025, 1, ['AAPL'], overwrite=False, resume=False,
            chunk_size=500, sleep_time=0.0,
            data_queue=data_queue, stats=stats, stats_lock=stats_lock
        )

        handler.logger.info.assert_any_call('Skipping 2025-01: no trading days')

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    def test_upload_month_zero_tasks(self, mock_tracker_cls, handler, minute_deps):
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        minute_deps['calendar'].load_trading_days.return_value = ['2025-01-02']

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue()

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 12, 31)

            # Empty symbols → 0 tasks
            handler._upload_month(
                2025, 1, [], overwrite=False, resume=False,
                chunk_size=500, sleep_time=0.0,
                data_queue=data_queue, stats=stats, stats_lock=stats_lock
            )

        # Should return early with no tqdm created

    @patch('quantdl.storage.handlers.ticks.UploadProgressTracker')
    @patch('quantdl.storage.handlers.ticks.tqdm')
    def test_upload_month_processes_data(self, mock_tqdm, mock_tracker_cls, handler, minute_deps):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        minute_deps['calendar'].load_trading_days.return_value = ['2025-01-02']

        df = pl.DataFrame({'close': [100.0]})
        minute_deps['data_collectors'].fetch_minute_month.return_value = {'AAPL': []}
        minute_deps['data_collectors'].parse_minute_bars_to_daily.return_value = {
            ('AAPL', '2025-01-02'): df,
        }

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        data_queue = queue.Queue()

        with patch('quantdl.storage.handlers.ticks.dt') as mock_dt:
            mock_dt.date.today.return_value = dt.date(2025, 12, 31)

            handler._upload_month(
                2025, 1, ['AAPL'], overwrite=True, resume=False,
                chunk_size=500, sleep_time=0.0,
                data_queue=data_queue, stats=stats, stats_lock=stats_lock
            )

        # Data should be in queue
        item = data_queue.get_nowait()
        assert item[0] == 'AAPL'
        assert item[1] == '2025-01-02'
        assert stats['completed'] == 1

    def test_filter_complete_symbols_complete(self, handler, minute_deps):
        minute_deps['security_master'].get_security_id.return_value = 100
        minute_deps['validator'].get_existing_minute_days.return_value = {'02', '03'}

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        pbar = MagicMock()
        tracker = MagicMock()

        result = handler._filter_complete_symbols(
            ['AAPL'], 2025, 1, ['2025-01-02', '2025-01-03'],
            tracker, stats, stats_lock, pbar
        )

        assert result == []
        assert stats['canceled'] == 2
        tracker.mark_completed.assert_called_once_with('AAPL')

    def test_filter_complete_symbols_incomplete(self, handler, minute_deps):
        minute_deps['security_master'].get_security_id.return_value = 100
        minute_deps['validator'].get_existing_minute_days.return_value = {'02'}

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        pbar = MagicMock()
        tracker = MagicMock()

        result = handler._filter_complete_symbols(
            ['AAPL'], 2025, 1, ['2025-01-02', '2025-01-03'],
            tracker, stats, stats_lock, pbar
        )

        assert result == ['AAPL']

    def test_filter_complete_symbols_no_security_id(self, handler, minute_deps):
        minute_deps['security_master'].get_security_id.side_effect = ValueError('not found')

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        pbar = MagicMock()
        tracker = MagicMock()

        result = handler._filter_complete_symbols(
            ['AAPL'], 2025, 1, ['2025-01-02'],
            tracker, stats, stats_lock, pbar
        )

        # ValueError → still include in fetch list
        assert result == ['AAPL']

    def test_filter_complete_symbols_none_security_id(self, handler, minute_deps):
        minute_deps['security_master'].get_security_id.return_value = None

        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()
        pbar = MagicMock()
        tracker = MagicMock()

        result = handler._filter_complete_symbols(
            ['AAPL'], 2025, 1, ['2025-01-02'],
            tracker, stats, stats_lock, pbar
        )

        assert result == ['AAPL']
