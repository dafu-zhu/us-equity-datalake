"""Unit tests for SentimentHandler, SentimentCheckpoint, MDACache."""

import json
import time
import threading
from queue import Queue

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging
import polars as pl

from quantdl.storage.handlers.sentiment import (
    SentimentHandler,
    SentimentCheckpoint,
    MDACache,
)


# ── SentimentCheckpoint ─────────────────────────────────────────────────────


class TestSentimentCheckpoint:

    @pytest.fixture
    def s3(self):
        return Mock()

    @pytest.fixture
    def logger(self):
        return Mock(spec=logging.Logger)

    def test_init_loads_checkpoint(self, s3, logger):
        body = Mock()
        body.read.return_value = json.dumps({'processed_ciks': ['CIK1', 'CIK2']}).encode()
        s3.get_object.return_value = {'Body': body}

        cp = SentimentCheckpoint(s3, 'bucket', logger)

        assert cp._processed == {'CIK1', 'CIK2'}
        logger.info.assert_any_call('Loaded checkpoint: 2 CIKs already processed')

    def test_init_no_checkpoint(self, s3, logger):
        s3.get_object.side_effect = Exception('NoSuchKey')

        cp = SentimentCheckpoint(s3, 'bucket', logger)

        assert cp._processed == set()
        logger.info.assert_any_call('No checkpoint found, starting fresh')

    def test_is_processed(self, s3, logger):
        s3.get_object.side_effect = Exception()
        cp = SentimentCheckpoint(s3, 'bucket', logger)
        cp._processed = {'CIK1'}

        assert cp.is_processed('CIK1') is True
        assert cp.is_processed('CIK2') is False

    def test_mark_processed_saves_every_10(self, s3, logger):
        s3.get_object.side_effect = Exception()
        cp = SentimentCheckpoint(s3, 'bucket', logger)

        for i in range(9):
            cp.mark_processed(f'CIK{i}')
        s3.put_object.assert_not_called()

        cp.mark_processed('CIK9')
        s3.put_object.assert_called_once()

    def test_mark_processed_no_save_before_10(self, s3, logger):
        s3.get_object.side_effect = Exception()
        cp = SentimentCheckpoint(s3, 'bucket', logger)

        cp.mark_processed('CIK0')
        s3.put_object.assert_not_called()

    def test_save_force(self, s3, logger):
        s3.get_object.side_effect = Exception()
        cp = SentimentCheckpoint(s3, 'bucket', logger)
        cp._processed = {'CIK1', 'CIK2'}

        cp.save()

        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'bucket'
        assert call_kwargs['Key'] == 'data/cache/sentiment_checkpoint.json'
        body = json.loads(call_kwargs['Body'].decode())
        assert set(body['processed_ciks']) == {'CIK1', 'CIK2'}

    def test_count(self, s3, logger):
        s3.get_object.side_effect = Exception()
        cp = SentimentCheckpoint(s3, 'bucket', logger)
        cp._processed = {'CIK1', 'CIK2', 'CIK3'}

        assert cp.count() == 3


# ── MDACache ─────────────────────────────────────────────────────────────────


class TestMDACache:

    @pytest.fixture
    def s3(self):
        return Mock()

    @pytest.fixture
    def logger(self):
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def cache(self, s3, logger):
        return MDACache(s3, 'bucket', logger)

    def test_init(self, cache, s3, logger):
        assert cache.s3_client is s3
        assert cache.bucket == 'bucket'
        assert cache._local_cache == {}

    def test_cache_key(self, cache):
        assert cache._cache_key('CIK123') == 'data/cache/mda/CIK123/mda_cache.json'

    def test_get_local_cache_hit(self, cache):
        cache._local_cache['CIK1'] = {'cik': 'CIK1', 'filings': []}

        result = cache.get('CIK1')

        assert result == {'cik': 'CIK1', 'filings': []}
        cache.s3_client.get_object.assert_not_called()

    def test_get_s3_hit(self, cache, s3):
        body = Mock()
        body.read.return_value = json.dumps({'cik': 'CIK1', 'filings': []}).encode()
        s3.get_object.return_value = {'Body': body}

        result = cache.get('CIK1')

        assert result == {'cik': 'CIK1', 'filings': []}
        assert cache._local_cache['CIK1'] == result

    def test_get_s3_miss(self, cache, s3):
        s3.get_object.side_effect = Exception('NoSuchKey')

        result = cache.get('CIK1')

        assert result is None

    def test_put_stores_locally_and_s3(self, cache, s3):
        filing_text = Mock()
        filing_text.to_dict.return_value = {'cik': 'CIK1', 'text': 'hello'}

        cache.put('CIK1', [filing_text])

        s3.put_object.assert_called_once()
        assert 'CIK1' in cache._local_cache

    def test_has_local_hit(self, cache, s3):
        cache._local_cache['CIK1'] = {'filings': []}

        assert cache.has('CIK1') is True
        s3.head_object.assert_not_called()

    def test_has_s3_hit(self, cache, s3):
        s3.head_object.return_value = {}

        assert cache.has('CIK1') is True

    def test_has_s3_miss(self, cache, s3):
        s3.head_object.side_effect = Exception('NoSuchKey')

        assert cache.has('CIK1') is False


# ── SentimentHandler ─────────────────────────────────────────────────────────


class TestSentimentHandler:
    """Tests for SentimentHandler class."""

    @pytest.fixture
    def mock_dependencies(self):
        return {
            's3_client': Mock(),
            'bucket': 'test-bucket',
            'cik_resolver': Mock(),
            'universe_manager': Mock(),
            'logger': Mock(spec=logging.Logger),
        }

    @pytest.fixture
    def sentiment_handler(self, mock_dependencies):
        from quantdl.storage.handlers.sentiment import SentimentHandler
        return SentimentHandler(**mock_dependencies)

    def test_init_sets_attributes(self, mock_dependencies):
        from quantdl.storage.handlers.sentiment import SentimentHandler

        handler = SentimentHandler(**mock_dependencies)

        assert handler.s3_client is mock_dependencies['s3_client']
        assert handler.bucket == 'test-bucket'
        assert handler.cik_resolver is mock_dependencies['cik_resolver']
        assert handler.universe_manager is mock_dependencies['universe_manager']
        assert handler.logger is mock_dependencies['logger']
        assert handler.success == 0
        assert handler.failed == 0
        assert handler.skipped == 0
        assert handler.skipped_exists == 0

    def test_prefetch_ciks_builds_symbol_list(self, sentiment_handler, mock_dependencies):
        mock_universe = mock_dependencies['universe_manager']
        mock_universe.load_symbols_for_year.side_effect = [
            ['AAPL', 'MSFT'],
            ['AAPL', 'GOOGL'],
        ]

        mock_cik_resolver = mock_dependencies['cik_resolver']
        mock_cik_resolver.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
        }

        cik_map, symbols_with_cik = sentiment_handler._prefetch_ciks(
            '2020-01-01', '2021-12-31'
        )

        assert mock_universe.load_symbols_for_year.call_count == 2
        assert 'AAPL' in cik_map
        assert 'MSFT' in cik_map
        assert 'GOOGL' in cik_map

    def test_prefetch_ciks_filters_symbols_without_cik(self, sentiment_handler, mock_dependencies):
        mock_universe = mock_dependencies['universe_manager']
        mock_universe.load_symbols_for_year.return_value = ['AAPL', 'NOCIK']

        mock_cik_resolver = mock_dependencies['cik_resolver']
        mock_cik_resolver.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'NOCIK': None,
        }

        cik_map, symbols_with_cik = sentiment_handler._prefetch_ciks(
            '2020-01-01', '2020-12-31'
        )

        assert 'AAPL' in symbols_with_cik
        assert 'NOCIK' not in symbols_with_cik

    def test_filter_existing_skips_existing_files(self, sentiment_handler, mock_dependencies):
        mock_s3 = mock_dependencies['s3_client']

        def head_side_effect(Bucket, Key):
            if '0000320193' in Key:
                return {}
            raise Exception("NoSuchKey")

        mock_s3.head_object.side_effect = head_side_effect

        result = sentiment_handler._filter_existing(
            ['AAPL', 'MSFT'],
            {'AAPL': '0000320193', 'MSFT': '0000789019'},
            overwrite=False,
        )

        assert len(result) == 1
        assert result[0] == ('MSFT', '0000789019')
        assert sentiment_handler.skipped_exists == 1

    def test_filter_existing_with_overwrite(self, sentiment_handler, mock_dependencies):
        mock_s3 = mock_dependencies['s3_client']
        mock_s3.head_object.return_value = {}

        result = sentiment_handler._filter_existing(
            ['AAPL', 'MSFT'],
            {'AAPL': '0000320193', 'MSFT': '0000789019'},
            overwrite=True,
        )

        assert len(result) == 2

    def test_build_stats_calculates_correctly(self, sentiment_handler):
        sentiment_handler.success = 10
        sentiment_handler.failed = 2
        sentiment_handler.skipped = 3
        sentiment_handler.skipped_exists = 5

        start_time = time.time() - 100

        result = sentiment_handler._build_stats(
            start_time=start_time, model_time=5.0, fetch_time=90.0, processed=20
        )

        assert result['success'] == 10
        assert result['failed'] == 2
        assert result['skipped'] == 3
        assert result['skipped_exists'] == 5
        assert result['model_time'] == 5.0
        assert result['fetch_time'] == 90.0
        assert abs(result['avg_rate'] - 0.22) < 0.01

    def test_build_stats_handles_zero_fetch_time(self, sentiment_handler):
        result = sentiment_handler._build_stats(
            start_time=time.time(), model_time=0, fetch_time=0, processed=0
        )
        assert result['avg_rate'] == 0

    def test_upload_sentiment_writes_to_s3(self, sentiment_handler, mock_dependencies):
        mock_s3 = mock_dependencies['s3_client']
        df = pl.DataFrame({
            'cik': ['0000320193'],
            'metric': ['sentiment_score'],
            'value': [0.5],
        })

        sentiment_handler._upload_sentiment('0000320193', df)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Key'] == 'data/derived/features/sentiment/0000320193/sentiment.parquet'


# ── SentimentHandler.upload() ────────────────────────────────────────────────


class TestSentimentHandlerUpload:
    """Tests for SentimentHandler.upload() method."""

    @pytest.fixture
    def mock_dependencies(self):
        return {
            's3_client': Mock(),
            'bucket': 'test-bucket',
            'cik_resolver': Mock(),
            'universe_manager': Mock(),
            'logger': Mock(spec=logging.Logger),
        }

    @pytest.fixture
    def sentiment_handler(self, mock_dependencies):
        from quantdl.storage.handlers.sentiment import SentimentHandler
        return SentimentHandler(**mock_dependencies)

    @patch('quantdl.storage.handlers.sentiment.SentimentCheckpoint')
    @patch('quantdl.storage.handlers.sentiment.MDACache')
    @patch('quantdl.storage.handlers.sentiment.RateLimiter')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.models.finbert.FinBERTModel')
    def test_upload_returns_early_when_no_symbols(
        self, mock_finbert, mock_collector, mock_compute,
        mock_rl, mock_mda, mock_ckpt, sentiment_handler
    ):
        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch:
            mock_prefetch.return_value = ({}, [])
            result = sentiment_handler.upload('2020-01-01', '2020-12-31')
        assert result['success'] == 0

    @patch('quantdl.storage.handlers.sentiment.SentimentCheckpoint')
    @patch('quantdl.storage.handlers.sentiment.MDACache')
    @patch('quantdl.storage.handlers.sentiment.RateLimiter')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.models.finbert.FinBERTModel')
    def test_upload_unloads_model_on_completion(
        self, mock_finbert, mock_collector, mock_compute,
        mock_rl, mock_mda, mock_ckpt, sentiment_handler
    ):
        mock_model = Mock()
        mock_finbert.return_value = mock_model

        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch, \
             patch.object(sentiment_handler, '_filter_existing') as mock_filter:
            mock_prefetch.return_value = ({'AAPL': '123'}, ['AAPL'])
            mock_filter.return_value = []

            sentiment_handler.upload('2020-01-01', '2020-12-31')

            mock_model.unload.assert_called_once()

    @patch('quantdl.storage.handlers.sentiment.SentimentCheckpoint')
    @patch('quantdl.storage.handlers.sentiment.MDACache')
    @patch('quantdl.storage.handlers.sentiment.RateLimiter')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.models.finbert.FinBERTModel')
    def test_upload_full_flow_with_checkpoint(
        self, mock_finbert, mock_collector, mock_compute,
        mock_rl, mock_mda_cls, mock_ckpt_cls, sentiment_handler
    ):
        mock_model = Mock()
        mock_finbert.return_value = mock_model

        mock_checkpoint = Mock()
        mock_checkpoint.is_processed.side_effect = lambda cik: cik == 'CIK_DONE'
        mock_checkpoint.count.return_value = 1
        mock_ckpt_cls.return_value = mock_checkpoint

        mock_mda = Mock()
        mock_mda_cls.return_value = mock_mda

        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch, \
             patch.object(sentiment_handler, '_filter_existing') as mock_filter, \
             patch.object(sentiment_handler, '_process_symbols') as mock_process:
            mock_prefetch.return_value = ({'AAPL': 'CIK1', 'DONE': 'CIK_DONE'}, ['AAPL', 'DONE'])
            mock_filter.return_value = [('AAPL', 'CIK1'), ('DONE', 'CIK_DONE')]

            sentiment_handler.upload('2020-01-01', '2020-12-31')

            # _process_symbols should receive only non-checkpointed symbols
            call_kwargs = mock_process.call_args
            syms = call_kwargs[1]['symbols_to_process']
            assert ('AAPL', 'CIK1') in syms
            assert ('DONE', 'CIK_DONE') not in syms

    @patch('quantdl.storage.handlers.sentiment.SentimentCheckpoint')
    @patch('quantdl.storage.handlers.sentiment.MDACache')
    @patch('quantdl.storage.handlers.sentiment.RateLimiter')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.models.finbert.FinBERTModel')
    def test_upload_finally_saves_checkpoint(
        self, mock_finbert, mock_collector, mock_compute,
        mock_rl, mock_mda_cls, mock_ckpt_cls, sentiment_handler
    ):
        mock_model = Mock()
        mock_finbert.return_value = mock_model

        mock_checkpoint = Mock()
        mock_checkpoint.is_processed.return_value = False
        mock_checkpoint.count.return_value = 0
        mock_ckpt_cls.return_value = mock_checkpoint

        mock_mda = Mock()
        mock_mda_cls.return_value = mock_mda

        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch, \
             patch.object(sentiment_handler, '_filter_existing') as mock_filter, \
             patch.object(sentiment_handler, '_process_symbols', side_effect=Exception('boom')):
            mock_prefetch.return_value = ({'AAPL': 'CIK1'}, ['AAPL'])
            mock_filter.return_value = [('AAPL', 'CIK1')]

            with pytest.raises(Exception, match='boom'):
                sentiment_handler.upload('2020-01-01', '2020-12-31')

            mock_checkpoint.save.assert_called_once()
            mock_model.unload.assert_called_once()


# ── SentimentHandler._process_symbols() ──────────────────────────────────────


class TestSentimentHandlerProcessSymbols:
    """Tests for _process_symbols() consumer/producer pipeline."""

    @pytest.fixture
    def mock_dependencies(self):
        return {
            's3_client': Mock(),
            'bucket': 'test-bucket',
            'cik_resolver': Mock(),
            'universe_manager': Mock(),
            'logger': Mock(spec=logging.Logger),
        }

    @pytest.fixture
    def handler(self, mock_dependencies):
        return SentimentHandler(**mock_dependencies)

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.name = 'finbert'
        model.version = '1.0'
        return model

    @pytest.fixture
    def mock_collector(self):
        return Mock()

    @pytest.fixture
    def mock_checkpoint(self):
        cp = Mock()
        cp.count.return_value = 0
        return cp

    @pytest.fixture
    def mock_mda_cache(self):
        return Mock()

    def _make_filing_text(self, cik='CIK1', text='Positive outlook for growth.'):
        return {
            'cik': cik,
            'accession_number': '0001-23-456',
            'filing_date': '2020-06-30',
            'filing_type': '10-K',
            'section': 'MD&A',
            'text': text,
            'fiscal_year': 2020,
            'fiscal_quarter': None,
        }

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_empty_list(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        handler._process_symbols(
            symbols_to_process=[],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        assert handler.success == 0
        assert handler.failed == 0

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_no_filings(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        mock_mda_cache.get.return_value = None
        mock_collector.get_filings_metadata.return_value = []

        handler._process_symbols(
            symbols_to_process=[('AAPL', 'CIK1')],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        time.sleep(0.5)
        assert handler.skipped >= 1

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_fetch_error(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        mock_mda_cache.get.return_value = None
        mock_collector.get_filings_metadata.side_effect = Exception('Network error')

        handler._process_symbols(
            symbols_to_process=[('AAPL', 'CIK1')],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        time.sleep(0.5)
        assert handler.failed >= 1

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_cache_hit(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        cached_filing = self._make_filing_text()
        mock_mda_cache.get.return_value = {'cik': 'CIK1', 'filings': [cached_filing]}

        # Model returns a result for the chunk
        mock_model.predict.return_value = [Mock(label='positive', score=0.9)]

        with patch('quantdl.derived.sentiment._aggregate_sentiment_results') as mock_agg, \
             patch('quantdl.derived.sentiment.compute_sentiment_long') as mock_compute:
            mock_agg.return_value = Mock()
            mock_compute.return_value = pl.DataFrame({'metric': ['score'], 'value': [0.9]})

            handler._process_symbols(
                symbols_to_process=[('AAPL', 'CIK1')],
                collector=mock_collector,
                model=mock_model,
                compute_fn=Mock(),
                start_date='2020-01-01',
                end_date='2020-12-31',
                mda_cache=mock_mda_cache,
                checkpoint=mock_checkpoint,
            )

            time.sleep(1.0)

        # Cache hit path: no SEC fetch needed
        mock_collector.get_filings_metadata.assert_not_called()

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_empty_filing_text(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        cached_filing = self._make_filing_text(text=None)
        mock_mda_cache.get.return_value = {'cik': 'CIK1', 'filings': [cached_filing]}

        mock_model.predict.return_value = []

        handler._process_symbols(
            symbols_to_process=[('AAPL', 'CIK1')],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        time.sleep(0.5)

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_batch_inference_failure(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        cached_filing = self._make_filing_text()
        mock_mda_cache.get.return_value = {'cik': 'CIK1', 'filings': [cached_filing]}

        mock_model.predict.side_effect = RuntimeError('GPU OOM')

        handler._process_symbols(
            symbols_to_process=[('AAPL', 'CIK1')],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        time.sleep(0.5)
        assert handler.failed >= 1

    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_checkpoint_marks(
        self, mock_tqdm, handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        cached_filing = self._make_filing_text(text='Revenue increased significantly.')
        mock_mda_cache.get.return_value = {'cik': 'CIK1', 'filings': [cached_filing]}

        mock_model.predict.return_value = [
            Mock(label='positive', score=0.9)
        ]

        with patch('quantdl.derived.sentiment._aggregate_sentiment_results') as mock_agg, \
             patch('quantdl.derived.sentiment.compute_sentiment_long') as mock_compute:
            mock_agg.return_value = Mock()
            mock_compute.return_value = pl.DataFrame({'metric': ['score'], 'value': [0.9]})

            handler._process_symbols(
                symbols_to_process=[('AAPL', 'CIK1')],
                collector=mock_collector,
                model=mock_model,
                compute_fn=Mock(),
                start_date='2020-01-01',
                end_date='2020-12-31',
                mda_cache=mock_mda_cache,
                checkpoint=mock_checkpoint,
            )

            time.sleep(1.0)

    @patch('quantdl.derived.sentiment.compute_sentiment_long')
    @patch('quantdl.derived.sentiment._aggregate_sentiment_results')
    @patch('quantdl.derived.sentiment.chunk_text')
    @patch('quantdl.storage.handlers.sentiment.tqdm')
    def test_process_symbols_batch_inference_success(
        self, mock_tqdm, mock_chunk, mock_agg, mock_compute,
        handler, mock_model, mock_collector, mock_checkpoint, mock_mda_cache
    ):
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        cached_filing = self._make_filing_text(text='Revenue grew by 20%.')
        mock_mda_cache.get.return_value = {'cik': 'CIK1', 'filings': [cached_filing]}

        mock_chunk.return_value = ['Revenue grew by 20%.']
        mock_model.predict.return_value = [Mock(label='positive', score=0.9)]
        mock_agg.return_value = Mock()
        mock_compute.return_value = pl.DataFrame({'metric': ['score'], 'value': [0.9]})

        handler._process_symbols(
            symbols_to_process=[('AAPL', 'CIK1')],
            collector=mock_collector,
            model=mock_model,
            compute_fn=Mock(),
            start_date='2020-01-01',
            end_date='2020-12-31',
            mda_cache=mock_mda_cache,
            checkpoint=mock_checkpoint,
        )

        time.sleep(1.0)
