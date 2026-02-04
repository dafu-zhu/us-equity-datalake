"""Unit tests for SentimentHandler."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging


class TestSentimentHandler:
    """Tests for SentimentHandler class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for SentimentHandler."""
        return {
            's3_client': Mock(),
            'bucket': 'test-bucket',
            'cik_resolver': Mock(),
            'universe_manager': Mock(),
            'logger': Mock(spec=logging.Logger)
        }

    @pytest.fixture
    def sentiment_handler(self, mock_dependencies):
        """Create SentimentHandler with mocked dependencies."""
        from quantdl.storage.handlers.sentiment import SentimentHandler
        return SentimentHandler(**mock_dependencies)

    def test_init_sets_attributes(self, mock_dependencies):
        """Test that __init__ sets all attributes correctly."""
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
        """Test that _prefetch_ciks builds symbol list from universe."""
        mock_universe = mock_dependencies['universe_manager']
        mock_universe.load_symbols_for_year.side_effect = [
            ['AAPL', 'MSFT'],  # 2020
            ['AAPL', 'GOOGL'],  # 2021
        ]

        mock_cik_resolver = mock_dependencies['cik_resolver']
        mock_cik_resolver.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044'
        }

        cik_map, symbols_with_cik = sentiment_handler._prefetch_ciks(
            '2020-01-01', '2021-12-31'
        )

        # Should have called universe for both years
        assert mock_universe.load_symbols_for_year.call_count == 2

        # Should have CIKs for all symbols
        assert 'AAPL' in cik_map
        assert 'MSFT' in cik_map
        assert 'GOOGL' in cik_map

    def test_prefetch_ciks_filters_symbols_without_cik(self, sentiment_handler, mock_dependencies):
        """Test that _prefetch_ciks filters out symbols without CIKs."""
        mock_universe = mock_dependencies['universe_manager']
        mock_universe.load_symbols_for_year.return_value = ['AAPL', 'NOCIK']

        mock_cik_resolver = mock_dependencies['cik_resolver']
        mock_cik_resolver.batch_prefetch_ciks.return_value = {
            'AAPL': '0000320193',
            'NOCIK': None  # No CIK
        }

        cik_map, symbols_with_cik = sentiment_handler._prefetch_ciks(
            '2020-01-01', '2020-12-31'
        )

        # Only AAPL should be in symbols_with_cik
        assert 'AAPL' in symbols_with_cik
        assert 'NOCIK' not in symbols_with_cik

    def test_filter_existing_skips_existing_files(self, sentiment_handler, mock_dependencies):
        """Test that _filter_existing skips already-processed symbols."""
        mock_s3 = mock_dependencies['s3_client']

        # First symbol exists, second doesn't
        def head_side_effect(Bucket, Key):
            if '0000320193' in Key:
                return {}  # Exists
            raise Exception("NoSuchKey")

        mock_s3.head_object.side_effect = head_side_effect

        symbols_with_cik = ['AAPL', 'MSFT']
        cik_map = {
            'AAPL': '0000320193',
            'MSFT': '0000789019'
        }

        result = sentiment_handler._filter_existing(symbols_with_cik, cik_map, overwrite=False)

        # Only MSFT should be in the result (AAPL exists)
        assert len(result) == 1
        assert result[0] == ('MSFT', '0000789019')
        assert sentiment_handler.skipped_exists == 1

    def test_filter_existing_with_overwrite(self, sentiment_handler, mock_dependencies):
        """Test that _filter_existing includes all symbols when overwrite=True."""
        mock_s3 = mock_dependencies['s3_client']
        mock_s3.head_object.return_value = {}  # All exist

        symbols_with_cik = ['AAPL', 'MSFT']
        cik_map = {
            'AAPL': '0000320193',
            'MSFT': '0000789019'
        }

        result = sentiment_handler._filter_existing(symbols_with_cik, cik_map, overwrite=True)

        # Both should be included despite existing
        assert len(result) == 2

    def test_build_stats_calculates_correctly(self, sentiment_handler):
        """Test that _build_stats calculates statistics correctly."""
        sentiment_handler.success = 10
        sentiment_handler.failed = 2
        sentiment_handler.skipped = 3
        sentiment_handler.skipped_exists = 5

        import time
        start_time = time.time() - 100  # 100 seconds ago

        result = sentiment_handler._build_stats(
            start_time=start_time,
            model_time=5.0,
            fetch_time=90.0,
            processed=20
        )

        assert result['success'] == 10
        assert result['failed'] == 2
        assert result['skipped'] == 3
        assert result['skipped_exists'] == 5
        assert result['model_time'] == 5.0
        assert result['fetch_time'] == 90.0
        # Rate should be ~0.22 (20/90)
        assert abs(result['avg_rate'] - 0.22) < 0.01

    def test_build_stats_handles_zero_fetch_time(self, sentiment_handler):
        """Test that _build_stats handles zero fetch time."""
        import time

        result = sentiment_handler._build_stats(
            start_time=time.time(),
            model_time=0,
            fetch_time=0,
            processed=0
        )

        assert result['avg_rate'] == 0

    def test_upload_sentiment_writes_to_s3(self, sentiment_handler, mock_dependencies):
        """Test that _upload_sentiment writes parquet to S3."""
        import polars as pl

        mock_s3 = mock_dependencies['s3_client']
        df = pl.DataFrame({
            'cik': ['0000320193'],
            'metric': ['sentiment_score'],
            'value': [0.5]
        })

        sentiment_handler._upload_sentiment('0000320193', df)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'test-bucket'
        assert call_kwargs['Key'] == 'data/derived/features/sentiment/0000320193/sentiment.parquet'


class TestSentimentHandlerUpload:
    """Tests for SentimentHandler.upload() method."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for SentimentHandler."""
        return {
            's3_client': Mock(),
            'bucket': 'test-bucket',
            'cik_resolver': Mock(),
            'universe_manager': Mock(),
            'logger': Mock(spec=logging.Logger)
        }

    @pytest.fixture
    def sentiment_handler(self, mock_dependencies):
        """Create SentimentHandler with mocked dependencies."""
        from quantdl.storage.handlers.sentiment import SentimentHandler
        return SentimentHandler(**mock_dependencies)

    @patch('quantdl.models.finbert.FinBERTModel')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    def test_upload_returns_early_when_no_symbols(
        self, mock_compute, mock_collector, mock_finbert, sentiment_handler
    ):
        """Test that upload returns early when no symbols with CIKs."""
        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch:
            mock_prefetch.return_value = ({}, [])

            result = sentiment_handler.upload('2020-01-01', '2020-12-31')

            assert result['success'] == 0

    @patch('quantdl.models.finbert.FinBERTModel')
    @patch('quantdl.collection.sentiment.SentimentCollector')
    @patch('quantdl.derived.sentiment.compute_sentiment_for_cik')
    def test_upload_unloads_model_on_completion(
        self, mock_compute, mock_collector, mock_finbert, sentiment_handler
    ):
        """Test that upload unloads model after processing."""
        mock_model = Mock()
        mock_finbert.return_value = mock_model

        with patch.object(sentiment_handler, '_prefetch_ciks') as mock_prefetch, \
             patch.object(sentiment_handler, '_filter_existing') as mock_filter:
            mock_prefetch.return_value = ({'AAPL': '123'}, ['AAPL'])
            mock_filter.return_value = []  # No symbols to process

            sentiment_handler.upload('2020-01-01', '2020-12-31')

            mock_model.unload.assert_called_once()
