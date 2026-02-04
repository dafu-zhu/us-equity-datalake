"""Unit tests for storage handlers base class."""

import pytest
from unittest.mock import Mock
import logging


class TestBaseHandler:
    """Tests for BaseHandler class."""

    def test_init_creates_logger_and_stats(self):
        """Test that __init__ sets up logger and stats."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        assert handler.logger is mock_logger
        assert handler.stats == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def test_reset_stats_clears_all_counters(self):
        """Test that reset_stats() resets all counters to zero."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        # Modify stats
        handler.stats['success'] = 10
        handler.stats['failed'] = 5
        handler.stats['skipped'] = 3
        handler.stats['canceled'] = 2

        # Reset
        handler.reset_stats()

        assert handler.stats == {'success': 0, 'failed': 0, 'skipped': 0, 'canceled': 0}

    def test_log_summary_calculates_rate(self):
        """Test that log_summary() calculates rate correctly."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        handler.stats['success'] = 80
        handler.stats['failed'] = 10
        handler.stats['skipped'] = 5
        handler.stats['canceled'] = 5

        result = handler.log_summary("Test Task", total=100, elapsed=10.0)

        # Check rate calculation
        assert result['rate'] == 10.0  # 100 items / 10 seconds
        assert result['total'] == 100
        assert result['elapsed'] == 10.0
        assert result['success'] == 80
        assert result['failed'] == 10
        assert result['skipped'] == 5
        assert result['canceled'] == 5

        # Check logger was called
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Test Task" in log_message
        assert "80 success" in log_message
        assert "10 failed" in log_message

    def test_log_summary_handles_zero_elapsed(self):
        """Test that log_summary() handles zero elapsed time."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        result = handler.log_summary("Test Task", total=100, elapsed=0.0)

        # Rate should be 0 when elapsed is 0
        assert result['rate'] == 0

    def test_log_summary_handles_negative_elapsed(self):
        """Test that log_summary() handles negative elapsed time gracefully."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        result = handler.log_summary("Test Task", total=100, elapsed=-1.0)

        # Rate calculation with negative elapsed: 100 / -1 = -100
        # But the code uses `if elapsed > 0` check, so rate = 0
        assert result['rate'] == 0

    def test_stats_can_be_incremented(self):
        """Test that stats dictionary can be modified."""
        from quantdl.storage.handlers.base import BaseHandler

        mock_logger = Mock(spec=logging.Logger)
        handler = BaseHandler(logger=mock_logger)

        for _ in range(5):
            handler.stats['success'] += 1

        for _ in range(2):
            handler.stats['failed'] += 1

        assert handler.stats['success'] == 5
        assert handler.stats['failed'] == 2
        assert handler.stats['skipped'] == 0
        assert handler.stats['canceled'] == 0
