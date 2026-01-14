"""
Unit tests for utils.logger module
Tests logger setup and LoggerFactory
"""
import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

from quantdl.utils.logger import setup_logger, LoggerFactory


class TestSetupLogger:
    """Test setup_logger function"""

    def test_setup_logger_non_daily_rotation(self):
        """Test non-daily rotation creates logger-specific log file (line 67)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_logger(
                name='test.logger',
                log_dir=tmp_dir,
                level=logging.DEBUG,
                daily_rotation=False
            )

            # Should create a file named after the logger
            log_file = Path(tmp_dir) / 'test_logger.log'
            assert log_file.exists()

            # Clean up handlers to avoid file lock issues
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_setup_logger_daily_rotation(self):
        """Test daily rotation creates dated log file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_logger(
                name='test.daily',
                log_dir=tmp_dir,
                level=logging.DEBUG,
                daily_rotation=True
            )

            # Should create a dated log file
            log_files = list(Path(tmp_dir).glob('logs_*.log'))
            assert len(log_files) == 1

            # Clean up
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


class TestLoggerFactory:
    """Test LoggerFactory class"""

    def test_logger_factory_initialization(self):
        """Test LoggerFactory stores configuration (lines 124-128)."""
        factory = LoggerFactory(
            log_dir='test/logs',
            level=logging.INFO,
            log_format='%(message)s',
            daily_rotation=False,
            console_output=True
        )

        assert factory.log_dir == 'test/logs'
        assert factory.level == logging.INFO
        assert factory.log_format == '%(message)s'
        assert factory.daily_rotation is False
        assert factory.console_output is True

    def test_logger_factory_get_logger(self):
        """Test LoggerFactory.get_logger creates configured logger (line 138)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            factory = LoggerFactory(
                log_dir=tmp_dir,
                level=logging.WARNING,
                daily_rotation=False,
                console_output=False
            )

            logger = factory.get_logger('test.factory')

            assert isinstance(logger, logging.Logger)
            assert logger.name == 'test.factory'

            # Clean up
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_logger_factory_multiple_loggers(self):
        """Test LoggerFactory creates multiple loggers with same config."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            factory = LoggerFactory(
                log_dir=tmp_dir,
                level=logging.DEBUG,
                daily_rotation=False
            )

            logger1 = factory.get_logger('module1')
            logger2 = factory.get_logger('module2')

            assert logger1.name == 'module1'
            assert logger2.name == 'module2'

            # Both should have same log level
            assert logger1.level == logging.DEBUG
            assert logger2.level == logging.DEBUG

            # Clean up
            for logger in [logger1, logger2]:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
