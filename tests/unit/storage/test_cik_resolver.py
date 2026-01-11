"""
Unit tests for storage.cik_resolver module
Tests CIKResolver functionality against SecurityMaster-backed data.
"""
import datetime as dt
from unittest.mock import Mock

import polars as pl

from quantdl.storage.cik_resolver import CIKResolver


class TestCIKResolver:
    """Test CIKResolver class"""

    def _make_master_tb(self, security_id: str, cik_value):
        return pl.DataFrame(
            {
                "security_id": [security_id],
                "start_date": [dt.date(2020, 1, 1)],
                "end_date": [dt.date(2025, 12, 31)],
                "cik": [cik_value],
            }
        )

    def test_get_cik_primary_date(self):
        """Uses master table for the primary date."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik == "320193"
        security_master.get_security_id.assert_called_once_with(
            symbol="AAPL",
            day="2024-06-30",
            auto_resolve=True,
        )

    def test_get_cik_null_cik_returns_none(self):
        """NULL CIK in master table returns None."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", None)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None

    def test_get_cik_uses_fallback_dates(self):
        """Falls back to later dates when primary date is inactive."""
        security_master = Mock()

        def _get_sid(symbol, day, auto_resolve=True):
            if day == "2024-01-15":
                raise ValueError("not active on")
            return "sid_1"

        security_master.get_security_id.side_effect = _get_sid
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-01-15", year=2024)

        assert cik == "320193"

    def test_get_cik_prefers_sec_mapping_for_2025_plus(self):
        """Uses SEC mapping when year is 2025+."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)
        security_master._fetch_sec_cik_mapping.return_value = pl.DataFrame(
            {"ticker": ["AAPL"], "cik": ["0000320193"]}
        )

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2025-06-30", year=2025)

        assert cik == "0000320193"

    def test_get_cik_sec_mapping_exception_logs_debug(self):
        """Logs debug when SEC mapping lookup fails."""
        security_master = Mock()
        security_master.get_security_id.return_value = "sid_1"
        security_master.master_tb = self._make_master_tb("sid_1", 320193)
        security_master._fetch_sec_cik_mapping.side_effect = RuntimeError("sec down")

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2025-06-30", year=2025)

        assert cik == "320193"
        logger.debug.assert_called()

    def test_get_cik_value_error_logs_warning(self):
        """Logs warning on unexpected ValueError from SecurityMaster."""
        security_master = Mock()
        security_master.get_security_id.side_effect = ValueError("boom")
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None
        logger.warning.assert_called()

    def test_get_cik_unexpected_exception_logs_error(self):
        """Logs error on unexpected exceptions from SecurityMaster."""
        security_master = Mock()
        security_master.get_security_id.side_effect = RuntimeError("boom")
        security_master.master_tb = self._make_master_tb("sid_1", 320193)

        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        cik = resolver.get_cik(symbol="AAPL", date="2024-06-30", year=2024)

        assert cik is None
        logger.error.assert_called()

    def test_batch_prefetch_uses_cache(self):
        """Returns cached results without calling get_cik."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)
        resolver._cik_cache[("AAPL", 2024)] = "320193"

        resolver.get_cik = Mock()

        result = resolver.batch_prefetch_ciks(["AAPL"], year=2024)

        assert result == {"AAPL": "320193"}
        resolver.get_cik.assert_not_called()

    def test_batch_prefetch_populates_cache(self):
        """Populates cache entries for missing symbols."""
        security_master = Mock()
        logger = Mock()
        resolver = CIKResolver(security_master=security_master, logger=logger)

        resolver.get_cik = Mock(side_effect=["320193", None])

        result = resolver.batch_prefetch_ciks(["AAPL", "NOPE"], year=2024, batch_size=2)

        assert result == {"AAPL": "320193", "NOPE": None}
        assert resolver._cik_cache[("AAPL", 2024)] == "320193"
        assert resolver._cik_cache[("NOPE", 2024)] is None
