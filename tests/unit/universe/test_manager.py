"""
Unit tests for universe.manager module
Tests universe manager functionality for symbol management
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
from pathlib import Path
from quantdl.universe.manager import UniverseManager


class TestUniverseManager:
    """Test UniverseManager class"""

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_default(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test UniverseManager initialization with defaults"""
        mock_crsp = Mock()
        mock_sm = Mock()
        mock_crsp.security_master = mock_sm
        mock_crsp_class.return_value = mock_crsp

        manager = UniverseManager()

        # Verify CRSP fetcher was created
        mock_crsp_class.assert_called_once()

        # Verify Alpaca fetcher was created
        mock_ticks_class.assert_called_once()

        # Verify security master was set
        assert manager.security_master == mock_sm
        assert manager._owns_wrds_conn is True

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_with_crsp_fetcher(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test initialization with provided CRSP fetcher"""
        mock_crsp = Mock()
        mock_sm = Mock()
        mock_crsp.security_master = mock_sm

        manager = UniverseManager(crsp_fetcher=mock_crsp)

        # Verify CRSP fetcher was NOT created
        mock_crsp_class.assert_not_called()

        # Verify provided fetcher was used
        assert manager.crsp_fetcher == mock_crsp
        assert manager.security_master == mock_sm
        assert manager._owns_wrds_conn is False

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test initialization with provided SecurityMaster"""
        mock_sm = Mock()
        mock_sm.db = Mock()

        mock_crsp = Mock()
        mock_crsp_class.return_value = mock_crsp

        manager = UniverseManager(security_master=mock_sm)

        # Verify CRSP fetcher was created with security master's db and require_wrds=False
        mock_crsp_class.assert_called_once_with(conn=mock_sm.db, require_wrds=False)

        # Verify provided security master was used
        assert manager.security_master == mock_sm
        assert manager._owns_wrds_conn is False

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_initialization_with_both(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test initialization with both CRSP fetcher and SecurityMaster"""
        mock_crsp = Mock()
        mock_sm = Mock()

        manager = UniverseManager(crsp_fetcher=mock_crsp, security_master=mock_sm)

        # Verify provided instances were used
        assert manager.crsp_fetcher == mock_crsp
        assert manager.security_master == mock_sm
        assert manager._owns_wrds_conn is False

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_no_cache(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test get_current_symbols without cache"""
        # Mock fetch_all_stocks
        mock_df = pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.get_current_symbols(refresh=False)

        # Verify fetch was called
        mock_fetch.assert_called_once_with(refresh=False, logger=manager.logger)

        # Verify result
        assert symbols == ['AAPL', 'MSFT', 'GOOGL']

        # Verify cache was set
        assert manager._current_symbols_cache == ['AAPL', 'MSFT', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_with_cache(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test get_current_symbols with cache"""
        manager = UniverseManager()

        # Set cache
        manager._current_symbols_cache = ['CACHED1', 'CACHED2']

        symbols = manager.get_current_symbols(refresh=False)

        # Verify fetch was NOT called
        mock_fetch.assert_not_called()

        # Verify cached result was returned
        assert symbols == ['CACHED1', 'CACHED2']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_refresh(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test get_current_symbols with refresh=True"""
        # Mock fetch_all_stocks
        mock_df = pd.DataFrame({'Ticker': ['NEW1', 'NEW2']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()

        # Set cache with old data
        manager._current_symbols_cache = ['OLD1', 'OLD2']

        symbols = manager.get_current_symbols(refresh=True)

        # Verify fetch was called with refresh=True
        mock_fetch.assert_called_once_with(refresh=True, logger=manager.logger)

        # Verify new data was returned and cached
        assert symbols == ['NEW1', 'NEW2']
        assert manager._current_symbols_cache == ['NEW1', 'NEW2']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_empty_result(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test get_current_symbols with empty result"""
        # Mock empty DataFrame
        mock_fetch.return_value = pd.DataFrame()

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_none_result(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test get_current_symbols with None result"""
        mock_fetch.return_value = None

        manager = UniverseManager()

        with pytest.raises(ValueError, match="Failed to fetch symbols"):
            manager.get_current_symbols()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_alpaca(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test load_symbols_for_year with year >= 2025 (Alpaca format)"""
        # Mock fetch_all_stocks
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="alpaca")

        # Verify fetch was called
        mock_fetch.assert_called_once()

        # Verify Alpaca format (same as Nasdaq)
        assert symbols == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_sec(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test load_symbols_for_year with year >= 2025 (SEC format)"""
        # Mock fetch_all_stocks
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_df

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'GOOGL']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_for_year_historical_alpaca(self, mock_get_hist, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test load_symbols_for_year with historical year (Alpaca format)"""
        # Mock historical universe
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'MSFT']})
        mock_get_hist.return_value = mock_df

        mock_crsp = Mock()
        mock_crsp.conn = Mock()
        mock_crsp_class.return_value = mock_crsp

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        # Verify get_hist_universe_nasdaq was called
        mock_get_hist.assert_called_once_with(2024, with_validation=False, db=mock_crsp.conn)

        # Verify Alpaca format
        assert symbols == ['AAPL', 'BRK.B', 'MSFT']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_for_year_historical_sec(self, mock_get_hist, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test load_symbols_for_year with historical year (SEC format)"""
        # Mock historical universe
        mock_df = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'MSFT']})
        mock_get_hist.return_value = mock_df

        mock_crsp = Mock()
        mock_crsp.conn = Mock()
        mock_crsp_class.return_value = mock_crsp

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="sec")

        # Verify SEC format (dots replaced with hyphens)
        assert symbols == ['AAPL', 'BRK-B', 'MSFT']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_load_symbols_for_year_invalid_type(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test load_symbols_for_year with invalid sym_type"""
        mock_logger.return_value = Mock()
        manager = UniverseManager()

        symbols = manager.load_symbols_for_year(year=2024, sym_type="invalid")
        assert symbols == []
        manager.logger.error.assert_called_once()

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    def test_store_dir_creation(self, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test that store directory is created"""
        manager = UniverseManager()

        assert manager.store_dir == Path("data/symbols")

    def test_get_top_3000_crsp(self):
        """CRSP path returns ranked symbols."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [10.0, 20.0],
            "volume": [200.0, 100.0]
        })
        manager.crsp_fetcher.recent_daily_ticks.return_value = {
            "AAA": df,
            "BBB": df
        }

        result = manager.get_top_3000("2024-06-30", ["AAA", "BBB"], source="crsp")

        assert "AAA" in result
        assert "BBB" in result
        manager.crsp_fetcher.recent_daily_ticks.assert_called_once()

    def test_get_top_3000_alpaca(self):
        """Alpaca path returns ranked symbols."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [5.0],
            "volume": [500.0]
        })
        manager.alpaca_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], source="alpaca")

        assert result == ["AAA"]
        manager.alpaca_fetcher.recent_daily_ticks.assert_called_once()

    def test_get_top_3000_invalid_source(self):
        """Invalid source raises ValueError."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        with pytest.raises(ValueError, match="Invalid source"):
            manager.get_top_3000("2024-06-30", ["AAA"], source="bad")

    def test_get_top_3000_no_liquidity(self):
        """No symbols pass liquidity filter."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        manager.crsp_fetcher.recent_daily_ticks.return_value = {}

        result = manager.get_top_3000("2024-06-30", ["AAA"], source="crsp")

        assert result == []
        manager.logger.error.assert_called_once()

    def test_get_top_3000_filters_threshold(self):
        """Filters out symbols below average dollar volume threshold."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        high_df = pl.DataFrame({"close": [50.0], "volume": [100.0]})
        low_df = pl.DataFrame({"close": [1.0], "volume": [10.0]})
        manager.crsp_fetcher.recent_daily_ticks.return_value = {
            "HIGH": high_df,
            "LOW": low_df
        }

        result = manager.get_top_3000("2024-06-30", ["HIGH", "LOW"], source="crsp")

        assert result == ["HIGH"]

    def test_get_top_3000_source_case_insensitive(self):
        """Source comparison is case-insensitive."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({"close": [5.0], "volume": [500.0]})
        manager.crsp_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], source="CRSP")

        assert result == ["AAA"]


class TestUniverseManagerEdgeCases:
    """Test edge cases for UniverseManager"""

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_cache_invalidation_on_refresh(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test that cache is updated when refresh=True"""
        manager = UniverseManager()

        # First call with old data
        mock_df1 = pd.DataFrame({'Ticker': ['OLD1', 'OLD2']})
        mock_fetch.return_value = mock_df1
        symbols1 = manager.get_current_symbols(refresh=False)

        assert symbols1 == ['OLD1', 'OLD2']

        # Second call with refresh and new data
        mock_df2 = pd.DataFrame({'Ticker': ['NEW1', 'NEW2', 'NEW3']})
        mock_fetch.return_value = mock_df2
        symbols2 = manager.get_current_symbols(refresh=True)

        assert symbols2 == ['NEW1', 'NEW2', 'NEW3']
        assert manager._current_symbols_cache == ['NEW1', 'NEW2', 'NEW3']

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_crsp_connection_reuse(self, mock_get_hist, mock_logger, mock_crsp_class, mock_ticks_class):
        """Test that CRSP connection is reused"""
        mock_df = pl.DataFrame({'Ticker': ['AAPL']})
        mock_get_hist.return_value = mock_df

        mock_crsp = Mock()
        mock_conn = Mock()
        mock_crsp.conn = mock_conn
        mock_crsp_class.return_value = mock_crsp

        manager = UniverseManager()
        manager.load_symbols_for_year(year=2020, sym_type="alpaca")

        # Verify connection was passed to get_hist_universe_nasdaq
        mock_get_hist.assert_called_once()
        call_kwargs = mock_get_hist.call_args[1]
        assert call_kwargs['db'] == mock_conn

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_year_2025_empty(self, mock_fetch, mock_logger, mock_crsp_class, mock_ticks_class):
        """Empty current universe returns empty list."""
        mock_fetch.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2025, sym_type="alpaca")

        assert symbols == []

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_for_year_historical_empty(self, mock_get_hist, mock_logger, mock_crsp_class, mock_ticks_class):
        """Empty historical universe returns empty list."""
        mock_get_hist.return_value = pl.DataFrame({'Ticker': []})
        mock_logger.return_value = Mock()

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        assert symbols == []

    @patch('quantdl.universe.manager.Ticks')
    @patch('quantdl.universe.manager.CRSPDailyTicks')
    @patch('quantdl.universe.manager.setup_logger')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_for_year_exception(self, mock_get_hist, mock_logger, mock_crsp_class, mock_ticks_class):
        """Exceptions while loading symbols return empty list."""
        mock_get_hist.side_effect = Exception("boom")

        manager = UniverseManager()
        symbols = manager.load_symbols_for_year(year=2024, sym_type="alpaca")

        assert symbols == []

    def test_close_closes_wrds_connection(self):
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager._owns_wrds_conn = True
        manager.crsp_fetcher = Mock()
        manager.crsp_fetcher.conn = Mock()

        manager.close()

        manager.crsp_fetcher.conn.close.assert_called_once()

    def test_close_does_not_close_when_not_owner(self):
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager._owns_wrds_conn = False
        manager.crsp_fetcher = Mock()
        manager.crsp_fetcher.conn = Mock()

        manager.close()

        manager.crsp_fetcher.conn.close.assert_not_called()

    def test_close_when_conn_is_none(self):
        """Test close when connection is None."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager._owns_wrds_conn = True
        manager.crsp_fetcher = Mock()
        manager.crsp_fetcher.conn = None

        # Should not raise an error
        manager.close()

    def test_close_when_no_conn_attribute(self):
        """Test close when crsp_fetcher has no conn attribute."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager._owns_wrds_conn = True
        manager.crsp_fetcher = Mock(spec=[])  # Mock with no attributes

        # Should not raise an error
        manager.close()

    def test_close_when_no_crsp_fetcher(self):
        """Test close when crsp_fetcher doesn't exist."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager._owns_wrds_conn = True

        # Should not raise an error
        manager.close()

    def test_get_top_3000_with_auto_resolve_true(self):
        """Test get_top_3000 with auto_resolve=True."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        manager.crsp_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], source="crsp", auto_resolve=True)

        # Verify auto_resolve was passed through
        manager.crsp_fetcher.recent_daily_ticks.assert_called_once_with(
            ["AAA"],
            end_day="2024-06-30",
            auto_resolve=True
        )
        assert result == ["AAA"]

    def test_get_top_3000_with_auto_resolve_false(self):
        """Test get_top_3000 with auto_resolve=False."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        manager.crsp_fetcher.recent_daily_ticks.return_value = {"AAA": df}

        result = manager.get_top_3000("2024-06-30", ["AAA"], source="crsp", auto_resolve=False)

        # Verify auto_resolve was passed through
        manager.crsp_fetcher.recent_daily_ticks.assert_called_once_with(
            ["AAA"],
            end_day="2024-06-30",
            auto_resolve=False
        )
        assert result == ["AAA"]

    def test_get_top_3000_ranks_by_liquidity(self):
        """Test that get_top_3000 correctly ranks symbols by liquidity."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        # Symbol with high liquidity
        high_df = pl.DataFrame({
            "close": [100.0, 100.0],
            "volume": [1000.0, 1000.0]
        })
        # Symbol with medium liquidity
        mid_df = pl.DataFrame({
            "close": [50.0, 50.0],
            "volume": [500.0, 500.0]
        })
        # Symbol with low liquidity (but above threshold)
        low_df = pl.DataFrame({
            "close": [10.0, 10.0],
            "volume": [200.0, 200.0]
        })

        manager.crsp_fetcher.recent_daily_ticks.return_value = {
            "MID": mid_df,
            "HIGH": high_df,
            "LOW": low_df
        }

        result = manager.get_top_3000("2024-06-30", ["MID", "HIGH", "LOW"], source="crsp")

        # Verify symbols are ranked by average dollar volume
        assert result == ["HIGH", "MID", "LOW"]

    def test_get_top_3000_empty_dataframe_excluded(self):
        """Test that symbols with empty dataframes are excluded."""
        manager = UniverseManager.__new__(UniverseManager)
        manager.logger = Mock()
        manager.crsp_fetcher = Mock()
        manager.alpaca_fetcher = Mock()

        good_df = pl.DataFrame({
            "close": [10.0],
            "volume": [200.0]
        })
        empty_df = pl.DataFrame({
            "close": [],
            "volume": []
        })

        manager.crsp_fetcher.recent_daily_ticks.return_value = {
            "GOOD": good_df,
            "EMPTY": empty_df
        }

        result = manager.get_top_3000("2024-06-30", ["GOOD", "EMPTY"], source="crsp")

        # Only GOOD should be in the result
        assert result == ["GOOD"]
