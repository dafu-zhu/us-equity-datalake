"""
Integration tests for UniverseManager
Tests orchestration of fetchers, security master, and symbol management
"""
import pytest
import polars as pl
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from quantdl.universe.manager import UniverseManager


@pytest.mark.integration
class TestUniverseManagerIntegration:
    """Integration tests for UniverseManager orchestration"""

    @pytest.fixture
    def mock_crsp_fetcher(self):
        """Create a mock CRSP fetcher"""
        mock = MagicMock()
        mock.conn = MagicMock()  # Mock WRDS connection
        mock.security_master = MagicMock()
        return mock

    @pytest.fixture
    def mock_security_master(self):
        """Create a mock SecurityMaster"""
        mock = MagicMock()
        mock.db = MagicMock()
        return mock

    def test_initialization_with_defaults(self):
        """Test UniverseManager initialization with default parameters"""
        with patch('quantdl.universe.manager.CRSPDailyTicks') as mock_crsp, \
             patch('quantdl.universe.manager.Ticks') as mock_alpaca:

            # Setup mocks
            mock_crsp_instance = MagicMock()
            mock_crsp_instance.security_master = MagicMock()
            mock_crsp.return_value = mock_crsp_instance

            manager = UniverseManager()

            assert manager.crsp_fetcher is not None
            assert manager.alpaca_fetcher is not None
            assert manager.security_master is not None
            assert manager._owns_wrds_conn is True

    def test_initialization_with_custom_crsp(self, mock_crsp_fetcher):
        """Test initialization with custom CRSP fetcher"""
        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            assert manager.crsp_fetcher == mock_crsp_fetcher
            assert manager._owns_wrds_conn is False

    def test_initialization_with_security_master(self, mock_security_master):
        """Test initialization with custom SecurityMaster"""
        with patch('quantdl.universe.manager.CRSPDailyTicks') as mock_crsp, \
             patch('quantdl.universe.manager.Ticks'):

            mock_crsp_instance = MagicMock()
            mock_crsp.return_value = mock_crsp_instance

            manager = UniverseManager(security_master=mock_security_master)

            assert manager.security_master == mock_security_master
            assert manager._owns_wrds_conn is False

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols(self, mock_fetch, mock_crsp_fetcher):
        """Test getting current symbols from Nasdaq"""
        # Mock fetch_all_stocks response
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'Name': ['Apple', 'Microsoft', 'Alphabet', 'Tesla', 'Nvidia']
        })
        mock_fetch.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            symbols = manager.get_current_symbols()

            assert len(symbols) == 5
            assert 'AAPL' in symbols
            assert 'MSFT' in symbols
            mock_fetch.assert_called_once()

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_with_cache(self, mock_fetch, mock_crsp_fetcher):
        """Test that get_current_symbols uses cache on subsequent calls"""
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT'],
            'Name': ['Apple', 'Microsoft']
        })
        mock_fetch.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # First call
            symbols1 = manager.get_current_symbols()
            # Second call (should use cache)
            symbols2 = manager.get_current_symbols(refresh=False)

            assert symbols1 == symbols2
            assert mock_fetch.call_count == 1  # Only called once

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_refresh(self, mock_fetch, mock_crsp_fetcher):
        """Test refreshing current symbols"""
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL'],
            'Name': ['Apple']
        })
        mock_fetch.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # First call
            manager.get_current_symbols()
            # Refresh call
            manager.get_current_symbols(refresh=True)

            assert mock_fetch.call_count == 2  # Called twice

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_get_current_symbols_failure(self, mock_fetch, mock_crsp_fetcher):
        """Test handling of fetch failure"""
        mock_fetch.return_value = None

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            with pytest.raises(ValueError, match="Failed to fetch symbols"):
                manager.get_current_symbols()

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_load_symbols_for_future_year(self, mock_fetch, mock_crsp_fetcher):
        """Test loading symbols for year >= 2025 (uses current list)"""
        mock_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Name': ['Apple', 'Microsoft', 'Alphabet']
        })
        mock_fetch.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            symbols = manager.load_symbols_for_year(2025, sym_type='alpaca')

            assert len(symbols) == 3
            assert 'AAPL' in symbols
            mock_fetch.assert_called_once()

    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_for_historical_year(self, mock_hist, mock_crsp_fetcher):
        """Test loading symbols for year < 2025 (uses historical CRSP)"""
        mock_df = pl.DataFrame({
            'Ticker': ['AAPL', 'MSFT'],
            'Name': ['Apple', 'Microsoft'],
            'PERMNO': [12345, 67890]
        })
        mock_hist.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            symbols = manager.load_symbols_for_year(2020, sym_type='alpaca')

            assert len(symbols) == 2
            assert 'AAPL' in symbols
            mock_hist.assert_called_once_with(2020, with_validation=False, db=mock_crsp_fetcher.conn)

    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_load_symbols_sec_format(self, mock_hist, mock_crsp_fetcher):
        """Test loading symbols with SEC format (period → hyphen)"""
        mock_df = pl.DataFrame({
            'Ticker': ['BRK.B', 'BF.A'],
            'Name': ['Berkshire', 'Brown-Forman'],
            'PERMNO': [12345, 67890]
        })
        mock_hist.return_value = mock_df

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            symbols = manager.load_symbols_for_year(2020, sym_type='sec')

            assert len(symbols) == 2
            assert 'BRK-B' in symbols  # Period replaced with hyphen
            assert 'BF-A' in symbols
            assert 'BRK.B' not in symbols  # Original format not present

    def test_load_symbols_invalid_type(self, mock_crsp_fetcher):
        """Test load_symbols_for_year with invalid sym_type"""
        with patch('quantdl.universe.manager.Ticks'), \
             patch('quantdl.universe.manager.fetch_all_stocks') as mock_fetch:

            mock_fetch.return_value = pd.DataFrame({'Ticker': ['AAPL'], 'Name': ['Apple']})

            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            with pytest.raises(ValueError, match="Expected sym_type"):
                manager.load_symbols_for_year(2025, sym_type='invalid')

    def test_get_top_3000_with_crsp(self, mock_crsp_fetcher):
        """Test get_top_3000 using CRSP data source"""
        # Mock recent_daily_ticks response
        mock_data = {
            'AAPL': pl.DataFrame({
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'close': [150.0, 152.0, 151.0],
                'volume': [1000000, 1100000, 1050000]
            }),
            'MSFT': pl.DataFrame({
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'close': [300.0, 305.0, 302.0],
                'volume': [900000, 950000, 920000]
            }),
            'GOOGL': pl.DataFrame({
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'close': [100.0, 101.0, 99.0],
                'volume': [500000, 520000, 510000]
            })
        }
        mock_crsp_fetcher.recent_daily_ticks.return_value = mock_data

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            top_3000 = manager.get_top_3000(
                day='2024-01-03',
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                source='crsp'
            )

            # All 3 symbols should be returned (sorted by liquidity)
            assert len(top_3000) == 3
            # MSFT should be first (highest avg dollar volume)
            assert top_3000[0] == 'MSFT'
            mock_crsp_fetcher.recent_daily_ticks.assert_called_once()

    def test_get_top_3000_with_alpaca(self, mock_crsp_fetcher):
        """Test get_top_3000 using Alpaca data source"""
        mock_alpaca = MagicMock()
        mock_data = {
            'AAPL': pl.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'close': [150.0, 152.0],
                'volume': [2000000, 2100000]
            }),
            'MSFT': pl.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'close': [300.0, 305.0],
                'volume': [1500000, 1600000]
            })
        }
        mock_alpaca.recent_daily_ticks.return_value = mock_data

        with patch('quantdl.universe.manager.Ticks', return_value=mock_alpaca):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            top_3000 = manager.get_top_3000(
                day='2024-01-02',
                symbols=['AAPL', 'MSFT'],
                source='alpaca'
            )

            assert len(top_3000) == 2
            mock_alpaca.recent_daily_ticks.assert_called_once()

    def test_get_top_3000_filters_by_liquidity(self, mock_crsp_fetcher):
        """Test that get_top_3000 filters out low-liquidity stocks"""
        mock_data = {
            'HIGH_LIQ': pl.DataFrame({
                'date': ['2024-01-01'],
                'close': [100.0],
                'volume': [1000000]  # High liquidity: $100M
            }),
            'LOW_LIQ': pl.DataFrame({
                'date': ['2024-01-01'],
                'close': [1.0],
                'volume': [500]  # Low liquidity: $500
            })
        }
        mock_crsp_fetcher.recent_daily_ticks.return_value = mock_data

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            top_3000 = manager.get_top_3000(
                day='2024-01-01',
                symbols=['HIGH_LIQ', 'LOW_LIQ'],
                source='crsp'
            )

            # Only high liquidity stock should pass filter (> $1000 avg dollar vol)
            assert len(top_3000) == 1
            assert top_3000[0] == 'HIGH_LIQ'

    def test_get_top_3000_limits_to_3000(self, mock_crsp_fetcher):
        """Test that get_top_3000 limits results to 3000 stocks"""
        # Create mock data for 3500 symbols
        mock_data = {}
        for i in range(3500):
            symbol = f'SYM{i:04d}'
            mock_data[symbol] = pl.DataFrame({
                'date': ['2024-01-01'],
                'close': [100.0],
                'volume': [10000 + i * 100]  # Varying liquidity
            })

        mock_crsp_fetcher.recent_daily_ticks.return_value = mock_data

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            top_3000 = manager.get_top_3000(
                day='2024-01-01',
                symbols=list(mock_data.keys()),
                source='crsp'
            )

            # Should be limited to 3000
            assert len(top_3000) == 3000

    def test_get_top_3000_invalid_source(self, mock_crsp_fetcher):
        """Test get_top_3000 with invalid data source"""
        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            with pytest.raises(ValueError, match="Invalid source"):
                manager.get_top_3000(
                    day='2024-01-01',
                    symbols=['AAPL'],
                    source='invalid'
                )

    def test_get_top_3000_no_data(self, mock_crsp_fetcher):
        """Test get_top_3000 when no data is returned"""
        mock_crsp_fetcher.recent_daily_ticks.return_value = {}

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            top_3000 = manager.get_top_3000(
                day='2024-01-01',
                symbols=['AAPL'],
                source='crsp'
            )

            assert len(top_3000) == 0

    def test_close_when_owns_connection(self):
        """Test close() when manager owns WRDS connection"""
        with patch('quantdl.universe.manager.CRSPDailyTicks') as mock_crsp, \
             patch('quantdl.universe.manager.Ticks'):

            mock_crsp_instance = MagicMock()
            mock_crsp_instance.conn = MagicMock()
            mock_crsp_instance.security_master = MagicMock()
            mock_crsp.return_value = mock_crsp_instance

            manager = UniverseManager()
            manager.close()

            # Connection should be closed
            mock_crsp_instance.conn.close.assert_called_once()

    def test_close_when_not_owns_connection(self, mock_crsp_fetcher):
        """Test close() when manager doesn't own WRDS connection"""
        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)
            manager.close()

            # Connection should NOT be closed
            mock_crsp_fetcher.conn.close.assert_not_called()


@pytest.mark.integration
class TestUniverseManagerWorkflow:
    """Integration tests for complete UniverseManager workflows"""

    @pytest.fixture
    def mock_crsp_fetcher(self):
        """Create a comprehensive mock CRSP fetcher"""
        mock = MagicMock()
        mock.conn = MagicMock()
        mock.security_master = MagicMock()

        # Mock recent_daily_ticks for workflow tests
        def mock_recent_ticks(symbols, end_day, auto_resolve=True):
            data = {}
            for sym in symbols[:10]:  # Return data for first 10 symbols
                data[sym] = pl.DataFrame({
                    'date': [end_day],
                    'close': [100.0 + hash(sym) % 50],
                    'volume': [1000000 + hash(sym) % 500000]
                })
            return data

        mock.recent_daily_ticks = mock_recent_ticks
        return mock

    @patch('quantdl.universe.manager.fetch_all_stocks')
    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_complete_workflow_current_year(self, mock_hist, mock_fetch, mock_crsp_fetcher):
        """Test complete workflow for current year: get symbols → get top 3000"""
        # Setup mocks
        mock_fetch.return_value = pd.DataFrame({
            'Ticker': [f'SYM{i:03d}' for i in range(20)],
            'Name': [f'Company {i}' for i in range(20)]
        })

        with patch('quantdl.universe.manager.Ticks') as mock_alpaca_class:
            mock_alpaca = MagicMock()
            mock_alpaca.recent_daily_ticks.return_value = {
                f'SYM{i:03d}': pl.DataFrame({
                    'date': ['2025-01-15'],
                    'close': [100.0 + i],
                    'volume': [1000000 + i * 10000]
                })
                for i in range(20)
            }
            mock_alpaca_class.return_value = mock_alpaca

            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # Step 1: Load symbols for current year
            symbols = manager.load_symbols_for_year(2025)
            assert len(symbols) == 20

            # Step 2: Get top 3000 (which is all 20 in this case)
            top_3000 = manager.get_top_3000(
                day='2025-01-15',
                symbols=symbols,
                source='alpaca'
            )

            assert len(top_3000) == 20

    @patch('quantdl.universe.manager.get_hist_universe_nasdaq')
    def test_complete_workflow_historical_year(self, mock_hist, mock_crsp_fetcher):
        """Test complete workflow for historical year: load historical → get top 3000"""
        # Setup historical universe mock
        mock_hist.return_value = pl.DataFrame({
            'Ticker': [f'SYM{i:03d}' for i in range(15)],
            'Name': [f'Company {i}' for i in range(15)],
            'PERMNO': list(range(15))
        })

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # Step 1: Load historical symbols
            symbols = manager.load_symbols_for_year(2020)
            assert len(symbols) == 15

            # Step 2: Get top 3000 using CRSP data
            top_3000 = manager.get_top_3000(
                day='2020-12-31',
                symbols=symbols,
                source='crsp'
            )

            # Should have 10 symbols (mock_crsp_fetcher returns first 10)
            assert len(top_3000) == 10

    @patch('quantdl.universe.manager.fetch_all_stocks')
    def test_workflow_with_symbol_format_conversion(self, mock_fetch, mock_crsp_fetcher):
        """Test workflow with symbol format conversion (Alpaca → SEC)"""
        mock_fetch.return_value = pd.DataFrame({
            'Ticker': ['BRK.B', 'BF.A', 'AAPL'],
            'Name': ['Berkshire', 'Brown-Forman', 'Apple']
        })

        with patch('quantdl.universe.manager.Ticks'):
            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # Load with SEC format
            sec_symbols = manager.load_symbols_for_year(2025, sym_type='sec')

            assert 'BRK-B' in sec_symbols
            assert 'BF-A' in sec_symbols
            assert 'AAPL' in sec_symbols
            assert 'BRK.B' not in sec_symbols  # Should be converted

    def test_workflow_error_recovery(self, mock_crsp_fetcher):
        """Test workflow with error recovery"""
        with patch('quantdl.universe.manager.Ticks'), \
             patch('quantdl.universe.manager.fetch_all_stocks') as mock_fetch:

            # Simulate fetch failure
            mock_fetch.side_effect = Exception("Network error")

            manager = UniverseManager(crsp_fetcher=mock_crsp_fetcher)

            # load_symbols_for_year should catch exception and return empty list
            symbols = manager.load_symbols_for_year(2025)

            assert symbols == []
