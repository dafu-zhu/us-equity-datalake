"""
Unit tests for storage.data_collectors module
Tests data collection functionality with new DataCollector architecture
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from collections import OrderedDict
import threading
import polars as pl
import datetime as dt
import logging
import datetime as dt


class TestTicksDataCollector:
    """Test TicksDataCollector class (new architecture)"""

    def test_initialization(self):
        """Test TicksDataCollector initialization with dependency injection"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_headers = {'Authorization': 'Bearer token'}
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers=mock_headers,
            logger=mock_logger
        )

        assert collector.crsp_ticks == mock_crsp
        assert collector.alpaca_ticks == mock_alpaca
        assert collector.alpaca_headers == mock_headers
        assert collector.logger == mock_logger

    def test_collect_daily_ticks_year_crsp(self):
        """Test collecting daily ticks for year < 2025 (uses CRSP)"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.collect_daily_ticks.return_value = [
            {'timestamp': '2024-01-01', 'open': 100.0, 'close': 101.0, 'volume': 1000000}
        ]
        mock_alpaca = Mock()
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year('AAPL', 2024)

        # Should call CRSP for year < 2025
        assert mock_crsp.collect_daily_ticks.called
        assert not mock_alpaca.fetch_daily_year_bulk.called

    def test_collect_daily_ticks_year_alpaca(self):
        """Test collecting daily ticks for year >= 2025 (uses Alpaca)"""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickDataPoint

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {
            "AAPL": [{
                "t": "2025-01-01T05:00:00Z",
                "o": 100.0,
                "h": 102.0,
                "l": 99.0,
                "c": 101.0,
                "v": 1000000,
                "n": 5000,
                "vw": 100.5
            }]
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-01-01T00:00:00",
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000,
                num_trades=5000,
                vwap=100.5
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year('AAPL', 2025)

        # Should call Alpaca for year >= 2025
        assert not mock_crsp.collect_daily_ticks.called
        mock_alpaca.fetch_daily_year_bulk.assert_called_once_with(
            symbols=['AAPL'],
            year=2025,
            adjusted=True
        )
        assert len(result) == 1
        assert result["close"][0] == 101.0

    def test_fetch_minute_month(self):
        """Test fetching minute data for month"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_alpaca = Mock()
        mock_alpaca.fetch_minute_month_bulk.return_value = {
            'AAPL': [{'t': '2024-01-01T09:30:00Z', 'o': 100.0, 'c': 101.0}]
        }
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.fetch_minute_month(['AAPL'], 2024, 1, sleep_time=0.2)

        mock_alpaca.fetch_minute_month_bulk.assert_called_once_with(['AAPL'], 2024, 1, 0.2)
        assert 'AAPL' in result

    def test_collect_daily_ticks_year_crsp_skips_inactive_months(self):
        """CRSP path skips inactive months and formats output."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_logger = Mock(spec=logging.Logger)

        def _month_side_effect(*args, **kwargs):
            if kwargs.get("month") == 2:
                raise ValueError("not active on")
            return [{"timestamp": "2024-01-31", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]

        mock_crsp.collect_daily_ticks.side_effect = _month_side_effect

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year("BRK.B", 2024)

        assert len(result) > 0
        assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(result.columns)

    def test_collect_daily_ticks_year_alpaca_failure(self):
        """Alpaca path returns empty on exceptions."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.side_effect = Exception("boom")
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year("AAPL", 2025)

        assert result.is_empty()

    def test_collect_daily_ticks_year_bulk_alpaca(self):
        """Bulk Alpaca year fetch returns normalized DataFrames."""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickDataPoint

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {
            "AAPL": [{
                "t": "2025-01-02T05:00:00Z",
                "o": 150.0,
                "h": 155.0,
                "l": 149.0,
                "c": 154.0,
                "v": 1000,
                "n": 5,
                "vw": 153.0
            }],
            "MSFT": []
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-01-02T00:00:00",
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=1000,
                num_trades=5,
                vwap=153.0
            )
        ]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year_bulk(["AAPL", "MSFT"], 2025)

        mock_alpaca.fetch_daily_year_bulk.assert_called_once_with(["AAPL", "MSFT"], 2025, adjusted=True)
        assert result["AAPL"]["close"][0] == 154.0
        assert result["MSFT"].is_empty()

    def test_parse_minute_bars_to_daily_empty(self):
        """Empty bars -> empty frames for each day."""
        from quantdl.storage.data_collectors import TicksDataCollector

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )
        result = collector.parse_minute_bars_to_daily({"AAPL": []}, ["2024-06-03", "2024-06-04"])

        assert result[("AAPL", "2024-06-03")].is_empty()
        assert result[("AAPL", "2024-06-04")].is_empty()

    def test_parse_minute_bars_to_daily_success(self):
        """Valid bars are split by trade day."""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickField

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )
        bars = [
            {
                TickField.TIMESTAMP.value: "2024-06-03T14:30:00Z",
                TickField.OPEN.value: 1.0,
                TickField.HIGH.value: 1.1,
                TickField.LOW.value: 0.9,
                TickField.CLOSE.value: 1.0,
                TickField.VOLUME.value: 100,
                TickField.NUM_TRADES.value: 5,
                TickField.VWAP.value: 1.02
            },
            {
                TickField.TIMESTAMP.value: "2024-06-04T14:30:00Z",
                TickField.OPEN.value: 2.0,
                TickField.HIGH.value: 2.1,
                TickField.LOW.value: 1.9,
                TickField.CLOSE.value: 2.0,
                TickField.VOLUME.value: 200,
                TickField.NUM_TRADES.value: 10,
                TickField.VWAP.value: 2.02
            }
        ]

        result = collector.parse_minute_bars_to_daily({"AAPL": bars}, ["2024-06-03", "2024-06-04"])

        assert len(result[("AAPL", "2024-06-03")]) == 1
        assert len(result[("AAPL", "2024-06-04")]) == 1

    def test_parse_minute_bars_to_daily_error(self):
        """Invalid bars return empty frames and log error."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )
        result = collector.parse_minute_bars_to_daily({"AAPL": [{"bad": "data"}]}, ["2024-06-03"])

        assert result[("AAPL", "2024-06-03")].is_empty()
        mock_logger.error.assert_called()

    def test_collect_daily_ticks_month_filters_correctly(self):
        """Test that collect_daily_ticks_month calls month-specific API for Alpaca (2025+)"""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickDataPoint

        mock_crsp = Mock()
        mock_alpaca = Mock()

        # Mock Alpaca's get_daily() to return June data only
        june_bars = [
            {
                "t": "2025-06-30T04:00:00Z",
                "o": 103.0,
                "h": 108.0,
                "l": 102.0,
                "c": 107.0,
                "v": 1300000,
                "n": 1000,
                "vw": 105.5
            }
        ]
        mock_alpaca.fetch_daily_month_bulk.return_value = {"AAPL": june_bars}
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-06-30T00:00:00",
                open=103.0,
                high=108.0,
                low=102.0,
                close=107.0,
                volume=1300000,
                num_trades=1000,
                vwap=105.5
            )
        ]

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2025 (uses Alpaca)
        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        # Verify bulk month fetch
        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL"],
            year=2025,
            month=6,
            adjusted=True
        )
        mock_alpaca.parse_ticks.assert_called_once_with(june_bars)

        assert len(result) == 1
        assert result["timestamp"][0] == "2025-06-30"
        assert result["close"][0] == 107.0

    def test_collect_daily_ticks_month_uses_year_df(self):
        """Test that collect_daily_ticks_month filters from provided year_df."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30", "2024-07-01"],
            "open": [190.0, 191.0],
            "high": [195.0, 196.0],
            "low": [189.0, 190.0],
            "close": [193.0, 194.0],
            "volume": [50000000, 51000000]
        })

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=year_df)

        assert len(result) == 1
        assert result["timestamp"][0] == "2024-06-30"
        mock_crsp.collect_daily_ticks.assert_not_called()
        mock_alpaca.fetch_daily_month_bulk.assert_not_called()

    def test_collect_daily_ticks_month_empty_when_no_data(self):
        """Test that collect_daily_ticks_month returns empty when month has no data (Alpaca)"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()

        # Mock Alpaca's get_daily() to return empty data for requested month
        mock_alpaca.fetch_daily_month_bulk.return_value = {"AAPL": []}

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2025 which has no data
        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        # Verify bulk month fetch
        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL"],
            year=2025,
            month=6,
            adjusted=True
        )

        assert result.is_empty()

    def test_collect_daily_ticks_month_bulk_alpaca(self):
        """Bulk month fetch returns normalized DataFrames for Alpaca years."""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickDataPoint

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_month_bulk.return_value = {
            "AAPL": [{
                "t": "2025-06-30T05:00:00Z",
                "o": 200.0,
                "h": 210.0,
                "l": 195.0,
                "c": 205.0,
                "v": 2000,
                "n": 10,
                "vw": 204.0
            }],
            "MSFT": []
        }
        mock_alpaca.parse_ticks.return_value = [
            TickDataPoint(
                timestamp="2025-06-30T00:00:00",
                open=200.0,
                high=210.0,
                low=195.0,
                close=205.0,
                volume=2000,
                num_trades=10,
                vwap=204.0
            )
        ]

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        result = collector.collect_daily_ticks_month_bulk(["AAPL", "MSFT"], 2025, 6, sleep_time=0.0)

        mock_alpaca.fetch_daily_month_bulk.assert_called_once_with(
            symbols=["AAPL", "MSFT"],
            year=2025,
            month=6,
            sleep_time=0.0,
            adjusted=True
        )
        assert result["AAPL"]["close"][0] == 205.0
        assert result["MSFT"].is_empty()

    def test_collect_daily_ticks_month_crsp(self):
        """Test that collect_daily_ticks_month calls CRSP API for years < 2025"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()

        # Mock CRSP's collect_daily_ticks() to return month data
        june_data = [
            {
                "timestamp": "2024-06-30",
                "open": 190.0,
                "high": 195.0,
                "low": 189.0,
                "close": 193.0,
                "volume": 50000000
            }
        ]
        mock_crsp.collect_daily_ticks.return_value = june_data

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2024 (uses CRSP)
        result = collector.collect_daily_ticks_month("AAPL", 2024, 6)

        # Verify collect_daily_ticks() was called with month parameter
        mock_crsp.collect_daily_ticks.assert_called_once_with(
            symbol="AAPL",
            year=2024,
            month=6,
            adjusted=True,
            auto_resolve=True
        )

        assert len(result) == 1
        assert result["timestamp"][0] == "2024-06-30"

    def test_collect_daily_ticks_month_crsp_empty(self):
        """Test that collect_daily_ticks_month returns empty when CRSP has no data"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()

        # Mock CRSP's collect_daily_ticks() to return empty data
        mock_crsp.collect_daily_ticks.return_value = []

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Test June (month=6) for year 2024 which has no data
        result = collector.collect_daily_ticks_month("AAPL", 2024, 6)

        # Verify collect_daily_ticks() was called with month parameter
        mock_crsp.collect_daily_ticks.assert_called_once_with(
            symbol="AAPL",
            year=2024,
            month=6,
            adjusted=True,
            auto_resolve=True
        )

        assert result.is_empty()

    def test_collect_daily_ticks_month_crsp_not_active(self):
        """Test that collect_daily_ticks_month handles 'not active' errors from CRSP"""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()

        # Mock CRSP to raise ValueError with "not active on" message
        mock_crsp.collect_daily_ticks.side_effect = ValueError("AAPL not active on 2024-06")

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        # Should return empty DataFrame without raising error
        result = collector.collect_daily_ticks_month("AAPL", 2024, 6)

        assert result.is_empty()


class TestFundamentalDataCollector:
    """Test FundamentalDataCollector class (refactored)"""

    def test_initialization_with_logger(self):
        """Test FundamentalDataCollector initialization with logger"""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        assert collector.logger == mock_logger
        assert isinstance(collector._fundamental_cache, OrderedDict)
        assert isinstance(collector._fundamental_cache_lock, type(threading.Lock()))

    @patch('quantdl.storage.data_collectors.setup_logger')
    def test_initialization_without_logger(self, mock_setup_logger):
        """Test FundamentalDataCollector creates logger if not provided"""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        mock_setup_logger.return_value = mock_logger

        collector = FundamentalDataCollector()

        mock_setup_logger.assert_called_once()
        assert collector.logger == mock_logger

    def test_shared_cache_initialization(self):
        """Test FundamentalDataCollector uses shared cache when provided"""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        shared_cache = OrderedDict()
        shared_lock = threading.Lock()
        mock_logger = Mock(spec=logging.Logger)

        collector = FundamentalDataCollector(
            logger=mock_logger,
            fundamental_cache=shared_cache,
            fundamental_cache_lock=shared_lock
        )

        # Should use the shared cache
        assert collector._fundamental_cache is shared_cache
        assert collector._fundamental_cache_lock is shared_lock

    def test_load_concepts_with_provided_list(self):
        """Test _load_concepts returns provided list"""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        concepts = ['Revenue', 'Assets', 'NetIncome']
        result = collector._load_concepts(concepts=concepts)

        assert result == concepts

    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_load_concepts_from_config(self, mock_yaml_load, mock_open):
        """Test _load_concepts loads from config file"""
        from quantdl.storage.data_collectors import FundamentalDataCollector
        from pathlib import Path

        mock_yaml_load.return_value = {'Revenue': 'mapping1', 'Assets': 'mapping2'}
        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        result = collector._load_concepts()

        assert 'Revenue' in result
        assert 'Assets' in result

    def test_get_or_create_fundamental_cache(self):
        """Cache hit returns same object; LRU eviction occurs."""
        from quantdl.storage import data_collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger, fundamental_cache_size=1)

        with patch('quantdl.storage.data_collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            f3 = Mock()
            mock_fundamental.side_effect = [f1, f2, f3]

            one = collector._get_or_create_fundamental("0001")
            two = collector._get_or_create_fundamental("0002")
            again = collector._get_or_create_fundamental("0001")

        assert one is f1
        assert two is f2
        assert again is f3

    def test_collect_fundamental_long_records(self):
        """Builds records from concept data within date range."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"
        dp.is_instant = False

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert len(result) == 1
        assert result.select("concept").item() == "rev"

    def test_collect_fundamental_long_handles_concept_error(self):
        """Errors in one concept don't block others."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"
        dp.is_instant = False

        fund = Mock()
        fund.get_concept_data.side_effect = [Exception("bad"), [dp]]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["bad", "rev"]
        )

        assert len(result) == 1
        assert result.select("concept").item() == "rev"

    def test_collect_ttm_long_range_filters_dates(self):
        """TTM data is filtered to date range."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        ttm_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["rev"],
            "value": [123.0]
        })

        with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=ttm_df):
            result = collector.collect_ttm_long_range(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev"]
            )

        assert len(result) == 1

    def test_build_metrics_wide_no_ttm_records(self):
        """No duration concept records returns empty output."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))

        fund = Mock()
        fund.get_concept_data.return_value = []
        collector._get_or_create_fundamental = Mock(return_value=fund)

        with patch('quantdl.storage.data_collectors.DURATION_CONCEPTS', ["rev"]):
            metrics_df, metadata = collector._build_metrics_wide(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev"]
            )

        assert metrics_df.is_empty()
        assert metadata is None

    def test_build_metrics_wide_raw_records_empty(self):
        """No stock concepts in range still returns duration data with nulls."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        ttm_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["rev"],
            "value": [123.0],
            "accn": ["0001"],
            "form": ["10-Q"],
            "frame": ["CY2024Q2"]
        })

        with patch('quantdl.storage.data_collectors.DURATION_CONCEPTS', ["rev"]):
            with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=ttm_df):
                metrics_df, metadata = collector._build_metrics_wide(
                    cik="0001",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    symbol="AAPL",
                    concepts=["rev", "eps"]
                )

        assert "rev" in metrics_df.columns
        assert "eps" in metrics_df.columns

    def test_collect_derived_long_derived_empty(self):
        """Derived empty returns reason."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        collector._build_metrics_wide = Mock(return_value=(pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "rev": [100.0]
        }), None))

        with patch('quantdl.storage.data_collectors.compute_derived', return_value=pl.DataFrame()):
            result, reason = collector.collect_derived_long(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev"]
            )

        assert result.is_empty()
        assert reason == "derived_empty"

    def test_collect_derived_long_empty_metrics(self):
        """Derived returns reason when metrics_wide is empty."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        collector._build_metrics_wide = Mock(return_value=(pl.DataFrame(), None))

        result, reason = collector.collect_derived_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert result.is_empty()
        assert reason == "metrics_wide_empty"


class TestUniverseDataCollector:
    """Test UniverseDataCollector class (enhanced)"""

    def test_initialization_with_logger(self):
        """Test UniverseDataCollector initialization with logger"""
        from quantdl.storage.data_collectors import UniverseDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)

        assert collector.logger == mock_logger

    @patch('quantdl.storage.data_collectors.setup_logger')
    def test_initialization_without_logger(self, mock_setup_logger):
        """Test UniverseDataCollector creates logger if not provided"""
        from quantdl.storage.data_collectors import UniverseDataCollector

        mock_logger = Mock(spec=logging.Logger)
        mock_setup_logger.return_value = mock_logger

        collector = UniverseDataCollector()

        mock_setup_logger.assert_called_once()
        assert collector.logger == mock_logger

    @patch('quantdl.storage.data_collectors.fetch_all_stocks')
    def test_collect_current_universe(self, mock_fetch):
        """Test collecting current universe"""
        from quantdl.storage.data_collectors import UniverseDataCollector

        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Company Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.']
        })
        mock_fetch.return_value = mock_stocks

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)
        result = collector.collect_current_universe()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        mock_fetch.assert_called_with(with_filter=True, refresh=False)

    @patch('quantdl.storage.data_collectors.fetch_all_stocks')
    def test_collect_universe_with_filter(self, mock_fetch):
        """Test collecting universe with filter"""
        from quantdl.storage.data_collectors import UniverseDataCollector

        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'MSFT'],
            'Company Name': ['Apple Inc.', 'Microsoft Corp.']
        })
        mock_fetch.return_value = mock_stocks

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)
        result = collector.collect_current_universe(with_filter=True, refresh=True)

        assert isinstance(result, pl.DataFrame)
        mock_fetch.assert_called_with(with_filter=True, refresh=True)


class TestDataCollectorsOrchestrator:
    """Test DataCollectors orchestrator (delegation pattern)"""

    def test_initialization_creates_specialized_collectors(self):
        """Test DataCollectors creates all specialized collectors"""
        from quantdl.storage.data_collectors import DataCollectors

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_headers = {}
        mock_logger = Mock(spec=logging.Logger)

        orchestrator = DataCollectors(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers=mock_headers,
            logger=mock_logger
        )

        # Verify specialized collectors were created
        assert hasattr(orchestrator, 'ticks_collector')
        assert hasattr(orchestrator, 'fundamental_collector')
        assert hasattr(orchestrator, 'universe_collector')

    def test_shared_fundamental_cache(self):
        """Test DataCollectors creates shared fundamental cache"""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger,
            fundamental_cache_size=256
        )

        # Orchestrator should have the cache
        assert hasattr(orchestrator, '_fundamental_cache')
        assert hasattr(orchestrator, '_fundamental_cache_lock')

        # Fundamental collector should share the cache
        assert orchestrator.fundamental_collector._fundamental_cache is orchestrator._fundamental_cache
        assert orchestrator.fundamental_collector._fundamental_cache_lock is orchestrator._fundamental_cache_lock

    def test_delegation_to_ticks_collector(self):
        """Test DataCollectors delegates to TicksDataCollector"""
        from quantdl.storage.data_collectors import DataCollectors

        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_year_bulk.return_value = {}
        mock_logger = Mock(spec=logging.Logger)

        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        # Call delegated method
        orchestrator.collect_daily_ticks_year('AAPL', 2025)

        # Should delegate to ticks_collector
        mock_alpaca.fetch_daily_year_bulk.assert_called_once()

    def test_delegation_to_ticks_collector_month_bulk(self):
        """Delegates bulk month fetch to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_month_bulk = Mock(return_value={})

        orchestrator.collect_daily_ticks_month_bulk(["AAPL"], 2025, 6, sleep_time=0.1)

        orchestrator.ticks_collector.collect_daily_ticks_month_bulk.assert_called_once_with(
            ["AAPL"], 2025, 6, 0.1
        )

    def test_delegation_to_fundamental_collector(self):
        """Test DataCollectors delegates to FundamentalDataCollector"""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        # Mock the fundamental collector's method
        orchestrator.fundamental_collector.collect_fundamental_long = Mock(return_value=pl.DataFrame())

        # Call delegated method
        result = orchestrator.collect_fundamental_long('320193', '2024-01-01', '2024-12-31', 'AAPL')

        # Should delegate to fundamental_collector
        orchestrator.fundamental_collector.collect_fundamental_long.assert_called_once_with(
            '320193', '2024-01-01', '2024-12-31', 'AAPL', None, None
        )

    @patch('quantdl.storage.data_collectors.fetch_all_stocks')
    def test_backward_compatibility(self, mock_fetch):
        """Test DataCollectors maintains backward compatibility"""
        from quantdl.storage.data_collectors import DataCollectors

        mock_fetch.return_value = pl.DataFrame({'Ticker': ['AAPL']})
        mock_logger = Mock(spec=logging.Logger)

        # Should work with same constructor as before
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger,
            sec_rate_limiter=None,
            fundamental_cache_size=128
        )

        # All old methods should still work
        assert hasattr(orchestrator, 'collect_daily_ticks_year')
        assert hasattr(orchestrator, 'collect_fundamental_long')
        assert hasattr(orchestrator, 'collect_ttm_long_range')
        assert hasattr(orchestrator, 'collect_derived_long')
        assert hasattr(orchestrator, '_load_concepts')


class TestDataCollectorInheritance:
    """Test that all collectors properly inherit from DataCollector ABC"""

    def test_ticks_collector_inherits_datacollector(self):
        """Test TicksDataCollector inherits from DataCollector"""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')

    def test_fundamental_collector_inherits_datacollector(self):
        """Test FundamentalDataCollector inherits from DataCollector"""
        from quantdl.storage.data_collectors import FundamentalDataCollector
        from quantdl.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')

    def test_universe_collector_inherits_datacollector(self):
        """Test UniverseDataCollector inherits from DataCollector"""
        from quantdl.storage.data_collectors import UniverseDataCollector
        from quantdl.collection.models import DataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = UniverseDataCollector(logger=mock_logger)

        assert isinstance(collector, DataCollector)
        assert hasattr(collector, 'logger')
