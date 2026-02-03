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

    def test_collect_daily_ticks_year_crsp_raises_on_other_errors(self):
        """CRSP path re-raises unexpected errors."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.collect_daily_ticks.side_effect = ValueError("boom")
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        with pytest.raises(ValueError, match="boom"):
            collector.collect_daily_ticks_year("AAPL", 2024)

    def test_collect_daily_ticks_year_crsp_empty_returns_empty(self):
        """CRSP path returns empty DataFrame when no months return data."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.collect_daily_ticks.side_effect = [[] for _ in range(12)]
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year("AAPL", 2024)

        assert result.is_empty()

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

    def test_fetch_minute_day(self):
        """Test fetching minute data for day."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_alpaca = Mock()
        mock_alpaca.fetch_minute_day_bulk.return_value = {"AAPL": []}
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.fetch_minute_day(["AAPL"], "2024-01-02", sleep_time=0.1)

        mock_alpaca.fetch_minute_day_bulk.assert_called_once_with(["AAPL"], "2024-01-02", 0.1)
        assert "AAPL" in result

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

    def test_collect_daily_ticks_year_bulk_crsp_delegates(self):
        """CRSP bulk year fetch delegates and returns mapping."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.collect_daily_ticks_year_bulk.return_value = {"AAPL": pl.DataFrame()}
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_year_bulk(["AAPL"], 2024)

        mock_crsp.collect_daily_ticks_year_bulk.assert_called_once_with(
            ["AAPL"], 2024, adjusted=True, auto_resolve=True
        )
        assert "AAPL" in result

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

    def test_parse_minute_bars_to_daily_missing_day_empty(self):
        """Days without data return empty DataFrames."""
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
            }
        ]

        result = collector.parse_minute_bars_to_daily({"AAPL": bars}, ["2024-06-03", "2024-06-04"])

        assert len(result[("AAPL", "2024-06-03")]) == 1
        assert result[("AAPL", "2024-06-04")].is_empty()

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

    def test_parse_minute_bars_to_daily_nanosecond_timestamps(self):
        """Alpaca returns RFC-3339 timestamps with nanoseconds - verify parsing works."""
        from quantdl.storage.data_collectors import TicksDataCollector
        from quantdl.collection.models import TickField

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )
        # Alpaca returns timestamps with nanosecond precision
        bars = [
            {
                TickField.TIMESTAMP.value: "2024-06-03T14:30:00.123456789Z",
                TickField.OPEN.value: 1.0,
                TickField.HIGH.value: 1.1,
                TickField.LOW.value: 0.9,
                TickField.CLOSE.value: 1.0,
                TickField.VOLUME.value: 100,
                TickField.NUM_TRADES.value: 5000,  # High num_trades = regular hours
                TickField.VWAP.value: 1.02
            },
            {
                TickField.TIMESTAMP.value: "2024-06-03T15:30:00.987654321Z",
                TickField.OPEN.value: 1.1,
                TickField.HIGH.value: 1.2,
                TickField.LOW.value: 1.0,
                TickField.CLOSE.value: 1.15,
                TickField.VOLUME.value: 200,
                TickField.NUM_TRADES.value: 8000,
                TickField.VWAP.value: 1.12
            }
        ]

        result = collector.parse_minute_bars_to_daily({"AAPL": bars}, ["2024-06-03"])

        # Both bars should be parsed (not dropped due to nanoseconds)
        assert len(result[("AAPL", "2024-06-03")]) == 2
        # Verify timestamps are converted to ET (14:30 UTC = 10:30 ET during summer)
        df = result[("AAPL", "2024-06-03")]
        assert df["num_trades"][0] == 5000
        assert df["num_trades"][1] == 8000

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

    def test_collect_daily_ticks_month_year_df_empty(self):
        """Year DF empty returns empty."""
        from quantdl.storage.data_collectors import TicksDataCollector

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=pl.DataFrame())

        assert result.is_empty()

    def test_collect_daily_ticks_month_year_df_filtered_empty(self):
        """Year DF with other months returns empty."""
        from quantdl.storage.data_collectors import TicksDataCollector

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-07-01"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })

        result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=year_df)

        assert result.is_empty()

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

    def test_collect_daily_ticks_month_crsp_other_error_raises(self):
        """Unexpected CRSP errors are re-raised."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.collect_daily_ticks.side_effect = ValueError("boom")

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        with pytest.raises(ValueError, match="boom"):
            collector.collect_daily_ticks_month("AAPL", 2024, 6)

    def test_collect_daily_ticks_month_alpaca_exception(self):
        """Alpaca path returns empty and logs warning on exception."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()
        mock_alpaca.fetch_daily_month_bulk.side_effect = RuntimeError("boom")
        mock_logger = Mock(spec=logging.Logger)

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=mock_logger
        )

        result = collector.collect_daily_ticks_month("AAPL", 2025, 6)

        assert result.is_empty()
        mock_logger.warning.assert_called()

    def test_collect_daily_ticks_month_year_df_aligns_calendar(self, tmp_path):
        """Year_df branch aligns calendar when calendar file exists."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.calendar_path = str(tmp_path / "calendar.csv")
        (tmp_path / "calendar.csv").write_text("date\n")
        mock_alpaca = Mock()

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        aligned = [{
            "timestamp": "2024-06-30",
            "open": 190.0,
            "high": 195.0,
            "low": 189.0,
            "close": 193.0,
            "volume": 50000000
        }]

        with patch('quantdl.storage.data_collectors.align_calendar', return_value=aligned) as mock_align:
            result = collector.collect_daily_ticks_month("AAPL", 2024, 6, year_df=year_df)

        assert result["timestamp"][0] == "2024-06-30"
        mock_align.assert_called()

    def test_collect_daily_ticks_month_year_df_aligns_december_end(self, tmp_path):
        """Calendar alignment uses December 31 as end date."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_crsp.calendar_path = str(tmp_path / "calendar.csv")
        (tmp_path / "calendar.csv").write_text("date\n")

        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        year_df = pl.DataFrame({
            "timestamp": ["2024-12-15"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })

        with patch('quantdl.storage.data_collectors.align_calendar', return_value=year_df.to_dicts()) as mock_align:
            collector.collect_daily_ticks_month("AAPL", 2024, 12, year_df=year_df)

        call_args = mock_align.call_args[0]
        assert call_args[1] == dt.date(2024, 12, 1)
        assert call_args[2] == dt.date(2024, 12, 31)

    def test_collect_daily_ticks_month_bulk_crsp(self):
        """Bulk month fetch uses per-symbol CRSP path for years < 2025."""
        from quantdl.storage.data_collectors import TicksDataCollector

        mock_crsp = Mock()
        mock_alpaca = Mock()
        collector = TicksDataCollector(
            crsp_ticks=mock_crsp,
            alpaca_ticks=mock_alpaca,
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        collector.collect_daily_ticks_month = Mock(return_value=df)

        result = collector.collect_daily_ticks_month_bulk(["AAPL", "MSFT"], 2024, 6)

        assert result["AAPL"]["close"][0] == 193.0
        assert result["MSFT"]["close"][0] == 193.0
        assert collector.collect_daily_ticks_month.call_count == 2
        mock_alpaca.fetch_daily_month_bulk.assert_not_called()

    def test_normalize_daily_df_adds_missing_columns(self):
        """Missing columns are added and types normalized."""
        from quantdl.storage.data_collectors import TicksDataCollector

        collector = TicksDataCollector(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=Mock(spec=logging.Logger)
        )

        df = pl.DataFrame({
            "timestamp": ["2024-01-02"],
            "close": [101.123456]
        })

        result = collector._normalize_daily_df(df)

        for col in ["open", "high", "low", "volume"]:
            assert col in result.columns
        assert result["close"][0] == 101.1235


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

    def test_get_or_create_fundamental_no_cache(self):
        """Cache size <= 0 returns new instance each time."""
        from quantdl.storage import data_collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger, fundamental_cache_size=0)

        with patch('quantdl.storage.data_collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            mock_fundamental.side_effect = [f1, f2]

            one = collector._get_or_create_fundamental("0001")
            two = collector._get_or_create_fundamental("0001")

        assert one is f1
        assert two is f2
        assert len(collector._fundamental_cache) == 0

    def test_get_or_create_fundamental_cache_hit(self):
        """Cache hit returns existing instance without creating new one."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        cached = Mock()
        collector._fundamental_cache = OrderedDict([("0001", cached)])

        with patch('quantdl.storage.data_collectors.Fundamental') as mock_fundamental:
            result = collector._get_or_create_fundamental("0001")

        assert result is cached
        mock_fundamental.assert_not_called()

    def test_get_or_create_fundamental_cache_hit_moves_to_end(self):
        """Cache hit moves item to end (LRU behavior) - covers lines 466-467."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger), fundamental_cache_size=3)
        cached1 = Mock()
        cached2 = Mock()
        cached3 = Mock()

        # Populate cache with 3 items
        collector._fundamental_cache = OrderedDict([
            ("0001", cached1),
            ("0002", cached2),
            ("0003", cached3)
        ])

        with patch('quantdl.storage.data_collectors.Fundamental') as mock_fundamental:
            # Access the first item
            result = collector._get_or_create_fundamental("0001")

        # Should return cached item
        assert result is cached1
        mock_fundamental.assert_not_called()

        # Item should be moved to end (most recently used)
        keys_list = list(collector._fundamental_cache.keys())
        assert keys_list == ["0002", "0003", "0001"]

    def test_get_or_create_fundamental_evicts_oldest(self):
        """Cache evicts oldest when capacity exceeded."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger), fundamental_cache_size=1)

        with patch('quantdl.storage.data_collectors.Fundamental') as mock_fundamental:
            f1 = Mock()
            f2 = Mock()
            mock_fundamental.side_effect = [f1, f2]

            collector._get_or_create_fundamental("0001")
            collector._get_or_create_fundamental("0002")

        assert "0001" not in collector._fundamental_cache
        assert "0002" in collector._fundamental_cache

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

    def test_collect_fundamental_long_no_records(self):
        """No records returns empty and logs warning."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        fund = Mock()
        fund.get_concept_data.return_value = []
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert result.is_empty()
        mock_logger.warning.assert_called_once()

    def test_collect_fundamental_long_outer_exception(self):
        """Outer exception returns empty and logs error."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        collector._load_concepts = Mock(side_effect=ValueError("bad"))

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL"
        )

        assert result.is_empty()
        mock_logger.error.assert_called_once()

    def test_collect_fundamental_long_filters_out_of_range_records(self):
        """Records outside date range are skipped - covers line 511."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))

        # Create data points - one in range, one out of range
        dp_in_range = Mock()
        dp_in_range.timestamp = dt.date(2024, 6, 30)
        dp_in_range.accn = "0001"
        dp_in_range.form = "10-Q"
        dp_in_range.value = 123.0
        dp_in_range.start_date = dt.date(2024, 4, 1)
        dp_in_range.end_date = dt.date(2024, 6, 30)
        dp_in_range.frame = "CY2024Q2"
        dp_in_range.is_instant = False

        dp_out_of_range = Mock()
        dp_out_of_range.timestamp = dt.date(2025, 3, 31)  # Outside the end_date
        dp_out_of_range.accn = "0002"
        dp_out_of_range.form = "10-Q"
        dp_out_of_range.value = 456.0
        dp_out_of_range.start_date = dt.date(2025, 1, 1)
        dp_out_of_range.end_date = dt.date(2025, 3, 31)
        dp_out_of_range.frame = "CY2025Q1"
        dp_out_of_range.is_instant = False

        fund = Mock()
        fund.get_concept_data.return_value = [dp_out_of_range, dp_in_range]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_fundamental_long(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        # Should only include the in-range record
        assert len(result) == 1
        assert result.select("accn").item() == "0001"

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

    def test_collect_ttm_long_range_skips_after_end_date(self):
        """Records after end_date are skipped leading to empty output."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        dp = Mock()
        dp.timestamp = dt.date(2025, 1, 1)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 10, 1)
        dp.end_date = dt.date(2024, 12, 31)
        dp.frame = "CY2024Q4"

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_ttm_long_range(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert result.is_empty()
        mock_logger.warning.assert_called_once()

    def test_collect_ttm_long_range_concept_error(self):
        """Concept extraction errors are logged and skipped."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        fund = Mock()
        fund.get_concept_data.side_effect = Exception("bad")
        collector._get_or_create_fundamental = Mock(return_value=fund)

        result = collector.collect_ttm_long_range(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert result.is_empty()
        mock_logger.debug.assert_called()

    def test_collect_ttm_long_range_empty_ttm_df(self):
        """Empty TTM result returns empty DataFrame."""
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

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=pl.DataFrame()):
            result = collector.collect_ttm_long_range(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev"]
            )

        assert result.is_empty()

    def test_collect_ttm_long_range_outer_exception(self):
        """Outer exception returns empty and logs error."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)
        collector._load_concepts = Mock(side_effect=ValueError("bad"))

        result = collector.collect_ttm_long_range(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL"
        )

        assert result.is_empty()
        mock_logger.error.assert_called_once()

    def test_collect_ttm_long_range_skips_empty_concept_data(self):
        """Empty concept data is skipped - covers line 577."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(logger=mock_logger)

        dp = Mock()
        dp.timestamp = dt.date(2024, 6, 30)
        dp.accn = "0001"
        dp.form = "10-Q"
        dp.value = 123.0
        dp.start_date = dt.date(2024, 4, 1)
        dp.end_date = dt.date(2024, 6, 30)
        dp.frame = "CY2024Q2"

        fund = Mock()
        # First concept returns no data (empty list), second returns data
        fund.get_concept_data.side_effect = [[], [dp]]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        ttm_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["eps"],
            "value": [123.0]
        })

        with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=ttm_df):
            result = collector.collect_ttm_long_range(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev", "eps"]  # rev has no data, eps has data
            )

        # Should only include data from the second concept
        assert len(result) == 1
        assert result.select("concept").item() == "eps"

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

    def test_build_metrics_wide_fundamental_init_error(self):
        """Initialization failure returns empty output and None metadata."""
        from quantdl.storage import data_collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger)
        collector._get_or_create_fundamental = Mock(side_effect=RuntimeError("boom"))

        metrics_df, metadata = collector._build_metrics_wide(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert metrics_df.is_empty()
        assert metadata is None
        mock_logger.error.assert_called_once()

    def test_build_metrics_wide_handles_concept_error(self):
        """Concept errors are logged and skipped."""
        from quantdl.storage import data_collectors as dc

        mock_logger = Mock(spec=logging.Logger)
        collector = dc.FundamentalDataCollector(logger=mock_logger)
        fund = Mock()
        fund.get_concept_data.side_effect = Exception("bad")
        collector._get_or_create_fundamental = Mock(return_value=fund)

        metrics_df, metadata = collector._build_metrics_wide(
            cik="0001",
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbol="AAPL",
            concepts=["rev"]
        )

        assert metrics_df.is_empty()
        assert metadata is None
        mock_logger.debug.assert_called()

    def test_build_metrics_wide_empty_ttm_df(self):
        """Empty TTM output returns empty metrics."""
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

        with patch('quantdl.storage.data_collectors.DURATION_CONCEPTS', ["rev"]):
            with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=pl.DataFrame()):
                metrics_df, metadata = collector._build_metrics_wide(
                    cik="0001",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    symbol="AAPL",
                    concepts=["rev"]
                )

        assert metrics_df.is_empty()
        assert metadata is None

    def test_build_metrics_wide_filters_out_of_range(self):
        """TTM data outside date range returns empty."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp = Mock()
        dp.timestamp = dt.date(2023, 12, 31)
        dp.accn = "0001"
        dp.form = "10-K"
        dp.value = 123.0
        dp.start_date = dt.date(2023, 1, 1)
        dp.end_date = dt.date(2023, 12, 31)
        dp.frame = "CY2023"

        fund = Mock()
        fund.get_concept_data.return_value = [dp]
        collector._get_or_create_fundamental = Mock(return_value=fund)

        ttm_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2023-12-31"],
            "concept": ["rev"],
            "value": [123.0],
            "accn": ["0001"],
            "form": ["10-K"],
            "frame": ["CY2023"]
        })

        with patch('quantdl.storage.data_collectors.DURATION_CONCEPTS', ["rev"]):
            with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=ttm_df):
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

    def test_build_metrics_wide_missing_duration_columns(self):
        """Missing duration columns are added with nulls."""
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

        with patch('quantdl.storage.data_collectors.DURATION_CONCEPTS', ["rev", "eps"]):
            with patch('quantdl.storage.data_collectors.compute_ttm_long', return_value=ttm_df):
                metrics_df, metadata = collector._build_metrics_wide(
                    cik="0001",
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    symbol="AAPL",
                    concepts=["rev", "eps"]
                )

        assert "eps" in metrics_df.columns

    def test_build_metrics_wide_missing_stock_columns_added(self):
        """Missing stock columns are added when not present after join."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        dp_duration = Mock()
        dp_duration.timestamp = dt.date(2024, 6, 30)
        dp_duration.accn = "0001"
        dp_duration.form = "10-Q"
        dp_duration.value = 123.0
        dp_duration.start_date = dt.date(2024, 4, 1)
        dp_duration.end_date = dt.date(2024, 6, 30)
        dp_duration.frame = "CY2024Q2"

        dp_stock = Mock()
        dp_stock.timestamp = dt.date(2024, 6, 30)
        dp_stock.accn = "0001"
        dp_stock.form = "10-Q"
        dp_stock.value = 5.0
        dp_stock.start_date = None
        dp_stock.end_date = dt.date(2024, 6, 30)
        dp_stock.frame = "CY2024Q2"

        def _concept_data(concept):
            if concept == "rev":
                return [dp_duration]
            if concept == "eps":
                return [dp_stock]
            return []

        fund = Mock()
        fund.get_concept_data.side_effect = _concept_data
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
                    concepts=["rev", "eps", "pe"]
                )

        assert "pe" in metrics_df.columns

    def test_build_metrics_wide_adds_missing_stock_cols_when_raw_empty(self):
        """When raw_long is empty, missing stock columns are added with nulls - covers lines 751-754."""
        from quantdl.storage import data_collectors as dc

        collector = dc.FundamentalDataCollector(logger=Mock(spec=logging.Logger))

        # Create duration data point
        dp_duration = Mock()
        dp_duration.timestamp = dt.date(2024, 6, 30)
        dp_duration.accn = "0001"
        dp_duration.form = "10-Q"
        dp_duration.value = 123.0
        dp_duration.start_date = dt.date(2024, 4, 1)
        dp_duration.end_date = dt.date(2024, 6, 30)
        dp_duration.frame = "CY2024Q2"

        def _concept_data(concept):
            if concept == "rev":
                return [dp_duration]
            # Stock concepts (eps, pe) return no data
            return []

        fund = Mock()
        fund.get_concept_data.side_effect = _concept_data
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
                    concepts=["rev", "eps", "pe"]
                )

        # Stock columns should be added with null values
        assert "eps" in metrics_df.columns
        assert "pe" in metrics_df.columns
        # Values should be null since no raw data was provided
        assert metrics_df.select("eps").item() is None
        assert metrics_df.select("pe").item() is None

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

    def test_collect_derived_long_filters_dates(self):
        """Derived output is filtered to date range."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        collector = FundamentalDataCollector(logger=Mock(spec=logging.Logger))
        collector._build_metrics_wide = Mock(return_value=(pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "as_of_date": ["2024-01-01", "2024-12-31"],
            "rev": [1.0, 2.0]
        }), None))

        derived_df = pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "as_of_date": ["2024-01-01", "2025-01-01"],
            "metric": ["m1", "m2"],
            "value": [1.0, 2.0]
        })

        with patch('quantdl.storage.data_collectors.compute_derived', return_value=derived_df):
            result, reason = collector.collect_derived_long(
                cik="0001",
                start_date="2024-01-01",
                end_date="2024-12-31",
                symbol="AAPL",
                concepts=["rev"]
            )

        assert reason is None
        assert len(result) == 1


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

    def test_delegation_to_ticks_collector_month(self):
        """Delegates single month fetch to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_month = Mock(return_value=pl.DataFrame())

        orchestrator.collect_daily_ticks_month("AAPL", 2024, 6)

        orchestrator.ticks_collector.collect_daily_ticks_month.assert_called_once_with(
            "AAPL", 2024, 6, year_df=None
        )

    def test_delegation_to_ticks_collector_year_bulk(self):
        """Delegates yearly bulk fetch to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.collect_daily_ticks_year_bulk = Mock(return_value={})

        orchestrator.collect_daily_ticks_year_bulk(["AAPL", "MSFT"], 2024)

        orchestrator.ticks_collector.collect_daily_ticks_year_bulk.assert_called_once_with(
            ["AAPL", "MSFT"], 2024
        )

    def test_delegation_to_ticks_collector_fetch_minute_month(self):
        """Delegates minute month fetch to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.fetch_minute_month = Mock(return_value={})

        orchestrator.fetch_minute_month(["AAPL"], 2024, 6, sleep_time=0.1)

        orchestrator.ticks_collector.fetch_minute_month.assert_called_once_with(
            ["AAPL"], 2024, 6, 0.1
        )

    def test_delegation_to_ticks_collector_fetch_minute_day(self):
        """Delegates minute day fetch to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.fetch_minute_day = Mock(return_value={})

        orchestrator.fetch_minute_day(["AAPL"], "2024-06-03", sleep_time=0.1)

        orchestrator.ticks_collector.fetch_minute_day.assert_called_once_with(
            ["AAPL"], "2024-06-03", 0.1
        )

    def test_delegation_to_ticks_collector_parse_minute_bars(self):
        """Delegates minute bars parsing to TicksDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.ticks_collector.parse_minute_bars_to_daily = Mock(return_value={})

        orchestrator.parse_minute_bars_to_daily({"AAPL": []}, ["2024-06-03"])

        orchestrator.ticks_collector.parse_minute_bars_to_daily.assert_called_once_with(
            {"AAPL": []}, ["2024-06-03"]
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

    def test_delegation_to_fundamental_collector_load_concepts(self):
        """Delegates _load_concepts to FundamentalDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.fundamental_collector._load_concepts = Mock(return_value=["rev"])

        result = orchestrator._load_concepts(concepts=["rev"], config_path=None)

        assert result == ["rev"]
        orchestrator.fundamental_collector._load_concepts.assert_called_once_with(["rev"], None)

    def test_delegation_to_fundamental_collector_ttm(self):
        """Delegates collect_ttm_long_range to FundamentalDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.fundamental_collector.collect_ttm_long_range = Mock(return_value=pl.DataFrame())

        orchestrator.collect_ttm_long_range('320193', '2024-01-01', '2024-12-31', 'AAPL')

        orchestrator.fundamental_collector.collect_ttm_long_range.assert_called_once_with(
            '320193', '2024-01-01', '2024-12-31', 'AAPL', None, None
        )

    def test_delegation_to_fundamental_collector_derived(self):
        """Delegates collect_derived_long to FundamentalDataCollector."""
        from quantdl.storage.data_collectors import DataCollectors

        mock_logger = Mock(spec=logging.Logger)
        orchestrator = DataCollectors(
            crsp_ticks=Mock(),
            alpaca_ticks=Mock(),
            alpaca_headers={},
            logger=mock_logger
        )

        orchestrator.fundamental_collector.collect_derived_long = Mock(return_value=(pl.DataFrame(), None))

        orchestrator.collect_derived_long('320193', '2024-01-01', '2024-12-31', 'AAPL')

        orchestrator.fundamental_collector.collect_derived_long.assert_called_once_with(
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


class TestFundamentalDataCollectorCaching:
    """Test FundamentalDataCollector caching behavior"""

    def test_fundamental_cache_race_condition(self):
        """Test cache returns existing entry on race condition (lines 466,467)."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(
            logger=mock_logger,
            sec_rate_limiter=Mock()
        )

        # Pre-populate cache to simulate race condition
        mock_fundamental = Mock()
        collector._fundamental_cache['0000320193'] = mock_fundamental

        # Now try to get the same CIK - should return cached value
        with patch('quantdl.storage.data_collectors.Fundamental') as MockFundamental:
            MockFundamental.return_value = Mock()
            result = collector._get_or_create_fundamental('0000320193', 'AAPL')

        # Should return the cached entry
        assert result == mock_fundamental
        MockFundamental.assert_not_called()

    def test_ttm_long_empty_stock_long_handling(self):
        """Test collect_ttm_long_range handles empty stock_long DataFrame (lines 751-754)."""
        from quantdl.storage.data_collectors import FundamentalDataCollector

        mock_logger = Mock(spec=logging.Logger)
        collector = FundamentalDataCollector(
            logger=mock_logger,
            sec_rate_limiter=Mock()
        )

        # Mock _get_or_create_fundamental to return a mock Fundamental
        mock_fundamental = Mock()
        mock_fundamental.get_concept_data.return_value = []

        with patch.object(collector, '_get_or_create_fundamental', return_value=mock_fundamental):
            with patch.object(collector, '_load_concepts', return_value=['rev', 'assets', 'shares_out']):
                result = collector.collect_ttm_long_range(
                    cik='0000320193',
                    start_date='2024-01-01',
                    end_date='2024-12-31',
                    symbol='AAPL'
                )

        # Should return empty DataFrame without error
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
