"""
Unit tests for utils.calendar module
Tests trading calendar functionality
"""
import pytest
import datetime as dt
from pathlib import Path
from unittest.mock import Mock, patch
import polars as pl
from quantdl.utils.calendar import TradingCalendar


class TestTradingCalendar:
    """Test TradingCalendar class"""

    @pytest.fixture
    def mock_calendar_data(self, tmp_path):
        """Create mock calendar parquet file for testing"""
        # Create sample trading days for 2024
        dates = [
            dt.date(2024, 1, 2),  # Tuesday
            dt.date(2024, 1, 3),  # Wednesday
            dt.date(2024, 1, 4),  # Thursday
            dt.date(2024, 1, 5),  # Friday
            # Skip weekend
            dt.date(2024, 1, 8),  # Monday
            dt.date(2024, 1, 9),  # Tuesday
            # More days...
            dt.date(2024, 6, 3),
            dt.date(2024, 6, 4),
            dt.date(2024, 6, 5),
            dt.date(2024, 12, 30),
            dt.date(2024, 12, 31),
        ]

        df = pl.DataFrame({"timestamp": dates})
        calendar_path = tmp_path / "test_calendar.parquet"
        df.write_parquet(calendar_path)

        return calendar_path

    def test_initialization_default_path(self):
        """Test TradingCalendar initialization with default path"""
        calendar = TradingCalendar()
        assert calendar.calendar_path == Path("data/calendar/master.parquet")

    def test_initialization_custom_path(self, tmp_path):
        """Test TradingCalendar initialization with custom path"""
        custom_path = tmp_path / "custom_calendar.parquet"
        # Create empty calendar to prevent auto-generation
        df = pl.DataFrame({"timestamp": []})
        df.write_parquet(custom_path)

        calendar = TradingCalendar(calendar_path=custom_path)
        assert calendar.calendar_path == custom_path

    def test_load_trading_days_january(self, mock_calendar_data):
        """Test loading trading days for January 2024"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days(year=2024, month=1)

        assert len(trading_days) == 6
        assert "2024-01-02" in trading_days
        assert "2024-01-03" in trading_days
        assert "2024-01-09" in trading_days
        # Weekend should not be included
        assert "2024-01-06" not in trading_days
        assert "2024-01-07" not in trading_days

    def test_load_trading_days_june(self, mock_calendar_data):
        """Test loading trading days for June 2024"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days(year=2024, month=6)

        assert len(trading_days) == 3
        assert "2024-06-03" in trading_days
        assert "2024-06-04" in trading_days
        assert "2024-06-05" in trading_days

    def test_load_trading_days_december(self, mock_calendar_data):
        """Test loading trading days for December 2024"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days(year=2024, month=12)

        assert len(trading_days) == 2
        assert "2024-12-30" in trading_days
        assert "2024-12-31" in trading_days

    def test_load_trading_days_empty_month(self, mock_calendar_data):
        """Test loading trading days for a month with no data"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days(year=2024, month=2)

        assert len(trading_days) == 0

    def test_load_trading_days_year(self, mock_calendar_data):
        """Test loading all trading days for 2024"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days_year(year=2024)

        assert len(trading_days) == 11
        assert "2024-01-02" in trading_days
        assert "2024-06-03" in trading_days
        assert "2024-12-31" in trading_days

    def test_load_trading_days_year_empty(self, mock_calendar_data):
        """Test loading trading days for a year with no data"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days_year(year=2023)

        assert len(trading_days) == 0

    def test_is_trading_day_true(self, mock_calendar_data):
        """Test is_trading_day returns True for trading days"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)

        assert calendar.is_trading_day(dt.date(2024, 1, 2)) is True
        assert calendar.is_trading_day(dt.date(2024, 6, 3)) is True
        assert calendar.is_trading_day(dt.date(2024, 12, 31)) is True

    def test_is_trading_day_false(self, mock_calendar_data):
        """Test is_trading_day returns False for non-trading days"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)

        # Weekend day
        assert calendar.is_trading_day(dt.date(2024, 1, 6)) is False
        # Day not in dataset
        assert calendar.is_trading_day(dt.date(2023, 12, 31)) is False
        assert calendar.is_trading_day(dt.date(2025, 1, 1)) is False

    def test_load_trading_days_format(self, mock_calendar_data):
        """Test that dates are returned in correct format"""
        calendar = TradingCalendar(calendar_path=mock_calendar_data)
        trading_days = calendar.load_trading_days(year=2024, month=1)

        # Verify format is YYYY-MM-DD
        for day in trading_days:
            assert len(day) == 10
            assert day[4] == '-'
            assert day[7] == '-'
            # Verify it's a valid date
            dt.datetime.strptime(day, '%Y-%m-%d')

    @patch('quantdl.utils.calendar.requests.get')
    @patch('quantdl.utils.calendar.os.getenv')
    def test_auto_generate_calendar(self, mock_getenv, mock_requests_get, tmp_path):
        """Test calendar auto-generation when file missing"""
        # Mock Alpaca credentials
        mock_getenv.side_effect = lambda key: {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_API_SECRET': 'test_secret'
        }.get(key)

        # Mock Alpaca API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {'date': '2024-01-02'},
            {'date': '2024-01-03'},
        ]
        mock_requests_get.return_value = mock_response

        # Create calendar with non-existent path
        calendar_path = tmp_path / "auto_generated.parquet"
        calendar = TradingCalendar(calendar_path=calendar_path)

        # Verify calendar file was created
        assert calendar_path.exists()

        # Verify API was called correctly
        mock_requests_get.assert_called_once()
        call_args = mock_requests_get.call_args
        assert call_args[1]['params']['start'] == "2009-01-01T00:00:00Z"
        assert call_args[1]['params']['end'] == "2029-12-31T00:00:00Z"

    @patch('quantdl.utils.calendar.os.getenv')
    def test_auto_generate_missing_credentials(self, mock_getenv, tmp_path):
        """Test auto-generation fails gracefully without credentials"""
        # Mock missing credentials
        mock_getenv.return_value = None

        calendar_path = tmp_path / "missing_creds.parquet"

        with pytest.raises(RuntimeError, match="ALPACA_API_KEY and ALPACA_API_SECRET required"):
            TradingCalendar(calendar_path=calendar_path)
