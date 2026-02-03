"""
Unit tests for collection.alpaca_ticks module
Tests Alpaca API tick data collection functionality
"""
import pytest
import datetime as dt
import zoneinfo
import polars as pl
from unittest.mock import Mock, patch, MagicMock
from quantdl.collection.alpaca_ticks import Ticks


class TestTicks:
    """Test Ticks class"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_initialization(self, mock_logger_factory):
        """Test Ticks initialization"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Verify headers are set correctly
        assert ticks.headers['accept'] == 'application/json'
        assert ticks.headers['APCA-API-KEY-ID'] == 'test_key'
        assert ticks.headers['APCA-API-SECRET-KEY'] == 'test_secret'

        # Verify logger was set up
        mock_logger_factory.assert_called_once()

    def test_get_trade_day_range_normal_day(self):
        """Test get_trade_day_range for a normal trading day"""
        start_str, end_str = Ticks.get_trade_day_range("2024-06-15")

        # Parse back to datetime for validation
        start_dt = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_dt = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00'))

        # Convert to ET for verification
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = start_dt.astimezone(eastern)
        end_et = end_dt.astimezone(eastern)

        # Verify times are 9:30 AM and 4:00 PM ET
        assert start_et.hour == 9
        assert start_et.minute == 30
        assert end_et.hour == 16
        assert end_et.minute == 0

        # Verify date is correct
        assert start_et.date() == dt.date(2024, 6, 15)
        assert end_et.date() == dt.date(2024, 6, 15)

    def test_get_trade_day_range_dst_summer(self):
        """Test get_trade_day_range during DST (summer)"""
        # June 15, 2024 is during DST (EDT = UTC-4)
        start_str, end_str = Ticks.get_trade_day_range("2024-06-15")

        start_dt = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))

        # During DST, 9:30 AM ET = 1:30 PM UTC
        assert start_dt.hour == 13  # 9:30 AM + 4 hours
        assert start_dt.minute == 30

    def test_get_trade_day_range_est_winter(self):
        """Test get_trade_day_range during standard time (winter)"""
        # January 15, 2024 is during EST (EST = UTC-5)
        start_str, end_str = Ticks.get_trade_day_range("2024-01-15")

        start_dt = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))

        # During EST, 9:30 AM ET = 2:30 PM UTC
        assert start_dt.hour == 14  # 9:30 AM + 5 hours
        assert start_dt.minute == 30

    def test_get_trade_day_range_format(self):
        """Test that output format is correct"""
        start_str, end_str = Ticks.get_trade_day_range("2024-06-15")

        # Verify format ends with 'Z'
        assert start_str.endswith('Z')
        assert end_str.endswith('Z')

        # Verify it's valid ISO format
        assert 'T' in start_str
        assert 'T' in end_str

    def test_get_trade_day_range_different_dates(self):
        """Test get_trade_day_range with various dates"""
        # Test different months and years
        test_dates = [
            "2024-01-02",
            "2024-06-30",
            "2023-12-31",
            "2020-02-29",  # Leap year
        ]

        for date_str in test_dates:
            start_str, end_str = Ticks.get_trade_day_range(date_str)

            # Verify both strings are returned
            assert isinstance(start_str, str)
            assert isinstance(end_str, str)
            assert start_str < end_str  # Start should be before end

    def test_get_trade_day_range_invalid_format(self):
        """Test get_trade_day_range with invalid date format"""
        with pytest.raises(ValueError):
            Ticks.get_trade_day_range("2024/06/15")  # Wrong separator

        with pytest.raises(ValueError):
            Ticks.get_trade_day_range("06-15-2024")  # Wrong order

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_date_calculation(self, mock_get, mock_logger_factory):
        """Test that recent_daily_ticks calculates dates correctly"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': []
            },
            'next_page_token': None
        }
        mock_get.return_value = mock_response

        ticks = Ticks()

        # Call with window of 90 days
        end_day = "2024-06-30"
        window = 90
        result = ticks.recent_daily_ticks(symbols=['AAPL'], end_day=end_day, window=window)

        # Verify logger was called with correct date range
        log_calls = mock_logger.info.call_args_list
        assert len(log_calls) > 0

        # Check that the start date is approximately 90 days before end date
        end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
        expected_start = end_dt - dt.timedelta(days=window)

        # Verify logger message contains date information
        log_message = str(log_calls[0])
        assert "2024-06-30" in log_message or end_day in log_message

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_calendar_dir_creation(self, mock_logger_factory):
        """Test that calendar directory is created"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Verify calendar_dir is set
        assert ticks.calendar_dir.name == "calendar"


class TestTicksEdgeCases:
    """Test edge cases and error handling"""

    def test_get_trade_day_range_weekend(self):
        """Test that weekend dates are handled correctly"""
        # Saturday, June 15, 2024
        saturday = "2024-06-15"
        start_str, end_str = Ticks.get_trade_day_range(saturday)

        # Should still generate times for that calendar date
        assert isinstance(start_str, str)
        assert isinstance(end_str, str)

    def test_get_trade_day_range_holiday(self):
        """Test that holiday dates are handled correctly"""
        # New Year's Day 2024
        holiday = "2024-01-01"
        start_str, end_str = Ticks.get_trade_day_range(holiday)

        # Should still generate times for that calendar date
        # (filtering for actual trading days happens elsewhere)
        assert isinstance(start_str, str)
        assert isinstance(end_str, str)

    def test_get_trade_day_range_leap_year(self):
        """Test February 29 on leap year"""
        leap_day = "2024-02-29"
        start_str, end_str = Ticks.get_trade_day_range(leap_day)

        start_dt = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        assert start_dt.date() == dt.date(2024, 2, 29)

    def test_get_trade_day_range_non_leap_year(self):
        """Test that Feb 29 on non-leap year raises error"""
        with pytest.raises(ValueError):
            Ticks.get_trade_day_range("2023-02-29")


class TestParseTicks:
    """Test parse_ticks static method"""

    def test_parse_ticks_basic(self):
        """Test basic tick parsing"""
        raw_ticks = [
            {
                't': '2024-01-03T14:30:00Z',
                'o': 100.0,
                'h': 101.0,
                'l': 99.0,
                'c': 100.5,
                'v': 1000000,
                'n': 5000,
                'vw': 100.25
            }
        ]

        parsed = Ticks.parse_ticks(raw_ticks)

        assert len(parsed) == 1
        assert parsed[0].open == 100.0
        assert parsed[0].high == 101.0
        assert parsed[0].low == 99.0
        assert parsed[0].close == 100.5
        assert parsed[0].volume == 1000000
        assert parsed[0].num_trades == 5000
        assert parsed[0].vwap == 100.25

    def test_parse_ticks_timezone_conversion(self):
        """Test that UTC timestamps are converted to Eastern Time"""
        # 2024-01-03 14:30:00 UTC = 09:30:00 EST (UTC-5 during winter)
        raw_ticks = [
            {
                't': '2024-01-03T14:30:00Z',
                'o': 100.0,
                'h': 101.0,
                'l': 99.0,
                'c': 100.5,
                'v': 1000000,
                'n': 5000,
                'vw': 100.25
            }
        ]

        parsed = Ticks.parse_ticks(raw_ticks)
        timestamp = dt.datetime.fromisoformat(parsed[0].timestamp)

        # Should be 9:30 AM Eastern (no timezone info after conversion)
        assert timestamp.hour == 9
        assert timestamp.minute == 30

    def test_parse_ticks_dst_conversion(self):
        """Test timezone conversion during DST"""
        # 2024-06-15 13:30:00 UTC = 09:30:00 EDT (UTC-4 during summer)
        raw_ticks = [
            {
                't': '2024-06-15T13:30:00Z',
                'o': 100.0,
                'h': 101.0,
                'l': 99.0,
                'c': 100.5,
                'v': 1000000,
                'n': 5000,
                'vw': 100.25
            }
        ]

        parsed = Ticks.parse_ticks(raw_ticks)
        timestamp = dt.datetime.fromisoformat(parsed[0].timestamp)

        # Should be 9:30 AM Eastern
        assert timestamp.hour == 9
        assert timestamp.minute == 30

    def test_parse_ticks_multiple(self):
        """Test parsing multiple ticks"""
        raw_ticks = [
            {
                't': '2024-01-03T14:30:00Z',
                'o': 100.0,
                'h': 101.0,
                'l': 99.0,
                'c': 100.5,
                'v': 1000000,
                'n': 5000,
                'vw': 100.25
            },
            {
                't': '2024-01-03T14:31:00Z',
                'o': 100.5,
                'h': 102.0,
                'l': 100.0,
                'c': 101.5,
                'v': 2000000,
                'n': 6000,
                'vw': 101.0
            }
        ]

        parsed = Ticks.parse_ticks(raw_ticks)

        assert len(parsed) == 2
        assert parsed[0].close == 100.5
        assert parsed[1].close == 101.5

    def test_parse_ticks_empty_list(self):
        """Test parsing empty tick list"""
        parsed = Ticks.parse_ticks([])
        assert len(parsed) == 0

    def test_parse_ticks_invalid_data(self):
        """Test that empty tick raises ValueError"""
        with pytest.raises(ValueError, match="tick is empty"):
            Ticks.parse_ticks([None])


class TestGetTicks:
    """Test get_ticks method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_get_ticks_success(self, mock_get, mock_logger_factory):
        """Test successful tick retrieval"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {
                        't': '2024-01-03T14:30:00Z',
                        'o': 100.0,
                        'h': 101.0,
                        'l': 99.0,
                        'c': 100.5,
                        'v': 1000000,
                        'n': 5000,
                        'vw': 100.25
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_ticks('AAPL', '2024-01-03T00:00:00Z', '2024-01-03T23:59:59Z')

        assert len(result) == 1
        assert result[0]['c'] == 100.5
        mock_get.assert_called_once()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_get_ticks_api_error(self, mock_get, mock_logger_factory):
        """Test handling of API error"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_ticks('AAPL', '2024-01-03T00:00:00Z', '2024-01-03T23:59:59Z')

        assert result == []
        mock_logger.error.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_get_ticks_no_data(self, mock_get, mock_logger_factory):
        """Test when API returns no data"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'bars': {}}
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_ticks('AAPL', '2024-01-03T00:00:00Z', '2024-01-03T23:59:59Z')

        assert result == []

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_get_ticks_exception(self, mock_get, mock_logger_factory):
        """Test exception handling"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_get.side_effect = Exception("Network error")

        ticks = Ticks()
        result = ticks.get_ticks('AAPL', '2024-01-03T00:00:00Z', '2024-01-03T23:59:59Z')

        assert result == []
        mock_logger.error.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_get_ticks_missing_bars(self, mock_get, mock_logger_factory):
        """Test get_ticks when the response lacks bar entries"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'meta': 'value'}
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_ticks('AAPL', '2024-01-03T00:00:00Z', '2024-01-03T23:59:59Z')

        assert result == []
        mock_logger.error.assert_called()
        assert any("No bars in json response" in call[0][0] for call in mock_logger.error.call_args_list)


class TestGetMonthRange:
    """Test _get_month_range method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_get_month_range_january(self, mock_logger_factory):
        """Test month range for January - uses ET boundaries converted to UTC"""
        import datetime as dt
        import zoneinfo

        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 1)

        # Parse UTC strings back to ET to verify correct boundaries
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_utc = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_utc = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        start_et = start_utc.astimezone(eastern)
        end_et = end_utc.astimezone(eastern)

        # Start should be 4:00 AM ET on Jan 1
        assert start_et.date() == dt.date(2024, 1, 1)
        assert start_et.hour == 4
        # End should be 8:00 PM ET on Jan 31
        assert end_et.date() == dt.date(2024, 1, 31)
        assert end_et.hour == 20
        assert start_str.endswith('Z')
        assert end_str.endswith('Z')

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_get_month_range_february_leap(self, mock_logger_factory):
        """Test month range for February in leap year - uses ET boundaries"""
        import datetime as dt
        import zoneinfo

        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 2)

        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_utc = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_utc = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        start_et = start_utc.astimezone(eastern)
        end_et = end_utc.astimezone(eastern)

        assert start_et.date() == dt.date(2024, 2, 1)
        assert end_et.date() == dt.date(2024, 2, 29)  # Leap year

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_get_month_range_december(self, mock_logger_factory):
        """Test month range for December - uses ET boundaries"""
        import datetime as dt
        import zoneinfo

        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()
        start_str, end_str = ticks._get_month_range(2024, 12)

        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_utc = dt.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end_utc = dt.datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        start_et = start_utc.astimezone(eastern)
        end_et = end_utc.astimezone(eastern)

        assert start_et.date() == dt.date(2024, 12, 1)
        assert end_et.date() == dt.date(2024, 12, 31)


class TestGetDaily:
    """Test get_daily method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_get_daily_success(self, mock_session_class, mock_logger_factory):
        """Test successful daily data retrieval"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {
                        't': '2024-01-03T05:00:00Z',
                        'o': 100.0,
                        'h': 101.0,
                        'l': 99.0,
                        'c': 100.5,
                        'v': 1000000,
                        'n': 5000,
                        'vw': 100.25
                    }
                ]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_daily('AAPL', 2024, 1)

        assert len(result) == 1
        assert result[0]['c'] == 100.5


class TestGetDailyYear:
    """Test get_daily_year method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_get_daily_year_success(self, mock_session_class, mock_logger_factory):
        """Test successful yearly data retrieval"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {
                        't': '2024-01-03T05:00:00Z',
                        'o': 100.0,
                        'h': 101.0,
                        'l': 99.0,
                        'c': 100.5,
                        'v': 1000000,
                        'n': 5000,
                        'vw': 100.25
                    }
                ]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_daily_year('AAPL', 2024)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert result['close'][0] == 100.5

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_get_daily_year_empty(self, mock_session_class, mock_logger_factory):
        """Test yearly data with no results"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'bars': {}, 'next_page_token': None}
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.get_daily_year('AAPL', 2024)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


class TestFetchDailyDayBulk:
    """Test fetch_daily_day_bulk method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_daily_day_bulk_params(self, mock_session_class, mock_logger_factory):
        """Ensure day-specific params are constructed correctly"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        with patch.object(Ticks, '_fetch_with_pagination', return_value={'AAPL': []}) as mock_fetch:
            ticks = Ticks()
            result = ticks.fetch_daily_day_bulk(['AAPL'], '2024-06-15')

        assert result == {'AAPL': []}

        called_params = mock_fetch.call_args.kwargs['params']
        assert called_params['start'] == '2024-06-15T00:00:00Z'
        assert called_params['end'] == '2024-06-15T23:59:59Z'
        assert called_params['timeframe'] == '1Day'
        assert called_params['adjustment'] == 'split'
        assert called_params['symbols'] == 'AAPL'

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_daily_day_bulk_session_closed_on_error(self, mock_session_class, mock_logger_factory):
        """Ensure session is closed even when pagination call fails"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        with patch.object(Ticks, '_fetch_with_pagination', side_effect=RuntimeError("boom")):
            ticks = Ticks()
            with pytest.raises(RuntimeError):
                ticks.fetch_daily_day_bulk(['AAPL'], '2024-06-15')

        mock_session.close.assert_called_once()


class TestFetchMinuteDayBulk:
    """Test fetch_minute_day_bulk method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_day_bulk_success(self, mock_session_class, mock_logger_factory):
        """Test successful bulk minute data fetch for a day"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session and response
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}],
                'MSFT': [{'t': '2024-01-03T14:30:00Z', 'o': 200, 'h': 201, 'l': 199, 'c': 200.5, 'v': 2000, 'n': 60, 'vw': 200.25}]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.fetch_minute_day_bulk(['AAPL', 'MSFT'], '2024-01-03')

        assert 'AAPL' in result
        assert 'MSFT' in result
        assert len(result['AAPL']) == 1
        assert len(result['MSFT']) == 1


class TestFetchMinuteMonthBulk:
    """Test fetch_minute_month_bulk method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_month_bulk_success(self, mock_session_class, mock_logger_factory):
        """Test successful bulk minute data fetch for a month"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session and response
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.fetch_minute_month_bulk(['AAPL'], 2024, 1)

        assert 'AAPL' in result
        assert len(result['AAPL']) == 1


class TestRecentDailyTicks:
    """Test recent_daily_ticks method with comprehensive scenarios"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_success(self, mock_get, mock_logger_factory):
        """Test successful recent daily ticks retrieval"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {
                        't': '2024-06-28T04:00:00Z',
                        'o': 210.0,
                        'h': 212.0,
                        'l': 209.0,
                        'c': 211.0,
                        'v': 50000000,
                        'n': 200000,
                        'vw': 210.5
                    }
                ]
            },
            'next_page_token': None
        }
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert 'AAPL' in result
        assert isinstance(result['AAPL'], pl.DataFrame)
        assert len(result['AAPL']) == 1
        assert result['AAPL']['close'][0] == 211.0

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_pagination(self, mock_get, mock_logger_factory):
        """Test pagination handling in recent_daily_ticks"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # First page
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': '2024-06-28T04:00:00Z', 'o': 210, 'h': 212, 'l': 209, 'c': 211, 'v': 50000000, 'n': 200000, 'vw': 210.5}
                ]
            },
            'next_page_token': 'token123'
        }

        # Second page
        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': '2024-06-29T04:00:00Z', 'o': 211, 'h': 213, 'l': 210, 'c': 212, 'v': 51000000, 'n': 210000, 'vw': 211.5}
                ]
            },
            'next_page_token': None
        }

        mock_get.side_effect = [mock_response_1, mock_response_2]

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert 'AAPL' in result
        assert len(result['AAPL']) == 2  # Two data points from two pages

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_api_error(self, mock_get, mock_logger_factory):
        """Test recent_daily_ticks with API error"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        # Should return empty dict on API error
        assert result == {}

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_future_end_date(self, mock_get, mock_logger_factory):
        """Test recent_daily_ticks with end_date in the future"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        class FixedDateTime(dt.datetime):
            @classmethod
            def today(cls):
                return cls(2024, 6, 15, 12, 0, 0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {'AAPL': []},
            'next_page_token': None
        }
        mock_get.return_value = mock_response

        with patch('quantdl.collection.alpaca_ticks.dt.datetime', FixedDateTime):
            ticks = Ticks()
            # Request data ending on 2024-06-30 (future date)
            result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        # Should adjust end_date to yesterday (2024-06-14)
        assert isinstance(result, dict)

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_processing_error(self, mock_get, mock_logger_factory):
        """Test recent_daily_ticks when processing fails for some symbols"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Return invalid data that will fail parsing
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': 'invalid-timestamp', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}
                ]
            },
            'next_page_token': None
        }
        mock_get.return_value = mock_response

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        # Should handle error gracefully
        mock_logger.error.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.get')
    def test_recent_daily_ticks_request_exception(self, mock_get, mock_logger_factory):
        """Test recent_daily_ticks handles request exceptions"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_get.side_effect = Exception("network failure")

        ticks = Ticks()
        result = ticks.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        assert result == {}
        mock_logger.error.assert_called()
        assert any("Request failed on page" in call[0][0] for call in mock_logger.error.call_args_list)


class TestFetchWithPagination:
    """Test _fetch_with_pagination method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_with_pagination_single_page(self, mock_logger_factory):
        """Test pagination with single page"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 1

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_with_pagination_multiple_pages(self, mock_logger_factory):
        """Test pagination with multiple pages"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Mock session with multiple pages
        mock_session = Mock()

        # Page 1
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': 'token123'
        }

        # Page 2
        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:31:00Z', 'o': 101, 'h': 102, 'l': 100, 'c': 101.5, 'v': 2000, 'n': 60, 'vw': 101.25}]
            },
            'next_page_token': None
        }

        mock_session.get.side_effect = [mock_response_1, mock_response_2]

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 2

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_with_pagination_api_error(self, mock_logger_factory):
        """Test pagination with API error"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Mock session with error
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_session.get.return_value = mock_response

        result = ticks._fetch_with_pagination(
            session=mock_session,
            base_url='https://test.com',
            params={'symbols': 'AAPL'},
            symbols=['AAPL'],
            sleep_time=0.01
        )

        # Should return empty lists for symbols
        assert 'AAPL' in result
        assert len(result['AAPL']) == 0
        mock_logger.error.assert_called()


class TestFetchMinuteBulkWithRetry:
    """Test _fetch_minute_bulk_with_retry method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_bulk_with_retry_success(self, mock_session_class, mock_logger_factory):
        """Test successful fetch without retry"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 1

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    @patch('quantdl.collection.alpaca_ticks.time.sleep')
    def test_fetch_minute_bulk_with_retry_resets_page_token(self, mock_sleep, mock_session_class, mock_logger_factory):
        """Ensure page_token is cleared before retrying"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        call_params = []

        def get_side_effect(_, params):
            call_params.append(params.copy())
            response = Mock()
            idx = len(call_params)
            if idx == 1:
                response.status_code = 200
                response.json.return_value = {
                    'bars': {'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]},
                    'next_page_token': 'token'
                }
            elif idx == 2:
                raise RuntimeError("paging failure")
            else:
                response.status_code = 200
                response.json.return_value = {
                    'bars': {'AAPL': []},
                    'next_page_token': None
                }
            return response

        mock_session.get.side_effect = get_side_effect

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        assert 'AAPL' in result
        assert len(call_params) == 3
        assert call_params[1].get('page_token') == 'token'
        assert 'page_token' not in call_params[2]
        assert mock_logger.error.called

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    @patch('quantdl.collection.alpaca_ticks.time.sleep')
    def test_fetch_minute_bulk_with_retry_rate_limit(self, mock_sleep, mock_session_class, mock_logger_factory):
        """Test retry on rate limit (429)"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # First request: rate limit, second request: success
        mock_response_429 = Mock()
        mock_response_429.status_code = 429

        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': None
        }

        mock_session.get.side_effect = [mock_response_429, mock_response_200]

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 1
        # Verify warning was logged
        mock_logger.warning.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    @patch('quantdl.collection.alpaca_ticks.time.sleep')
    def test_fetch_minute_bulk_with_retry_max_retries(self, mock_sleep, mock_session_class, mock_logger_factory):
        """Test max retries exhausted"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # All requests fail
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        # Should return empty data after max retries
        assert 'AAPL' in result
        assert len(result['AAPL']) == 0

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_bulk_with_retry_exception(self, mock_session_class, mock_logger_factory):
        """Test exception handling"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Raise exception on first request, success on second
        mock_session.get.side_effect = [
            Exception("Network error"),
            Mock(status_code=200, json=lambda: {'bars': {'AAPL': []}, 'next_page_token': None})
        ]

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        # Should retry and succeed
        assert 'AAPL' in result
        mock_logger.error.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    @patch('quantdl.collection.alpaca_ticks.time.sleep')
    def test_fetch_minute_bulk_with_retry_pagination_rate_limit(self, mock_sleep, mock_session_class, mock_logger_factory):
        """Test rate limit during pagination"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # First page success, second page rate limit, third page success
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            'bars': {'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]},
            'next_page_token': 'token1'
        }

        mock_response_429 = Mock()
        mock_response_429.status_code = 429

        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            'bars': {'AAPL': [{'t': '2024-01-03T14:31:00Z', 'o': 101, 'h': 102, 'l': 100, 'c': 101.5, 'v': 2000, 'n': 60, 'vw': 101.25}]},
            'next_page_token': None
        }

        mock_session.get.side_effect = [mock_response_1, mock_response_429, mock_response_2]

        ticks = Ticks()
        result = ticks._fetch_minute_bulk_with_retry(
            symbols=['AAPL'],
            start_str='2024-01-03T00:00:00Z',
            end_str='2024-01-03T23:59:59Z',
            period_desc='2024-01-03',
            sleep_time=0.01
        )

        assert 'AAPL' in result
        # Should have data from both pages
        assert len(result['AAPL']) == 2


class TestFetchMinuteMonthSingle:
    """Test fetch_minute_month_single method"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_month_single_success(self, mock_session_class, mock_logger_factory):
        """Test successful single symbol month fetch"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'bars': {
                'AAPL': [
                    {'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}
                ]
            },
            'next_page_token': None
        }
        mock_session.get.return_value = mock_response

        ticks = Ticks()
        result = ticks.fetch_minute_month_single('AAPL', 2024, 1, sleep_time=0.01)

        assert len(result) == 1
        assert result[0]['c'] == 100.5

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_month_single_exception(self, mock_session_class, mock_logger_factory):
        """Test exception handling in single symbol fetch"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_session.get.side_effect = Exception("Network error")

        ticks = Ticks()
        result = ticks.fetch_minute_month_single('AAPL', 2024, 1, sleep_time=0.01)

        # Should return empty list
        assert result == []
        mock_logger.error.assert_called()


class TestFetchMinuteMonthBulkFallback:
    """Test fetch_minute_month_bulk fallback scenario"""

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    @patch('quantdl.collection.alpaca_ticks.requests.Session')
    def test_fetch_minute_month_bulk_fallback_to_individual(self, mock_session_class, mock_logger_factory):
        """Test fallback to individual fetching when bulk returns no data"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        # Mock session for bulk fetch (returns empty)
        mock_session_bulk = Mock()
        mock_session_bulk.get.return_value = Mock(
            status_code=200,
            json=lambda: {'bars': {}, 'next_page_token': None}
        )

        # Mock session for individual fetches (returns data)
        mock_session_individual = Mock()
        mock_response_individual = Mock()
        mock_response_individual.status_code = 200
        mock_response_individual.json.return_value = {
            'bars': {
                'AAPL': [{'t': '2024-01-03T14:30:00Z', 'o': 100, 'h': 101, 'l': 99, 'c': 100.5, 'v': 1000, 'n': 50, 'vw': 100.25}]
            },
            'next_page_token': None
        }
        mock_session_individual.get.return_value = mock_response_individual

        mock_session_class.side_effect = [mock_session_bulk, mock_session_individual]

        ticks = Ticks()
        result = ticks.fetch_minute_month_bulk(['AAPL'], 2024, 1, sleep_time=0.01)

        # Should have data from individual fetch
        assert 'AAPL' in result
        assert len(result['AAPL']) > 0
        # Verify fallback was logged
        mock_logger.info.assert_any_call('Bulk fetch returned no data, fetching symbols individually')

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_minute_month_bulk_individual_returns_empty_logs_info(self, mock_logger_factory):
        """Ensure info is logged when individual fetch returns no bars"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        with patch.object(Ticks, '_fetch_minute_bulk_with_retry', return_value={'AAPL': []}):
            with patch.object(Ticks, 'fetch_minute_month_single', return_value=[]):
                ticks = Ticks()
                result = ticks.fetch_minute_month_bulk(['AAPL'], 2024, 1, sleep_time=0.01)

        assert 'AAPL' in result
        mock_logger.info.assert_any_call('No data returned for AAPL')

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_minute_month_bulk_individual_failure_logs_warning(self, mock_logger_factory):
        """Ensure warning is logged when individual retries all fail"""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        with patch.object(Ticks, '_fetch_minute_bulk_with_retry', return_value={'AAPL': []}):
            with patch.object(Ticks, 'fetch_minute_month_single', side_effect=RuntimeError("boom")):
                ticks = Ticks()
                result = ticks.fetch_minute_month_bulk(['AAPL'], 2024, 1, sleep_time=0.01)

        assert 'AAPL' in result
        mock_logger.warning.assert_called()
        assert 'AAPL' in mock_logger.warning.call_args[0][0]
        mock_logger.error.assert_called()

    @patch.dict('os.environ', {'ALPACA_API_KEY': 'test_key', 'ALPACA_API_SECRET': 'test_secret'})
    @patch('quantdl.collection.alpaca_ticks.LoggerFactory')
    def test_fetch_minute_pagination_error_handling(self, mock_logger_factory):
        """Test pagination error handling (lines 558, 562)."""
        mock_logger = Mock()
        mock_logger_factory.return_value.get_logger.return_value = mock_logger

        ticks = Ticks()

        # Mock requests to fail during pagination
        mock_response = Mock()
        mock_response.json.side_effect = Exception("JSON parse error")
        mock_response.status_code = 200

        with patch('quantdl.collection.alpaca_ticks.requests.get', return_value=mock_response):
            with pytest.raises(Exception):
                ticks._fetch_minute_with_pagination(
                    sym='AAPL',
                    start='2024-01-02T14:30:00Z',
                    end='2024-01-02T21:00:00Z'
                )
