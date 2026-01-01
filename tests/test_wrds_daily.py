import unittest
from unittest.mock import Mock, patch
from collection.wrds_daily import WRDSDailyTicks
import pandas as pd


class TestWRDSDailyTicks(unittest.TestCase):

    @patch('collection.wrds_daily.wrds.Connection')
    @patch('collection.wrds_daily.SecurityMaster')
    def setUp(self, mock_security_master, mock_wrds_conn):
        """Set up test fixtures"""
        self.mock_conn = Mock()
        self.wdt = WRDSDailyTicks(conn=self.mock_conn)

    def test_get_daily_returns_dict(self):
        """Test that get_daily returns expected dict format"""
        # Mock security master
        self.wdt.security_master.get_security_id = Mock(return_value='SEC001')
        self.wdt.security_master.sid_to_permno = Mock(return_value=14593)

        # Mock WRDS query response
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2021-12-31')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        self.mock_conn.raw_sql = Mock(return_value=mock_df)

        result = self.wdt.get_daily('AAPL', '2021-12-31')

        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'open' in result
        assert 'close' in result
        assert result['timestamp'] == '2021-12-31'

    def test_get_daily_empty_data(self):
        """Test get_daily returns empty dict when no data found"""
        self.wdt.security_master.get_security_id = Mock(return_value='SEC001')
        self.wdt.security_master.sid_to_permno = Mock(return_value=14593)

        # Mock empty response
        self.mock_conn.raw_sql = Mock(return_value=pd.DataFrame())

        result = self.wdt.get_daily('INVALID', '2021-12-31')

        assert result == {}

    def test_smart_symbol_resolution_fb_to_meta(self):
        """
        Test smart auto-resolve for ticker name changes (FB → META)

        Scenario: User queries 'META' for 2021-12-31
        - 'META' wasn't active on that date (it was still 'FB')
        - System should find security_id that later became META
        - Fetch correct data using permno
        """
        # Simulate: 'META' not found on 2021-12-31 (returns None)
        # Then fallback finds historical match
        def mock_get_security_id(symbol, day):
            if symbol == 'META' and day == '2021-12-31':
                return None  # META didn't exist yet
            elif symbol == 'FB' and day == '2021-12-31':
                return 1234  # FB was active
            return None

        self.wdt.security_master.get_security_id = Mock(side_effect=mock_get_security_id)

        # Mock master table for fallback lookup
        import polars as pl
        mock_master = pl.DataFrame({
            'security_id': [1234, 1234],
            'symbol': ['FB', 'META'],
            'start_date': [pl.date(2010, 1, 1), pl.date(2022, 6, 9)],
            'end_date': [pl.date(2022, 6, 8), pl.date(2024, 12, 31)]
        })
        self.wdt.security_master.master_table = Mock(return_value=mock_master)
        self.wdt.security_master.sid_to_permno = Mock(return_value=10112)

        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2021-12-31')],
            'open': [336.35],
            'high': [344.0],
            'low': [335.0],
            'close': [336.35],
            'volume': [2300000]
        })
        self.mock_conn.raw_sql = Mock(return_value=mock_df)

        # Query using 'META' for a date when it was still 'FB'
        result = self.wdt.get_daily('META', '2021-12-31')

        # Should successfully resolve and return data
        assert result['timestamp'] == '2021-12-31'
        assert result['close'] == 336.35

        # Verify it used the correct security_id
        self.wdt.security_master.sid_to_permno.assert_called_with(1234)


if __name__ == '__main__':
    unittest.main()