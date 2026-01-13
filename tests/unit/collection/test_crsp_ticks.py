"""
Unit tests for collection.crsp_ticks module
Tests WRDS CRSP daily tick data collection functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path
from sqlalchemy.exc import PendingRollbackError
from quantdl.collection.crsp_ticks import CRSPDailyTicks


class TestCRSPDailyTicks:
    """Test CRSPDailyTicks class"""

    @patch('quantdl.collection.crsp_ticks.wrds.Connection')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_initialization_with_custom_connection(self, mock_logger, mock_security_master, mock_wrds):
        """Test initialization with provided connection"""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)

        # Verify custom connection is used
        assert crsp.conn == mock_conn

        # Verify wrds.Connection was NOT called (since we provided conn)
        mock_wrds.assert_not_called()

    @patch.dict('os.environ', {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.collection.crsp_ticks.wrds.Connection')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_initialization_without_connection(self, mock_logger, mock_security_master, mock_wrds):
        """Test initialization without provided connection"""
        mock_conn = Mock()
        mock_wrds.return_value = mock_conn

        crsp = CRSPDailyTicks(conn=None)

        # Verify wrds.Connection was called with credentials
        mock_wrds.assert_called_once_with(
            wrds_username='test_user',
            wrds_password='test_pass'
        )

        # Verify connection is set
        assert crsp.conn == mock_conn

    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_calendar_path_setup(self, mock_logger, mock_security_master):
        """Test that calendar paths are set up correctly"""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)

        # Verify calendar paths
        assert crsp.calendar_dir == Path("data/calendar")
        assert crsp.calendar_path == Path("data/calendar/master.parquet")

    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_logger_setup(self, mock_logger, mock_security_master):
        """Test that logger is set up correctly"""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)

        # Verify logger setup was called
        mock_logger.assert_called_once()
        call_kwargs = mock_logger.call_args[1]
        assert call_kwargs['name'] == 'collection.crsp_ticks'

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_adjusted(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily with adjusted=True"""
        mock_conn = Mock()

        # Setup mocks
        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        # Mock database query result
        mock_df = pd.DataFrame({
            'date': ['2024-06-30'],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30', adjusted=True)

        # Verify security master methods were called
        mock_sm_instance.get_security_id.assert_called_once_with('AAPL', '2024-06-30', auto_resolve=True)
        mock_sm_instance.sid_to_permno.assert_called_once_with('sid_123')

        # Verify validation methods were called
        mock_validate_permno.assert_called_once_with(10516)
        mock_validate_date.assert_called_once_with('2024-06-30')

        # Verify SQL query was executed
        mock_conn.raw_sql.assert_called_once()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_unadjusted(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily with adjusted=False"""
        mock_conn = Mock()

        # Setup mocks
        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        # Mock database query result
        mock_df = pd.DataFrame({
            'date': ['2024-06-30'],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30', adjusted=False)

        # Verify query was called with different parameters for unadjusted
        mock_conn.raw_sql.assert_called_once()
        query = mock_conn.raw_sql.call_args[0][0]

        # Unadjusted query should NOT divide by cfacpr/cfacshr
        assert '/ cfacpr' not in query

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_with_auto_resolve_false(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily with auto_resolve=False"""
        mock_conn = Mock()

        # Setup mocks
        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        mock_df = pd.DataFrame({
            'date': ['2024-06-30'],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30', auto_resolve=False)

        # Verify auto_resolve was passed correctly
        mock_sm_instance.get_security_id.assert_called_once_with('AAPL', '2024-06-30', auto_resolve=False)

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_validation_called(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test that validation functions are called"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        # Make validation raise an error to verify it's called
        mock_validate_permno.side_effect = ValueError("Invalid permno")

        crsp = CRSPDailyTicks(conn=mock_conn)

        with pytest.raises(ValueError, match="Invalid permno"):
            crsp.get_daily(symbol='AAPL', day='2024-06-30')

        # Verify validation was attempted
        mock_validate_permno.assert_called_once()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_retries_on_pending_rollback(
        self,
        mock_logger,
        mock_security_master,
        mock_validate_permno,
        mock_validate_date
    ):
        """Retry and rollback when WRDS connection is in pending rollback state."""
        mock_conn = Mock()
        mock_conn.connection = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        mock_df = pd.DataFrame({
            'date': ['2024-06-30'],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.side_effect = [
            PendingRollbackError("pending rollback"),
            mock_df
        ]

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30', adjusted=True)

        assert result['close'] == 103.0
        mock_conn.connection.rollback.assert_called_once()


class TestCRSPDailyTicksEdgeCases:
    """Test edge cases and error handling"""

    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_security_master_initialization(self, mock_logger, mock_security_master_class):
        """Test that SecurityMaster is initialized with connection"""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)

        # Verify SecurityMaster was initialized (WRDS-free mode passes db=None initially)
        mock_security_master_class.assert_called_once_with(
            db=None, s3_client=None, bucket_name='us-equity-datalake'
        )

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_empty_result(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily with empty database result"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        # Mock empty result
        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30')

        # Should return empty dict
        assert result == {}

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_missing_ohlcv(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily with missing OHLCV data"""
        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        # Mock result with missing data (NaN in close)
        mock_df = pd.DataFrame({
            'date': ['2024-06-30'],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [None],  # Missing close
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30')

        # Should return empty dict
        assert result == {}

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_returns_correct_structure(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test that get_daily returns correct data structure"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.return_value = '2024-06-30'

        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily(symbol='AAPL', day='2024-06-30')

        # Verify structure
        assert 'timestamp' in result
        assert 'open' in result
        assert 'high' in result
        assert 'low' in result
        assert 'close' in result
        assert 'volume' in result

        # Verify values
        assert result['timestamp'] == '2024-06-30'
        assert result['open'] == 100.0
        assert result['close'] == 103.0
        assert result['volume'] == 1000000


class TestGetDailyRange:
    """Test get_daily_range method"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_success(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test successful date range fetch"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x  # Return input as-is

        # Mock multiple days of data
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-06-28'), pd.Timestamp('2024-06-29'), pd.Timestamp('2024-06-30')],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily_range(symbol='AAPL', start_day='2024-06-28', end_day='2024-06-30')

        # Should return list of 3 dicts
        assert len(result) == 3
        assert result[0]['timestamp'] == '2024-06-28'
        assert result[2]['timestamp'] == '2024-06-30'
        assert result[0]['close'] == 103.0
        assert result[2]['close'] == 105.0

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_empty_result(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily_range with empty result"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily_range(symbol='AAPL', start_day='2024-06-28', end_day='2024-06-30')

        assert result == []

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_unadjusted_query(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily_range uses unadjusted query when adjusted=False."""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        crsp.get_daily_range(symbol='AAPL', start_day='2024-06-28', end_day='2024-06-30', adjusted=False)

        query = mock_conn.raw_sql.call_args[0][0]
        assert 'cfacpr' not in query
        assert 'cfacshr' not in query

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_security_id_none(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily_range raises when security_id is None."""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = None
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)

        with pytest.raises(ValueError, match="security_id is None"):
            crsp.get_daily_range(symbol='AAPL', start_day='2024-06-01', end_day='2024-06-30')

        mock_validate_permno.assert_not_called()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_skip_missing_data(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test that rows with missing OHLCV are skipped"""
        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Create data with one row having missing close
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-06-28'), pd.Timestamp('2024-06-29'), pd.Timestamp('2024-06-30')],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, None, 107.0],  # Missing high on 2nd row
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily_range(symbol='AAPL', start_day='2024-06-28', end_day='2024-06-30')

        # Should only return 2 rows (skipped the one with missing high)
        assert len(result) == 2


class TestRecentDailyTicks:
    """Test recent_daily_ticks method"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_basic(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test basic recent_daily_ticks functionality"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Setup security master mocks
        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock the filter chain
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        final_mock.__getitem__.return_value.to_list.return_value = ['AAPL']

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        # Mock security_map
        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique
        security_map_unique.__getitem__.return_value.to_list.return_value = ['sid_123']

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock SQL query result
        mock_df = pd.DataFrame({
            'permno': [10516],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        # Mock polars conversion
        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.recent_daily_ticks(['AAPL'], '2024-06-30', window=90)

        # Verify result structure
        assert isinstance(result, dict)


class TestCollectDailyTicks:
    """Test collect_daily_ticks method"""

    @patch('quantdl.collection.crsp_ticks.align_calendar')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_success(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_align_calendar):
        """Test successful collect_daily_ticks"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock query result
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-06-28'), pd.Timestamp('2024-06-29'), pd.Timestamp('2024-06-30')],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_conn.raw_sql.return_value = mock_df

        # Mock align_calendar to return the data as-is
        mock_align_calendar.return_value = [
            {'timestamp': '2024-06-28', 'open': 100.0, 'high': 105.0, 'low': 99.0, 'close': 103.0, 'volume': 1000000},
            {'timestamp': '2024-06-29', 'open': 101.0, 'high': 106.0, 'low': 100.0, 'close': 104.0, 'volume': 1100000},
            {'timestamp': '2024-06-30', 'open': 102.0, 'high': 107.0, 'low': 101.0, 'close': 105.0, 'volume': 1200000}
        ]

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks('AAPL', 2024, 6)

        # Should return list of dicts
        assert isinstance(result, list)
        assert len(result) == 3

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_empty(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test collect_daily_ticks with no data"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock empty result
        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks('AAPL', 2024, 6)

        assert result == []


class TestClose:
    """Test close method"""

    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_close(self, mock_logger, mock_security_master):
        """Test that close method closes the connection"""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)
        crsp.close()

        # Verify connection close was called
        mock_conn.close.assert_called_once()


class TestRecentDailyTicksAdvanced:
    """Test recent_daily_ticks advanced scenarios"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_symbol_normalization(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test symbol normalization (removing dots and hyphens)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Setup security master mocks
        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock the filter chain for CRSP-normalized symbol (BRKB instead of BRK.B)
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        final_mock.__getitem__.return_value.to_list.return_value = ['BRKB']

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        # Mock security_map
        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique
        security_map_unique.__getitem__.return_value.to_list.return_value = ['sid_123']

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 17778
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 17778
        mock_validate_date.side_effect = lambda x: x

        # Mock SQL query result
        mock_df = pd.DataFrame({
            'permno': [17778],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        # Mock polars conversion
        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        # User provides BRK.B, system normalizes to BRKB
        result = crsp.recent_daily_ticks(['BRK.B'], '2024-06-30', window=90)

        # Verify result
        assert isinstance(result, dict)

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_mixed_success_failure(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test when some symbols succeed and some fail"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Setup security master mocks
        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock the filter chain
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        final_mock.__getitem__.return_value.to_list.return_value = ['AAPL']  # Only AAPL found

        mock_sm_instance.master_tb = mock_master_tb

        # AAPL succeeds, INVALID fails
        def mock_get_security_id(symbol, date, auto_resolve):
            if symbol == 'AAPL':
                return 'sid_aapl'
            raise ValueError("Symbol not found")

        mock_sm_instance.get_security_id.side_effect = mock_get_security_id

        # Mock security_map
        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique
        security_map_unique.__getitem__.return_value.to_list.return_value = ['sid_aapl']

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock SQL query result
        mock_df = pd.DataFrame({
            'permno': [10516],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        # Mock polars conversion
        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.recent_daily_ticks(['AAPL', 'INVALID'], '2024-06-30', window=90)

        # Should have AAPL but not INVALID
        assert isinstance(result, dict)
        # Verify warning was logged for failed symbol
        assert mock_logger_instance.warning.call_count >= 1

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_recent_daily_ticks_all_symbols_fail_resolution(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test when all symbols fail to resolve"""
        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Setup security master to fail for all symbols
        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock empty filter result
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        final_mock.__getitem__.return_value.to_list.return_value = []  # No symbols found

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.side_effect = ValueError("Not found")
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.recent_daily_ticks(['INVALID1', 'INVALID2'], '2024-06-30', window=90)

        # Should return empty dict
        assert result == {}
        # Verify error was logged
        mock_logger_instance.error.assert_called()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_custom_chunk_size(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test with custom chunk_size"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Setup security master mocks
        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock the filter chain
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        def _make_series(values):
            series = Mock()
            series.to_list.return_value = values
            return series

        final_mock.__getitem__.side_effect = [
            _make_series(['AAPL', 'MSFT', 'GOOGL']),
            _make_series(['sid_1', 'sid_2', 'sid_3'])
        ]

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        # Mock security_map
        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique
        security_map_unique.__getitem__.side_effect = [
            _make_series(['sid_1', 'sid_2', 'sid_3']),
            _make_series([10516, 10107, 12490])
        ]

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.side_effect = [10516, 10107, 12490]
        mock_validate_date.side_effect = lambda x: x

        # Mock SQL query result (2 chunks with chunk_size=2)
        mock_df = pd.DataFrame({
            'permno': [10516],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_conn.raw_sql.return_value = mock_df

        # Mock polars conversion
        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        # Use small chunk_size to test chunking logic
        result = crsp.recent_daily_ticks(
            ['AAPL', 'MSFT', 'GOOGL'],
            '2024-06-30',
            window=90,
            chunk_size=2  # Will create 2 chunks
        )

        # Should make 2 SQL queries (one per chunk)
        assert mock_conn.raw_sql.call_count >= 1


class TestGetDailyRangeDecember:
    """Test get_daily_range with December edge case"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_december(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test date range calculation for December"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock December data
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-12-30'), pd.Timestamp('2024-12-31')],
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [103.0, 104.0],
            'volume': [1000000, 1100000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily_range(symbol='AAPL', start_day='2024-12-30', end_day='2024-12-31')

        # Should handle December 31st correctly
        assert len(result) == 2
        assert result[1]['timestamp'] == '2024-12-31'


class TestCollectDailyTicksDecember:
    """Test collect_daily_ticks with December"""

    @patch('quantdl.collection.crsp_ticks.align_calendar')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_december(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_align_calendar):
        """Test collect_daily_ticks for December month"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock December data
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-12-02'), pd.Timestamp('2024-12-03'), pd.Timestamp('2024-12-31')],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_conn.raw_sql.return_value = mock_df

        mock_align_calendar.return_value = [
            {'timestamp': '2024-12-02', 'open': 100.0, 'high': 105.0, 'low': 99.0, 'close': 103.0, 'volume': 1000000},
            {'timestamp': '2024-12-03', 'open': 101.0, 'high': 106.0, 'low': 100.0, 'close': 104.0, 'volume': 1100000},
            {'timestamp': '2024-12-31', 'open': 102.0, 'high': 107.0, 'low': 101.0, 'close': 105.0, 'volume': 1200000}
        ]

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks('AAPL', 2024, 12)

        # Should handle December month (ends on 12/31) correctly
        assert isinstance(result, list)
        assert len(result) == 3

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_with_auto_resolve_false(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test collect_daily_ticks with auto_resolve=False"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks('AAPL', 2024, 6, auto_resolve=False)

        # Verify auto_resolve was passed correctly
        assert mock_sm_instance.get_security_id.call_args[1]['auto_resolve'] is False

    @patch('quantdl.collection.crsp_ticks.align_calendar')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_empty(self, mock_logger, mock_security_master, mock_align_calendar):
        """Test collect_daily_ticks returns empty list when no data."""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)
        crsp.get_daily_range = Mock(return_value=[])

        result = crsp.collect_daily_ticks('AAPL', 2024, 6)

        assert result == []
        mock_align_calendar.assert_not_called()


class TestCollectDailyTicksYearBulk:
    """Test collect_daily_ticks_year_bulk method."""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_year_bulk_basic(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Bulk fetch returns per-symbol dataframes with formatted timestamps."""
        import polars as pl

        mock_conn = Mock()
        mock_sm_instance = Mock()

        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })
        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_1"],
            "permno": [10516]
        }))
        mock_sm_instance.get_security_id = Mock()
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame({
            "permno": [10516],
            "date": [pd.Timestamp("2024-06-30")],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000000]
        })

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(["AAPL"], 2024, adjusted=True, auto_resolve=True)

        assert "AAPL" in result
        assert result["AAPL"]["timestamp"][0] == "2024-06-30"
        mock_sm_instance.get_security_id.assert_not_called()

    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_year_bulk_close(self, mock_logger, mock_security_master):
        """Close closes the WRDS connection."""
        mock_conn = Mock()

        crsp = CRSPDailyTicks(conn=mock_conn)
        crsp.close()

        mock_conn.close.assert_called_once()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_daily_ticks_year_bulk_unadjusted_query(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Bulk fetch uses unadjusted query when adjusted=False."""
        import polars as pl

        mock_conn = Mock()
        mock_sm_instance = Mock()

        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })
        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_1"],
            "permno": [10516]
        }))
        mock_sm_instance.get_security_id = Mock()
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        crsp.collect_daily_ticks_year_bulk(["AAPL"], 2024, adjusted=False, auto_resolve=True)

        query = mock_conn.raw_sql.call_args[0][0]
        assert 'cfacpr' not in query
        assert 'cfacshr' not in query


class TestGetDailyValidation:
    """Test get_daily validation and error handling"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_validation_permno_error(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily when permno validation fails"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 'invalid'
        mock_security_master.return_value = mock_sm_instance

        # Make validate_permno raise an error
        mock_validate_permno.side_effect = ValueError("Invalid permno")

        crsp = CRSPDailyTicks(conn=mock_conn)

        with pytest.raises(ValueError, match="Invalid permno"):
            crsp.get_daily('AAPL', '2024-06-30')

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_validation_date_error(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily when date validation fails"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        # Make validate_date_string raise an error
        mock_validate_date.side_effect = ValueError("Invalid date format")

        crsp = CRSPDailyTicks(conn=mock_conn)

        with pytest.raises(ValueError, match="Invalid date format"):
            crsp.get_daily('AAPL', 'invalid-date')


class TestGetDailyRangeValidation:
    """Test get_daily_range validation"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_get_daily_range_date_order(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test get_daily_range preserves date order in query"""
        mock_conn = Mock()

        mock_sm_instance = Mock()
        mock_sm_instance.get_security_id.return_value = 'sid_123'
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Mock data with correct date order
        mock_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-06-28'), pd.Timestamp('2024-06-29'), pd.Timestamp('2024-06-30')],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_conn.raw_sql.return_value = mock_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.get_daily_range(symbol='AAPL', start_day='2024-06-28', end_day='2024-06-30')

        # Verify dates are in ascending order
        assert len(result) == 3
        assert result[0]['timestamp'] == '2024-06-28'
        assert result[1]['timestamp'] == '2024-06-29'
        assert result[2]['timestamp'] == '2024-06-30'


class TestRecentDailyTicksChunking:
    """Test recent_daily_ticks chunking and edge cases"""

    @patch('quantdl.collection.crsp_ticks.raw_sql_with_retry')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_chunk_size_zero(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_raw_sql_with_retry):
        """Test recent_daily_ticks with chunk_size=0 (line 347)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock

        # Setup column access mocks for master_tb resolution
        symbol_col_mock = Mock()
        symbol_col_mock.to_list.return_value = ['AAPL']
        sid_col_mock = Mock()
        sid_col_mock.to_list.return_value = ['sid_123']
        final_mock.__getitem__.side_effect = lambda col: symbol_col_mock if col == 'symbol' else sid_col_mock

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique

        # Setup column access mocks for security_map resolution
        sid_col_mock2 = Mock()
        sid_col_mock2.to_list.return_value = ['sid_123']
        permno_col_mock = Mock()
        permno_col_mock.to_list.return_value = [10516]
        security_map_unique.__getitem__.side_effect = lambda col: sid_col_mock2 if col == 'security_id' else permno_col_mock

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_df = pd.DataFrame({
            'permno': [10516],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_raw_sql_with_retry.return_value = mock_df

        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        # chunk_size=0 should set chunk_size to len(permnos)
        result = crsp.recent_daily_ticks(['AAPL'], '2024-06-30', chunk_size=0)

        # Should make exactly 1 SQL query (all permnos in one chunk)
        assert mock_raw_sql_with_retry.call_count == 1
        assert isinstance(result, dict)

    @patch('quantdl.collection.crsp_ticks.raw_sql_with_retry')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_chunk_size_negative(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_raw_sql_with_retry):
        """Test recent_daily_ticks with negative chunk_size (line 347)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock

        # Setup column access mocks for master_tb resolution
        symbol_col_mock = Mock()
        symbol_col_mock.to_list.return_value = ['AAPL']
        sid_col_mock = Mock()
        sid_col_mock.to_list.return_value = ['sid_123']
        final_mock.__getitem__.side_effect = lambda col: symbol_col_mock if col == 'symbol' else sid_col_mock

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique

        # Setup column access mocks for security_map resolution
        sid_col_mock2 = Mock()
        sid_col_mock2.to_list.return_value = ['sid_123']
        permno_col_mock = Mock()
        permno_col_mock.to_list.return_value = [10516]
        security_map_unique.__getitem__.side_effect = lambda col: sid_col_mock2 if col == 'security_id' else permno_col_mock

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_df = pd.DataFrame({
            'permno': [10516],
            'date': [pd.Timestamp('2024-06-30')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [103.0],
            'volume': [1000000]
        })
        mock_raw_sql_with_retry.return_value = mock_df

        mock_polars_df = Mock()
        mock_filtered = Mock()
        mock_with_cols = Mock()
        mock_selected = MagicMock()

        mock_polars_df.filter.return_value = mock_filtered
        mock_filtered.with_columns.return_value = mock_with_cols
        mock_with_cols.select.return_value = mock_selected
        mock_selected.__len__.return_value = 1

        mock_from_pandas.return_value = mock_polars_df

        crsp = CRSPDailyTicks(conn=mock_conn)
        # chunk_size=-1 should set chunk_size to len(permnos)
        result = crsp.recent_daily_ticks(['AAPL'], '2024-06-30', chunk_size=-1)

        # Should make exactly 1 SQL query (all permnos in one chunk)
        assert mock_raw_sql_with_retry.call_count == 1
        assert isinstance(result, dict)

    @patch('quantdl.collection.crsp_ticks.raw_sql_with_retry')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_recent_daily_ticks_empty_frames_unadjusted(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_raw_sql_with_retry):
        """Test recent_daily_ticks with empty frames and unadjusted (line 382, 404)"""
        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock

        # Setup column access mocks for master_tb resolution
        symbol_col_mock = Mock()
        symbol_col_mock.to_list.return_value = ['AAPL']
        sid_col_mock = Mock()
        sid_col_mock.to_list.return_value = ['sid_123']
        final_mock.__getitem__.side_effect = lambda col: symbol_col_mock if col == 'symbol' else sid_col_mock

        mock_sm_instance.master_tb = mock_master_tb
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        security_map_mock = Mock()
        security_map_filtered = Mock()
        security_map_selected = Mock()
        security_map_unique = MagicMock()

        security_map_mock.filter.return_value = security_map_filtered
        security_map_filtered.select.return_value = security_map_selected
        security_map_selected.unique.return_value = security_map_unique

        # Setup column access mocks for security_map resolution
        sid_col_mock2 = Mock()
        sid_col_mock2.to_list.return_value = ['sid_123']
        permno_col_mock = Mock()
        permno_col_mock.to_list.return_value = [10516]
        security_map_unique.__getitem__.side_effect = lambda col: sid_col_mock2 if col == 'security_id' else permno_col_mock

        mock_sm_instance.security_map.return_value = security_map_mock
        mock_sm_instance.sid_to_permno.return_value = 10516
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Return empty DataFrame (no data in CRSP)
        mock_raw_sql_with_retry.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.recent_daily_ticks(['AAPL'], '2024-06-30', adjusted=False)

        # Should return empty dict
        assert result == {}
        # Verify unadjusted query was used
        query = mock_raw_sql_with_retry.call_args[0][1]
        assert 'cfacpr' not in query
        assert 'cfacshr' not in query

    @patch('quantdl.collection.crsp_ticks.raw_sql_with_retry')
    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    @patch('quantdl.collection.crsp_ticks.pl.from_pandas')
    def test_recent_daily_ticks_many_failed_symbols(self, mock_from_pandas, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date, mock_raw_sql_with_retry):
        """Test recent_daily_ticks with >20 failed symbols (lines 442-444)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()
        mock_master_tb = Mock()

        # Mock empty resolution (all symbols fail)
        filtered_mock = Mock()
        selected_mock = Mock()
        unique_mock = Mock()
        final_mock = MagicMock()

        mock_master_tb.filter.return_value = filtered_mock
        filtered_mock.select.return_value = selected_mock
        selected_mock.unique.return_value = unique_mock
        unique_mock.filter.return_value = final_mock
        final_mock.__getitem__.return_value.to_list.return_value = []  # No symbols resolved via fast path

        mock_sm_instance.master_tb = mock_master_tb
        # All auto_resolve attempts fail
        mock_sm_instance.get_security_id.side_effect = ValueError("Symbol not found")
        mock_security_master.return_value = mock_sm_instance

        # Create 25 invalid symbols
        failed_symbols = [f'INVALID{i}' for i in range(25)]

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.recent_daily_ticks(failed_symbols, '2024-06-30', auto_resolve=True)

        # Should return empty dict (all failed)
        assert result == {}
        # Verify error logging
        mock_logger_instance.error.assert_called_with("No symbols could be resolved to permnos")
        # Verify warning for failed symbols
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list if 'Failed symbols' in str(call)]
        assert len(warning_calls) >= 1
        # Check that "... and X more" message was logged
        more_message_calls = [call for call in mock_logger_instance.warning.call_args_list if '... and' in str(call)]
        assert len(more_message_calls) >= 1


class TestCollectDailyTicksYearBulkAdvanced:
    """Test collect_daily_ticks_year_bulk advanced scenarios"""

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_auto_resolve_path(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk with auto_resolve triggering fallback (lines 494, 499)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return empty (no fast path resolution)
        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": [],
            "start_date": [],
            "end_date": [],
            "security_id": []
        })

        # Auto-resolve should be called
        mock_sm_instance.get_security_id.return_value = 'sid_123'

        # Mock security_map
        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_123"],
            "permno": [10516]
        }))
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame({
            "permno": [10516],
            "date": [pd.Timestamp("2024-06-30")],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000000]
        })

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['AAPL'], 2024, auto_resolve=True)

        # Verify auto_resolve was called
        mock_sm_instance.get_security_id.assert_called_with('AAPL', '2024-12-31', auto_resolve=True)
        assert 'AAPL' in result

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_sid_none_raises(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk raises when security_id is None (line 501)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return empty
        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": [],
            "start_date": [],
            "end_date": [],
            "security_id": []
        })

        # Auto-resolve returns None
        mock_sm_instance.get_security_id.return_value = None
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['INVALID'], 2024, auto_resolve=True)

        # Should fail to resolve and return empty dict
        assert result == {}
        # Verify warning logged
        mock_logger_instance.warning.assert_called()

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_exception_handling(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk exception handling (lines 503-505)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return empty
        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": [],
            "start_date": [],
            "end_date": [],
            "security_id": []
        })

        # Auto-resolve raises exception
        mock_sm_instance.get_security_id.side_effect = RuntimeError("Database error")
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['AAPL'], 2024, auto_resolve=True)

        # Should fail gracefully and return empty dict
        assert result == {}
        # Verify warning logged with exception details
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list if 'Failed to resolve' in str(call)]
        assert len(warning_calls) >= 1

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_permno_resolution_fails(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk when permno resolution fails (lines 527-529)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return a result
        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })

        # security_map returns empty (permno not found)
        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": [],
            "permno": []
        }))
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['AAPL'], 2024)

        # Should fail to resolve permno and return empty dict
        assert result == {}
        # Verify warning logged
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list if 'Failed to resolve permno' in str(call)]
        assert len(warning_calls) >= 1

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_all_symbols_fail(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk when all symbols fail (lines 533-534)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return empty
        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": [],
            "start_date": [],
            "end_date": [],
            "security_id": []
        })

        # All symbols fail to resolve
        mock_sm_instance.get_security_id.side_effect = ValueError("Not found")
        mock_security_master.return_value = mock_sm_instance

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['INVALID1', 'INVALID2'], 2024, auto_resolve=True)

        # Should return empty dict
        assert result == {}
        # Verify error logged
        mock_logger_instance.error.assert_called_with("No symbols could be resolved to permnos")

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_chunk_size_zero(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk with chunk_size=0 (line 540)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })

        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_1"],
            "permno": [10516]
        }))
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame({
            "permno": [10516],
            "date": [pd.Timestamp("2024-06-30")],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000000]
        })

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['AAPL'], 2024, chunk_size=0)

        # Should make exactly 1 SQL query (all permnos in one chunk)
        assert mock_conn.raw_sql.call_count == 1
        assert 'AAPL' in result

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_empty_frames(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk with empty frames (line 595)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })

        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_1"],
            "permno": [10516]
        }))
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        # Return empty DataFrame (no data in CRSP)
        mock_conn.raw_sql.return_value = pd.DataFrame()

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(['AAPL'], 2024)

        # Should return empty dict
        assert result == {}

    @patch('quantdl.collection.crsp_ticks.validate_date_string')
    @patch('quantdl.collection.crsp_ticks.validate_permno')
    @patch('quantdl.collection.crsp_ticks.SecurityMaster')
    @patch('quantdl.collection.crsp_ticks.setup_logger')
    def test_collect_year_bulk_many_failed_symbols(self, mock_logger, mock_security_master, mock_validate_permno, mock_validate_date):
        """Test year bulk with >20 failed symbols (lines 629-631)"""
        import polars as pl

        mock_conn = Mock()
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        mock_sm_instance = Mock()

        # Mock master_tb to return one successful symbol
        end_day = "2024-12-31"
        date_check = pd.Timestamp(end_day).date()

        mock_sm_instance.master_tb = pl.DataFrame({
            "symbol": ["AAPL"],
            "start_date": [pd.Timestamp("2000-01-01").date()],
            "end_date": [date_check],
            "security_id": ["sid_1"]
        })

        mock_sm_instance.security_map = Mock(return_value=pl.DataFrame({
            "security_id": ["sid_1"],
            "permno": [10516]
        }))

        # All other symbols fail via auto_resolve
        def get_security_id_side_effect(symbol, date, auto_resolve):
            if symbol == 'AAPL':
                return 'sid_1'
            raise ValueError("Not found")

        mock_sm_instance.get_security_id.side_effect = get_security_id_side_effect
        mock_security_master.return_value = mock_sm_instance

        mock_validate_permno.return_value = 10516
        mock_validate_date.side_effect = lambda x: x

        mock_conn.raw_sql.return_value = pd.DataFrame({
            "permno": [10516],
            "date": [pd.Timestamp("2024-06-30")],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [1000000]
        })

        # Create 25 symbols: 1 valid (AAPL) + 24 invalid
        symbols = ['AAPL'] + [f'INVALID{i}' for i in range(24)]

        crsp = CRSPDailyTicks(conn=mock_conn)
        result = crsp.collect_daily_ticks_year_bulk(symbols, 2024, auto_resolve=True)

        # Should have AAPL but not the invalid ones
        assert 'AAPL' in result
        assert len(result) == 1

        # Verify warning for failed symbols
        warning_calls = [call for call in mock_logger_instance.warning.call_args_list if 'Failed symbols' in str(call)]
        assert len(warning_calls) >= 1

        # Check that "... and X more" message was logged
        more_message_calls = [call for call in mock_logger_instance.warning.call_args_list if '... and' in str(call)]
        assert len(more_message_calls) >= 1
