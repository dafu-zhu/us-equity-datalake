"""
Unit tests for universe.historical module
Tests historical universe retrieval functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import polars as pl
from quantdl.universe.historical import get_hist_universe_crsp, get_hist_universe_nasdaq


class TestGetHistUniverseCRSP:
    """Test get_hist_universe_crsp function"""

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    @patch('quantdl.universe.historical.wrds.Connection')
    def test_with_provided_connection(self, mock_wrds, mock_validate_date, mock_validate_month, mock_validate_year):
        """Test get_hist_universe_crsp with provided connection"""
        mock_conn = Mock()

        # Setup validation mocks
        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'

        # Mock SQL result
        mock_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'tsymbol': ['AAPL', 'MSFT', 'GOOGL'],
            'permno': [14593, 10107, 12345],
            'comnam': ['Apple Inc.', 'Microsoft Corp', 'Alphabet Inc.'],
            'shrcd': [10, 10, 10],
            'exchcd': [1, 1, 2]
        })
        mock_conn.raw_sql.return_value = mock_df

        result = get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        # Verify connection was used, not created
        mock_wrds.assert_not_called()

        # Verify SQL query was executed
        mock_conn.raw_sql.assert_called_once()

        # Verify connection was NOT closed
        mock_conn.close.assert_not_called()

        # Verify result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert 'Ticker' in result.columns
        assert 'Name' in result.columns
        assert 'PERMNO' in result.columns

    @patch.dict('os.environ', {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    @patch('quantdl.universe.historical.wrds.Connection')
    def test_without_provided_connection(self, mock_wrds, mock_validate_date, mock_validate_month, mock_validate_year):
        """Test get_hist_universe_crsp without provided connection"""
        mock_conn = Mock()
        mock_wrds.return_value = mock_conn

        # Setup validation mocks
        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'

        # Mock SQL result
        mock_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'permno': [14593],
            'comnam': ['Apple Inc.'],
            'shrcd': [10],
            'exchcd': [1]
        })
        mock_conn.raw_sql.return_value = mock_df

        result = get_hist_universe_crsp(year=2024, month=12, db=None)

        # Verify connection was created
        mock_wrds.assert_called_once_with(
            wrds_username='test_user',
            wrds_password='test_pass'
        )

        # Verify connection was closed
        mock_conn.close.assert_called_once()

    @patch.dict('os.environ', {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    @patch('quantdl.universe.historical.wrds.Connection')
    def test_closes_connection_on_exception(self, mock_wrds, mock_validate_date, mock_validate_month, mock_validate_year):
        """Connection created in function is closed on error."""
        mock_conn = Mock()
        mock_wrds.return_value = mock_conn
        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'
        mock_conn.raw_sql.side_effect = RuntimeError("db error")

        with pytest.raises(RuntimeError, match="db error"):
            get_hist_universe_crsp(year=2024, month=12, db=None)

        mock_conn.close.assert_called_once()

    @patch('quantdl.universe.historical.validate_year')
    def test_date_validation(self, mock_validate_year):
        """Test that year validation function is called"""
        mock_conn = Mock()

        # Setup validation mocks
        mock_validate_year.return_value = 2024

        # Mock SQL result
        mock_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'permno': [14593],
            'comnam': ['Apple Inc.'],
            'shrcd': [10],
            'exchcd': [1]
        })
        mock_conn.raw_sql.return_value = mock_df

        get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        # Verify year validation was called (month validation removed for survivorship-bias-free queries)
        mock_validate_year.assert_called_once_with(2024)

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    def test_does_not_close_provided_connection_on_error(self, mock_validate_date, mock_validate_month, mock_validate_year):
        """Provided connection should not be closed on error."""
        mock_conn = Mock()

        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'
        mock_conn.raw_sql.side_effect = RuntimeError("db error")

        with pytest.raises(RuntimeError, match="db error"):
            get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        mock_conn.close.assert_not_called()

    @patch('quantdl.universe.historical.validate_year')
    def test_sql_query_format(self, mock_validate_year):
        """Test that SQL query uses full year range for survivorship-bias-free results"""
        mock_conn = Mock()

        mock_validate_year.return_value = 2024

        mock_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'permno': [14593],
            'comnam': ['Apple Inc.'],
            'shrcd': [10],
            'exchcd': [1]
        })
        mock_conn.raw_sql.return_value = mock_df

        get_hist_universe_crsp(year=2024, month=6, db=mock_conn)

        # Verify SQL query uses full year range (not point-in-time) to avoid survivorship bias
        call_args = mock_conn.raw_sql.call_args[0][0]
        assert '2024-01-01' in call_args  # Year start
        assert '2024-12-31' in call_args  # Year end
        assert 'shrcd IN (10, 11)' in call_args
        assert 'exchcd IN (1, 2, 3)' in call_args

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    def test_uppercase_conversion(self, mock_validate_date, mock_validate_month, mock_validate_year):
        """Test that tickers are converted to uppercase"""
        mock_conn = Mock()

        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'

        # Mock SQL result with lowercase tickers
        mock_df = pd.DataFrame({
            'ticker': ['aapl', 'msft'],
            'tsymbol': ['aapl', 'msft'],
            'permno': [14593, 10107],
            'comnam': ['Apple Inc.', 'Microsoft Corp'],
            'shrcd': [10, 10],
            'exchcd': [1, 1]
        })
        mock_conn.raw_sql.return_value = mock_df

        result = get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        # Verify tickers are uppercase
        assert result['Ticker'][0] == 'AAPL'
        assert result['Ticker'][1] == 'MSFT'

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    def test_duplicate_removal(self, mock_validate_date, mock_validate_month, mock_validate_year):
        """Test that duplicate tickers are removed"""
        mock_conn = Mock()

        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'

        # Mock SQL result with duplicate tickers
        mock_df = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT'],
            'tsymbol': ['AAPL', 'AAPL', 'MSFT'],
            'permno': [14593, 14593, 10107],
            'comnam': ['Apple Inc.', 'Apple Inc.', 'Microsoft Corp'],
            'shrcd': [10, 10, 10],
            'exchcd': [1, 1, 1]
        })
        mock_conn.raw_sql.return_value = mock_df

        result = get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        # Verify duplicates are removed
        assert len(result) == 2
        assert result['Ticker'][0] == 'AAPL'
        assert result['Ticker'][1] == 'MSFT'


class TestGetHistUniverseNasdaq:
    """Test get_hist_universe_nasdaq function"""

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    def test_without_validation(self, mock_normalizer_class, mock_get_crsp):
        """Test get_hist_universe_nasdaq without validation"""
        # Mock CRSP data
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL', 'BRKB', 'GOOGL'],
            'Name': ['Apple Inc.', 'Berkshire Hathaway', 'Alphabet Inc.'],
            'PERMNO': [14593, 17778, 12345]
        })
        mock_get_crsp.return_value = mock_crsp_df

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL', 'BRK.B', 'GOOGL']
        mock_normalizer_class.return_value = mock_normalizer

        result = get_hist_universe_nasdaq(year=2024, with_validation=False)

        # Verify get_hist_universe_crsp was called
        mock_get_crsp.assert_called_once_with(2024, month=12, db=None)

        # Verify normalizer was created without security_master
        mock_normalizer_class.assert_called_once_with()

        # Verify batch_normalize was called without date
        mock_normalizer.batch_normalize.assert_called_once()
        call_args = mock_normalizer.batch_normalize.call_args
        assert call_args[0][0] == ['AAPL', 'BRKB', 'GOOGL']

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    def test_with_validation_and_security_master(self, mock_normalizer_class, mock_get_crsp):
        """Test get_hist_universe_nasdaq with validation and provided SecurityMaster"""
        # Mock CRSP data
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL', 'BRKB'],
            'Name': ['Apple Inc.', 'Berkshire Hathaway'],
            'PERMNO': [14593, 17778]
        })
        mock_get_crsp.return_value = mock_crsp_df

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL', 'BRK.B']
        mock_normalizer_class.return_value = mock_normalizer

        # Mock security master
        mock_sm = Mock()

        result = get_hist_universe_nasdaq(
            year=2024,
            with_validation=True,
            security_master=mock_sm
        )

        # Verify normalizer was created with security_master
        mock_normalizer_class.assert_called_once_with(security_master=mock_sm)

        # Provided security master should not be closed
        mock_sm.close.assert_not_called()

        # Result aligns with CRSP data length
        assert len(result) == 2

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    @patch('quantdl.universe.historical.SecurityMaster')
    def test_with_validation_without_security_master(self, mock_sm_class, mock_normalizer_class, mock_get_crsp):
        """Test get_hist_universe_nasdaq with validation but no provided SecurityMaster"""
        # Mock CRSP data
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL'],
            'Name': ['Apple Inc.'],
            'PERMNO': [14593]
        })
        mock_get_crsp.return_value = mock_crsp_df

        # Mock normalizer
        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL']
        mock_normalizer_class.return_value = mock_normalizer

        # Mock SecurityMaster
        mock_sm = Mock()
        mock_sm_class.return_value = mock_sm

        # Mock db connection
        mock_db = Mock()

        result = get_hist_universe_nasdaq(
            year=2024,
            with_validation=True,
            db=mock_db
        )

        # Verify SecurityMaster was created without db in this code path
        mock_sm_class.assert_called_once_with()

        # Verify normalizer was created with created SecurityMaster
        mock_normalizer_class.assert_called_once_with(security_master=mock_sm)

        # db is passed through to CRSP call
        mock_get_crsp.assert_called_once_with(2024, month=12, db=mock_db)

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    def test_uses_year_end_date(self, mock_normalizer_class, mock_get_crsp):
        """Test that function uses December (month=12) for year-end"""
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL'],
            'Name': ['Apple Inc.'],
            'PERMNO': [14593]
        })
        mock_get_crsp.return_value = mock_crsp_df

        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL']
        mock_normalizer_class.return_value = mock_normalizer

        get_hist_universe_nasdaq(year=2024, with_validation=False)

        # Verify month=12 was used
        mock_get_crsp.assert_called_once_with(2024, month=12, db=None)

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    @patch('quantdl.universe.historical.SecurityMaster')
    def test_with_validation_closes_created_security_master(self, mock_sm_class, mock_normalizer_class, mock_get_crsp):
        """Created SecurityMaster is closed after use."""
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL'],
            'Name': ['Apple Inc.'],
            'PERMNO': [14593]
        })
        mock_get_crsp.return_value = mock_crsp_df

        mock_sm = Mock()
        mock_sm_class.return_value = mock_sm

        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL']
        mock_normalizer_class.return_value = mock_normalizer

        get_hist_universe_nasdaq(year=2024, with_validation=True, db=None)

        mock_sm.close.assert_called_once()
        mock_normalizer.batch_normalize.assert_called_once_with(['AAPL'], day='2024-12-31')

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    def test_empty_crsp_universe(self, mock_normalizer_class, mock_get_crsp):
        """Empty CRSP input returns empty Nasdaq universe."""
        mock_crsp_df = pl.DataFrame({
            'Ticker': [],
            'Name': [],
            'PERMNO': []
        })
        mock_get_crsp.return_value = mock_crsp_df

        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = []
        mock_normalizer_class.return_value = mock_normalizer

        result = get_hist_universe_nasdaq(year=2024, with_validation=False)

        assert result.is_empty()

    @patch('quantdl.universe.historical.get_hist_universe_crsp')
    @patch('quantdl.universe.historical.SymbolNormalizer')
    def test_without_validation_uses_no_reference_day(self, mock_normalizer_class, mock_get_crsp):
        """Without validation, batch_normalize is called without day."""
        mock_crsp_df = pl.DataFrame({
            'Ticker': ['AAPL'],
            'Name': ['Apple Inc.'],
            'PERMNO': [14593]
        })
        mock_get_crsp.return_value = mock_crsp_df

        mock_normalizer = Mock()
        mock_normalizer.batch_normalize.return_value = ['AAPL']
        mock_normalizer_class.return_value = mock_normalizer

        get_hist_universe_nasdaq(year=2024, with_validation=False)

        mock_normalizer.batch_normalize.assert_called_once_with(['AAPL'], day=None)


class TestGetHistUniverseEdgeCases:
    """Test edge cases for historical universe functions"""

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    @patch('quantdl.universe.historical.validate_date_string')
    def test_empty_result(self, mock_validate_date, mock_validate_month, mock_validate_year):
        """Test handling of empty query results"""
        mock_conn = Mock()

        mock_validate_year.return_value = 2024
        mock_validate_month.return_value = 12
        mock_validate_date.return_value = '2024-12-28'

        # Mock empty SQL result
        mock_df = pd.DataFrame({
            'ticker': [],
            'tsymbol': [],
            'permno': [],
            'comnam': [],
            'shrcd': [],
            'exchcd': []
        })
        mock_conn.raw_sql.return_value = mock_df

        result = get_hist_universe_crsp(year=2024, month=12, db=mock_conn)

        assert len(result) == 0
        assert 'Ticker' in result.columns

    @patch('quantdl.universe.historical.validate_year')
    @patch('quantdl.universe.historical.validate_month')
    def test_validation_error_propagation(self, mock_validate_month, mock_validate_year):
        """Test that validation errors are propagated"""
        mock_conn = Mock()

        # Make validation raise an error
        mock_validate_year.side_effect = ValueError("Invalid year: 3000")

        with pytest.raises(ValueError, match="Invalid year"):
            get_hist_universe_crsp(year=3000, month=12, db=mock_conn)
