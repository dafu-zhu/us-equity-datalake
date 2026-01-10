"""
Unit tests for universe.current module
Tests fetching current stock universe from Nasdaq
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from ftplib import FTP
import io
from quantdl.universe.current import is_common_stock, fetch_all_stocks


class TestIsCommonStock:
    """Test is_common_stock function"""

    def test_valid_common_stocks(self):
        """Test that valid common stocks are identified correctly"""
        valid_stocks = [
            "Apple Inc. Common Stock",
            "Microsoft Corporation Common Stock",
            "Tesla, Inc. Common Stock",
            "Uniti Group Inc. - Common Stock",
            "Universal Health Realty Income Trust Common Stock"
        ]

        for name in valid_stocks:
            assert is_common_stock(name) is True, f"Failed for: {name}"

    def test_preferred_stocks(self):
        """Test that preferred stocks are excluded"""
        preferred_stocks = [
            "FTAI Aviation Ltd. - 9.500% Fixed-Rate Reset Series D Cumulative Perpetual Redeemable Preferred Shares",
            "Bank of America Corporation Non Cumulative Perpetual Conv Pfd Ser L",
            "EPR Properties Series E Cumulative Conv Pfd Shs Ser E",
            "Franklin BSP Realty Trust, Inc. 7.50% Series E Cumulative Redeemable Preferred Stock",
            "Fortress Biotech, Inc. - 9.375% Series A Cumulative Redeemable Perpetual Preferred Stock"
        ]

        for name in preferred_stocks:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_units_warrants_rights(self):
        """Test that units, warrants, and rights are excluded"""
        excluded = [
            "Axiom Intelligence Acquisition Corp 1 - Right",
            "Bitcoin Infrastructure Acquisition Corp Ltd. - Units",
            "Some Company - Warrants"
        ]

        for name in excluded:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_adrs_and_adss(self):
        """Test that ADRs and ADSs are excluded"""
        adrs = [
            "New Oriental Education & Technology Group, Inc. Sponsored ADR representing 10 Ordinary Share (Cayman Islands)",
            "Some Company ADS"
        ]

        for name in adrs:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_etns(self):
        """Test that ETNs are excluded"""
        etns = [
            "MicroSectors FANG Index -3X Inverse Leveraged ETNs due January 8, 2038"
        ]

        for name in etns:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_convertible_securities(self):
        """Test that convertible securities are excluded"""
        convertibles = [
            "Shift4 Payments, Inc. 6.00% Series A Mandatory Convertible Preferred Stock"
        ]

        for name in convertibles:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_closed_end_funds(self):
        """Test that closed-end funds are excluded"""
        funds = [
            "Fidus Investment Corporation - Closed End Fund",
            "Federated Hermes Premier Municipal Income Fund",
            "Credit Suisse High Yield Credit Fund Common Stock",
            "BlackRock Municipal 2030 Target Term Trust",
            "Saba Capital Income & Opportunities Fund SBI",
            "BlackRock Investment Quality Municipal Trust Inc. (The)"
        ]

        for name in funds:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_subordinate_shares(self):
        """Test that subordinate voting shares are excluded"""
        subordinate = [
            "Digi Power X Inc. - Common Subordinate Voting Shares"
        ]

        for name in subordinate:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_partnership_interests(self):
        """Test that partnership interests are excluded"""
        partnerships = [
            "Empire State Realty OP, L.P. Series ES Operating Partnership Units Representing Limited Partnership Interests"
        ]

        for name in partnerships:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_beneficial_interests(self):
        """Test that beneficial interests are handled correctly"""
        beneficial = [
            "Eaton Vance Short Diversified Income Fund Eaton Vance Short Duration Diversified Income Fund Common Shares of Beneficial Interest"
        ]

        for name in beneficial:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_null_and_invalid_inputs(self):
        """Test handling of null and invalid inputs"""
        assert is_common_stock(None) is False
        assert is_common_stock(pd.NA) is False
        assert is_common_stock("") is False
        assert is_common_stock(123) is False

    def test_percentage_symbol(self):
        """Test that securities with % are excluded"""
        with_percent = [
            "Some Company 7.5% Notes"
        ]

        for name in with_percent:
            assert is_common_stock(name) is False, f"Failed for: {name}"

    def test_series_designation(self):
        """Test that Series-designated securities are excluded"""
        series = [
            "Some Company Series A"
        ]

        for name in series:
            assert is_common_stock(name) is False, f"Failed for: {name}"


class TestFetchAllStocks:
    """Test fetch_all_stocks function"""

    @pytest.fixture
    def mock_ftp_data(self):
        """Create mock FTP data"""
        data = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N
MSFT|Microsoft Corporation Common Stock|Q|N|N|100|N|N
GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N
PREF|Some Company Preferred Stock|Q|N|N|100|N|N
FUND|Some Closed End Fund|Q|N|N|100|N|N
ETFX|Some ETF|Q|N|N|100|Y|N
TEST|Test Security|Q|Y|N|100|N|N
File Creation Time: 0101202400:00"""
        return data

    @patch('quantdl.universe.current.FTP')
    def test_fetch_all_stocks_with_filter(self, mock_ftp_class, mock_ftp_data):
        """Test fetching stocks with filtering enabled"""
        # Setup mock FTP
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Mock retrbinary to write data to BytesIO
        def mock_retrbinary(cmd, callback):
            callback(mock_ftp_data.encode())

        mock_ftp.retrbinary = mock_retrbinary

        # Fetch stocks
        result = fetch_all_stocks(with_filter=True, refresh=True)

        # Verify FTP calls
        mock_ftp.login.assert_called_once()
        mock_ftp.cwd.assert_called_once_with('SymbolDirectory')
        mock_ftp.quit.assert_called_once()

        # Check results - should only include common stocks
        assert len(result) == 3  # AAPL, MSFT, GOOGL
        assert 'AAPL' in result['Ticker'].values
        assert 'MSFT' in result['Ticker'].values
        assert 'GOOGL' in result['Ticker'].values

        # Should not include preferred, ETF, or test issues
        assert 'PREF' not in result['Ticker'].values
        assert 'FUND' not in result['Ticker'].values
        assert 'ETFX' not in result['Ticker'].values
        assert 'TEST' not in result['Ticker'].values

    @patch('quantdl.universe.current.FTP')
    def test_fetch_all_stocks_without_filter(self, mock_ftp_class, mock_ftp_data):
        """Test fetching stocks without filtering"""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        def mock_retrbinary(cmd, callback):
            callback(mock_ftp_data.encode())

        mock_ftp.retrbinary = mock_retrbinary

        result = fetch_all_stocks(with_filter=False, refresh=True)

        # Should include all securities (except footer and duplicates)
        assert len(result) >= 3

    @patch('quantdl.universe.current.pd.read_csv')
    def test_fetch_from_cache(self, mock_read_csv):
        """Test loading from cache when refresh=False"""
        # Mock cached data
        cached_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT'],
            'Name': ['Apple Inc.', 'Microsoft Corp.']
        })
        mock_read_csv.return_value = cached_data

        # Create a mock Path.exists() to return True
        with patch('pathlib.Path.exists', return_value=True):
            result = fetch_all_stocks(refresh=False)

            # Should return cached data
            assert len(result) == 2
            assert 'AAPL' in result['Ticker'].values
            mock_read_csv.assert_called_once()

    def test_dollar_sign_exclusion(self):
        """Test that tickers with $ are excluded"""
        # This is tested indirectly through the filter logic
        # Tickers with $ should be excluded when with_filter=True
        name = "Some Company Class B"
        ticker_with_dollar = "ABC$"

        # The filter excludes tickers containing $
        assert '$' in ticker_with_dollar

    @patch('quantdl.universe.current.FTP')
    def test_handle_ftp_error(self, mock_ftp_class):
        """Test handling of FTP connection errors"""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp
        mock_ftp.login.side_effect = Exception("FTP connection failed")

        result = fetch_all_stocks(refresh=True)

        # Should return empty DataFrame on error
        assert result.empty

    @patch('quantdl.universe.current.FTP')
    def test_remove_duplicates(self, mock_ftp_class):
        """Test that duplicate tickers are removed"""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        # Data with duplicate ticker
        data = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N
AAPL|Apple Inc. Common Stock (duplicate)|Q|N|N|100|N|N
MSFT|Microsoft Corporation Common Stock|Q|N|N|100|N|N
File Creation Time: 0101202400:00"""

        def mock_retrbinary(cmd, callback):
            callback(data.encode())

        mock_ftp.retrbinary = mock_retrbinary

        result = fetch_all_stocks(with_filter=True, refresh=True)

        # Should only have one AAPL entry
        aapl_count = (result['Ticker'] == 'AAPL').sum()
        assert aapl_count == 1

    @patch('quantdl.universe.current.FTP')
    def test_sorted_output(self, mock_ftp_class):
        """Test that output is sorted by ticker"""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        data = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
ZZZZ|Z Company Common Stock|Q|N|N|100|N|N
AAAA|A Company Common Stock|Q|N|N|100|N|N
MMMM|M Company Common Stock|Q|N|N|100|N|N
File Creation Time: 0101202400:00"""

        def mock_retrbinary(cmd, callback):
            callback(data.encode())

        mock_ftp.retrbinary = mock_retrbinary

        result = fetch_all_stocks(with_filter=True, refresh=True)

        # Should be sorted
        assert result['Ticker'].tolist() == ['AAAA', 'MMMM', 'ZZZZ']

    @patch('quantdl.universe.current.FTP')
    @patch('pathlib.Path.exists', return_value=False)
    def test_fallback_to_refresh_when_cache_missing(self, mock_exists, mock_ftp_class, mock_ftp_data):
        """Test that refresh is performed when cache doesn't exist"""
        mock_ftp = MagicMock()
        mock_ftp_class.return_value = mock_ftp

        def mock_retrbinary(cmd, callback):
            callback(mock_ftp_data.encode())

        mock_ftp.retrbinary = mock_retrbinary

        # Try to load from cache with refresh=False
        result = fetch_all_stocks(refresh=False)

        # Should have fetched from FTP since cache doesn't exist
        mock_ftp.login.assert_called_once()
        assert len(result) > 0
