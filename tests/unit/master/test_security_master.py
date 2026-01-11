"""
Unit tests for master.security_master module
Tests symbol normalization and security master functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
import datetime as dt
from quantdl.master.security_master import SymbolNormalizer, SecurityMaster


class TestSymbolNormalizer:
    """Test SymbolNormalizer class"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization(self, mock_logger, mock_fetch):
        """Test SymbolNormalizer initialization"""
        # Mock current stock list
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'BRK.B', 'GOOGL', 'ABC.D']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Verify fetch was called
        mock_fetch.assert_called_once_with(with_filter=True, refresh=False)

        # Verify symbol map was created
        assert 'AAPL' in normalizer.sym_map
        assert 'BRKB' in normalizer.sym_map  # Dots removed
        assert 'GOOGL' in normalizer.sym_map
        assert 'ABCD' in normalizer.sym_map  # Dots removed

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_fetch):
        """Test initialization with SecurityMaster instance"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.security_master == mock_sm

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_simple_match(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with simple symbol match"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Test uppercase conversion
        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('MSFT') == 'MSFT'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_separator(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with symbols containing separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B', 'ABC-D']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # BRKB should normalize to BRK.B
        assert normalizer.to_nasdaq_format('BRKB') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK.B') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK-B') == 'BRK.B'

        # ABCD should normalize to ABC-D
        assert normalizer.to_nasdaq_format('ABCD') == 'ABC-D'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_not_in_current_list(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with delisted symbol not in current list"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Symbol not in current list should return as-is (uppercased)
        assert normalizer.to_nasdaq_format('DELISTD') == 'DELISTD'
        assert normalizer.to_nasdaq_format('oldstock') == 'OLDSTOCK'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_same_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (same security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock same security_id for both dates
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB', day='2024-01-01')

        # Should return Nasdaq format since same security
        assert result == 'BRK.B'

        # Verify get_security_id was called twice
        assert mock_sm.get_security_id.call_count == 2

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_different_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (different security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['ABC.D']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock different security_ids
        mock_sm.get_security_id.side_effect = ['security_old', 'security_new']

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('ABCD', day='2022-01-01')

        # Should return original format since different security
        assert result == 'ABCD'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_validation_error(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format when validation raises error"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.side_effect = ValueError("Symbol not found")

        normalizer = SymbolNormalizer(security_master=mock_sm)

        # Should return original format when validation fails
        result = normalizer.to_nasdaq_format('AAPL', day='2024-01-01')
        assert result == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_empty_symbol(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with empty symbol"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('') == ''
        assert normalizer.to_nasdaq_format(None) is None

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_no_date_context(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format without date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB')

        # Should return Nasdaq format without validation
        assert result == 'BRK.B'

        # Verify get_security_id was NOT called
        mock_sm.get_security_id.assert_not_called()

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize(self, mock_logger, mock_fetch):
        """Test batch_normalize"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        symbols = ['aapl', 'BRKB', 'googl']
        result = normalizer.batch_normalize(symbols)

        assert result == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize_with_date(self, mock_logger, mock_fetch):
        """Test batch_normalize with date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        symbols = ['AAPL', 'BRKB']
        result = normalizer.batch_normalize(symbols, day='2024-01-01')

        assert result == ['AAPL', 'BRK.B']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_day_no_security_master(self, mock_logger, mock_fetch):
        """Date context without SecurityMaster uses Nasdaq format."""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer(security_master=None)

        assert normalizer.to_nasdaq_format('BRKB', day='2024-01-01') == 'BRK.B'

    def test_to_crsp_format(self):
        """Test to_crsp_format static method"""
        # Test various input formats
        assert SymbolNormalizer.to_crsp_format('BRK.B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('BRK-B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('ABC.D.E') == 'ABCDE'
        assert SymbolNormalizer.to_crsp_format('AAPL') == 'AAPL'
        assert SymbolNormalizer.to_crsp_format('aapl') == 'AAPL'

    def test_to_sec_format(self):
        """Test to_sec_format static method"""
        assert SymbolNormalizer.to_sec_format('BRK.B') == 'BRK-B'
        assert SymbolNormalizer.to_sec_format('aapl') == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_handles_non_string_tickers(self, mock_logger, mock_fetch):
        """Test that initialization handles non-string tickers gracefully"""
        # Include NaN and None values
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', None, 'MSFT', 'GOOGL']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Should skip non-string values
        assert 'AAPL' in normalizer.sym_map
        assert 'MSFT' in normalizer.sym_map
        assert 'GOOGL' in normalizer.sym_map
        # None should not cause issues


class TestSymbolNormalizerEdgeCases:
    """Test edge cases for SymbolNormalizer"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_case_insensitivity(self, mock_logger, mock_fetch):
        """Test that symbol matching is case-insensitive"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('AaPl') == 'AAPL'
        assert normalizer.to_nasdaq_format('brkb') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BrK.b') == 'BRK.B'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_multiple_separators(self, mock_logger, mock_fetch):
        """Test symbols with multiple separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['A.B.C', 'X-Y-Z']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('ABC') == 'A.B.C'
        assert normalizer.to_nasdaq_format('XYZ') == 'X-Y-Z'


class TestSecurityMaster:
    """Test SecurityMaster core behaviors with injected data"""

    def test_get_security_id_exact_match(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=True)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_no_match_no_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="Symbol BBB not found"):
            sm.get_security_id('BBB', '2020-06-30', auto_resolve=False)

    def test_get_security_id_no_match_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=555)

        result = sm.get_security_id('BBB', '2020-06-30', auto_resolve=True)

        assert result == 555
        sm.auto_resolve.assert_called_once_with('BBB', '2020-06-30')

    def test_get_security_id_exact_match_no_auto_resolve_flag(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_rejects_none_security_id(self):
        master_tb = pl.DataFrame({
            'security_id': [None],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="security_id is None"):
            sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

    def test_auto_resolve_selects_closest_symbol_usage(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2018, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2022, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2

    def test_auto_resolve_symbol_never_existed(self):
        master_tb = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="never existed"):
            sm.auto_resolve('ZZZ', '2020-06-15')

    def test_sid_to_permno(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [555]
        }))

        result = sm.sid_to_permno(101)

        assert result == 555
    
    def test_sid_to_permno_none(self):
        sm = SecurityMaster.__new__(SecurityMaster)

        with pytest.raises(ValueError, match="security_id is None"):
            sm.sid_to_permno(None)

    def test_sid_to_info(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb

        result = sm.sid_to_info(101, '2020-06-30', info='company')

        assert result == 'TestCo'

    def test_get_symbol_history(self):
        master_tb = pl.DataFrame({
            'security_id': [101, 101],
            'symbol': ['AAA', 'BBB'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_table = Mock(return_value=master_tb)

        result = sm.get_symbol_history(101)

        assert set(result) == {
            ('AAA', '2020-01-01', '2020-12-31'),
            ('BBB', '2021-01-01', '2021-12-31')
        }

    def test_auto_resolve_filters_null_security_ids(self):
        master_tb = pl.DataFrame({
            'security_id': [None, 101],
            'symbol': ['AAA', 'AAA'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_logs_error_on_sid_to_info_failure(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(side_effect=RuntimeError("boom"))

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.error.assert_called_once()

    def test_auto_resolve_no_active_security(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="was not active"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_auto_resolve_multiple_candidates_selects_closest(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 6, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2
        sm.logger.info.assert_called()

    def test_fetch_sec_cik_mapping_cached(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        cached = pl.DataFrame({'ticker': ['AAA'], 'cik': ['0000000001']})
        sm._sec_cik_cache = cached

        with patch('quantdl.master.security_master.requests.get') as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert result is cached
        mock_get.assert_not_called()

    def test_fetch_sec_cik_mapping_success(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "BRK.B", "cik_str": 123456},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response) as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert not result.is_empty()
        assert sm._sec_cik_cache is result
        mock_get.assert_called_once()

    def test_fetch_sec_cik_mapping_failure(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        with patch('quantdl.master.security_master.requests.get', side_effect=Exception("boom")):
            result = sm._fetch_sec_cik_mapping()

        assert result.is_empty()
        assert set(result.columns) == {"ticker", "cik"}

    def test_fetch_sec_cik_mapping_filters_zero_cik(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "ZERO", "cik_str": 0},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response):
            result = sm._fetch_sec_cik_mapping()

        tickers = result['ticker'].to_list()
        assert "ZERO" not in tickers
        assert "AAPL" in tickers

    def test_cik_cusip_mapping_sec_fallback_unavailable(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': [None],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': [], 'cik': []}))

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 1
        sm.logger.warning.assert_called()

    def test_cik_cusip_mapping_sec_fallback_fills_nulls(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1, 2],
            'ticker': ['AAA', 'BBB'],
            'tsymbol': ['AAA', 'BBB'],
            'comnam': ['AAA Corp', 'BBB Corp'],
            'ncusip': ['12345678', '87654321'],
            'cik': [None, '0000000002'],
            'cikdate1': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({
            'ticker': ['AAA'],
            'cik': ['0000000001']
        }))

        result = sm.cik_cusip_mapping()

        filled = result.filter(pl.col('symbol') == 'AAA').select('cik').item()
        assert filled == '0000000001'

    def test_cik_cusip_mapping_no_fallback_needed(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()
        sm._fetch_sec_cik_mapping = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 0
        sm._fetch_sec_cik_mapping.assert_not_called()

    def test_security_map_new_business_on_symbol_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0002'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_security_map_same_business_with_cik_overlap(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 1

    def test_security_map_new_business_on_permno_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 2],
            'symbol': ['AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '11111111'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_master_table_includes_security_id(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        }))

        result = sm.master_table()

        assert 'security_id' in result.columns
        assert result.select('security_id').item() == 101

    def test_master_table_preserves_security_id_with_null_cik(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        }))

        result = sm.master_table()

        assert result.select('security_id').item() == 101
