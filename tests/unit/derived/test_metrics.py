"""
Unit tests for derived.metrics module
Tests computation of derived fundamental metrics
"""
import pytest
import polars as pl
from quantdl.derived.metrics import compute_derived


class TestComputeDerived:
    """Test compute_derived function"""

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pl.DataFrame()
        result = compute_derived(empty_df)

        assert result.is_empty()

    def test_missing_required_columns(self):
        """Test with missing required columns"""
        df = pl.DataFrame({
            'symbol': ['AAPL'],
            'value': [1000000.0]
            # Missing: as_of_date, concept
        })

        result = compute_derived(df)
        assert result.is_empty()

    @pytest.fixture
    def sample_ttm_data(self):
        """Create sample TTM data for testing"""
        # Create 2 quarters of data with all required concepts
        dates = ['2024-06-30', '2024-09-30']

        data = {
            'symbol': ['AAPL'] * 32,  # 16 concepts × 2 quarters
            'as_of_date': dates * 16,
            'concept': (
                ['rev'] * 2 + ['cor'] * 2 + ['op_inc'] * 2 + ['net_inc'] * 2 +
                ['dna'] * 2 + ['std'] * 2 + ['ltd'] * 2 + ['cce'] * 2 +
                ['ca'] * 2 + ['cl'] * 2 + ['cfo'] * 2 + ['capex'] * 2 +
                ['ta'] * 2 + ['te'] * 2 + ['inc_tax_exp'] * 2 + ['ibt'] * 2
            ),
            'value': [
                # Revenue
                100000.0, 110000.0,
                # Cost of revenue
                60000.0, 65000.0,
                # Operating income
                30000.0, 35000.0,
                # Net income
                25000.0, 28000.0,
                # Depreciation and amortization
                5000.0, 5500.0,
                # Short-term debt
                10000.0, 11000.0,
                # Long-term debt
                50000.0, 52000.0,
                # Cash and cash equivalents
                20000.0, 22000.0,
                # Current assets
                80000.0, 85000.0,
                # Current liabilities
                50000.0, 52000.0,
                # Cash from operations
                35000.0, 38000.0,
                # CapEx
                15000.0, 16000.0,
                # Total assets
                200000.0, 210000.0,
                # Total equity
                120000.0, 125000.0,
                # Income tax expense
                8000.0, 9000.0,
                # Income before tax
                33000.0, 37000.0
            ]
        }

        return pl.DataFrame(data)

    def test_profitability_metrics(self, sample_ttm_data):
        """Test computation of profitability metrics"""
        result = compute_derived(sample_ttm_data)

        # Check that result is not empty
        assert not result.is_empty()

        # Get first quarter results
        q1 = result.filter(pl.col('as_of_date') == '2024-06-30')

        # Test gross profit = revenue - cost of revenue
        grs_pft = q1.filter(pl.col('metric') == 'grs_pft')
        assert len(grs_pft) == 1
        assert grs_pft['value'][0] == pytest.approx(40000.0)  # 100000 - 60000

        # Test gross margin = gross profit / revenue
        grs_mgn = q1.filter(pl.col('metric') == 'grs_mgn')
        assert len(grs_mgn) == 1
        assert grs_mgn['value'][0] == pytest.approx(0.4)  # 40000 / 100000

        # Test operating margin = operating income / revenue
        op_mgn = q1.filter(pl.col('metric') == 'op_mgn')
        assert len(op_mgn) == 1
        assert op_mgn['value'][0] == pytest.approx(0.3)  # 30000 / 100000

        # Test net margin = net income / revenue
        net_mgn = q1.filter(pl.col('metric') == 'net_mgn')
        assert len(net_mgn) == 1
        assert net_mgn['value'][0] == pytest.approx(0.25)  # 25000 / 100000

        # Test EBITDA = operating income + D&A
        ebitda = q1.filter(pl.col('metric') == 'ebitda')
        assert len(ebitda) == 1
        assert ebitda['value'][0] == pytest.approx(35000.0)  # 30000 + 5000

    def test_balance_sheet_metrics(self, sample_ttm_data):
        """Test computation of balance sheet metrics"""
        result = compute_derived(sample_ttm_data)
        q1 = result.filter(pl.col('as_of_date') == '2024-06-30')

        # Test total debt = short-term debt + long-term debt
        ttl_dbt = q1.filter(pl.col('metric') == 'ttl_dbt')
        assert len(ttl_dbt) == 1
        assert ttl_dbt['value'][0] == pytest.approx(60000.0)  # 10000 + 50000

        # Test net debt = total debt - cash
        net_dbt = q1.filter(pl.col('metric') == 'net_dbt')
        assert len(net_dbt) == 1
        assert net_dbt['value'][0] == pytest.approx(40000.0)  # 60000 - 20000

        # Test working capital = current assets - current liabilities
        wc = q1.filter(pl.col('metric') == 'wc')
        assert len(wc) == 1
        assert wc['value'][0] == pytest.approx(30000.0)  # 80000 - 50000

    def test_cash_flow_metrics(self, sample_ttm_data):
        """Test computation of cash flow metrics"""
        result = compute_derived(sample_ttm_data)
        q1 = result.filter(pl.col('as_of_date') == '2024-06-30')

        # Test free cash flow = CFO - CapEx
        fcf = q1.filter(pl.col('metric') == 'fcf')
        assert len(fcf) == 1
        assert fcf['value'][0] == pytest.approx(20000.0)  # 35000 - 15000

        # Test FCF margin = free cash flow / revenue
        fcf_mgn = q1.filter(pl.col('metric') == 'fcf_mgn')
        assert len(fcf_mgn) == 1
        assert fcf_mgn['value'][0] == pytest.approx(0.2)  # 20000 / 100000

        # Test CapEx ratio = capex / total assets
        capex_ratio = q1.filter(pl.col('metric') == 'capex_ratio')
        assert len(capex_ratio) == 1
        assert capex_ratio['value'][0] == pytest.approx(0.075)  # 15000 / 200000

    def test_return_metrics(self, sample_ttm_data):
        """Test computation of return metrics"""
        result = compute_derived(sample_ttm_data)
        q2 = result.filter(pl.col('as_of_date') == '2024-09-30')

        # Test effective tax rate = income tax expense / income before tax
        etr = q2.filter(pl.col('metric') == 'etr')
        assert len(etr) == 1
        assert etr['value'][0] == pytest.approx(9000.0 / 37000.0)

        # Test average assets (should use shift, so Q2 only)
        avg_ast = q2.filter(pl.col('metric') == 'avg_ast')
        assert len(avg_ast) == 1
        expected_avg_ast = (210000.0 + 200000.0) / 2  # Current + previous
        assert avg_ast['value'][0] == pytest.approx(expected_avg_ast)

        # Test average equity (should use shift, so Q2 only)
        avg_eqt = q2.filter(pl.col('metric') == 'avg_eqt')
        assert len(avg_eqt) == 1
        expected_avg_eqt = (125000.0 + 120000.0) / 2
        assert avg_eqt['value'][0] == pytest.approx(expected_avg_eqt)

        # Test ROA = net income / avg assets
        roa = q2.filter(pl.col('metric') == 'roa')
        assert len(roa) == 1
        assert roa['value'][0] == pytest.approx(28000.0 / expected_avg_ast)

        # Test ROE = net income / avg equity
        roe = q2.filter(pl.col('metric') == 'roe')
        assert len(roe) == 1
        assert roe['value'][0] == pytest.approx(28000.0 / expected_avg_eqt)

    def test_growth_metrics(self, sample_ttm_data):
        """Test computation of growth metrics"""
        result = compute_derived(sample_ttm_data)
        q2 = result.filter(pl.col('as_of_date') == '2024-09-30')

        # Test revenue growth = revenue(t) - revenue(t-1)
        rev_grw = q2.filter(pl.col('metric') == 'rev_grw')
        assert len(rev_grw) == 1
        assert rev_grw['value'][0] == pytest.approx(10000.0)  # 110000 - 100000

        # Test asset growth = total assets(t) - total assets(t-1)
        ast_grw = q2.filter(pl.col('metric') == 'ast_grw')
        assert len(ast_grw) == 1
        assert ast_grw['value'][0] == pytest.approx(10000.0)  # 210000 - 200000

    def test_accruals_metrics(self, sample_ttm_data):
        """Test computation of accruals metrics"""
        result = compute_derived(sample_ttm_data)
        q1 = result.filter(pl.col('as_of_date') == '2024-06-30')

        # Test accruals = net income - CFO
        acc = q1.filter(pl.col('metric') == 'acc')
        assert len(acc) == 1
        assert acc['value'][0] == pytest.approx(-10000.0)  # 25000 - 35000

    def test_null_handling(self):
        """Test that null values are handled correctly"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-06-30'] * 4,
            'concept': ['rev', 'cor', 'op_inc', 'net_inc'],
            'value': [100000.0, None, 30000.0, 25000.0]  # cor is None
        })

        result = compute_derived(df)

        # Gross profit should be None (requires cor)
        grs_pft = result.filter(pl.col('metric') == 'grs_pft')
        assert grs_pft.is_empty()  # Nulls are dropped

        # Gross margin should also be None
        grs_mgn = result.filter(pl.col('metric') == 'grs_mgn')
        assert grs_mgn.is_empty()

        # Operating margin should work (doesn't depend on cor)
        op_mgn = result.filter(pl.col('metric') == 'op_mgn')
        assert len(op_mgn) == 1
        assert op_mgn['value'][0] == pytest.approx(0.3)  # 30000 / 100000

    def test_division_by_zero(self):
        """Test that division by zero returns None"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 3,
            'as_of_date': ['2024-06-30'] * 3,
            'concept': ['rev', 'cor', 'net_inc'],
            'value': [0.0, 60000.0, 25000.0]  # Revenue is 0
        })

        result = compute_derived(df)

        # Gross margin should be None (division by zero)
        grs_mgn = result.filter(pl.col('metric') == 'grs_mgn')
        assert grs_mgn.is_empty()  # Null values are dropped

        # Net margin should also be None
        net_mgn = result.filter(pl.col('metric') == 'net_mgn')
        assert net_mgn.is_empty()

    def test_multiple_symbols(self):
        """Test computation for multiple symbols"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4 + ['MSFT'] * 4,
            'as_of_date': ['2024-06-30'] * 8,
            'concept': ['rev', 'cor', 'op_inc', 'net_inc'] * 2,
            'value': [
                100000.0, 60000.0, 30000.0, 25000.0,
                200000.0, 120000.0, 60000.0, 50000.0
            ]
        })

        result = compute_derived(df)

        # Check AAPL results
        aapl_results = result.filter(pl.col('symbol') == 'AAPL')
        aapl_grs_pft = aapl_results.filter(pl.col('metric') == 'grs_pft')
        assert len(aapl_grs_pft) == 1
        assert aapl_grs_pft['value'][0] == pytest.approx(40000.0)

        # Check MSFT results
        msft_results = result.filter(pl.col('symbol') == 'MSFT')
        msft_grs_pft = msft_results.filter(pl.col('metric') == 'grs_pft')
        assert len(msft_grs_pft) == 1
        assert msft_grs_pft['value'][0] == pytest.approx(80000.0)

    def test_output_format(self, sample_ttm_data):
        """Test that output is in long format with correct columns"""
        result = compute_derived(sample_ttm_data)

        # Check columns
        expected_columns = {'symbol', 'as_of_date', 'metric', 'value'}
        assert set(result.columns) == expected_columns

        # Check that values are not null
        assert result['value'].null_count() == 0

        # Check that metric names are correct (sample check)
        metrics = result['metric'].unique().to_list()
        expected_metrics = [
            'grs_pft', 'grs_mgn', 'op_mgn', 'net_mgn', 'ebitda',
            'ttl_dbt', 'net_dbt', 'wc',
            'fcf', 'fcf_mgn', 'capex_ratio'
        ]
        for expected_metric in expected_metrics:
            assert expected_metric in metrics

    def test_roic_computation(self, sample_ttm_data):
        """Test ROIC computation specifically"""
        result = compute_derived(sample_ttm_data)
        q2 = result.filter(pl.col('as_of_date') == '2024-09-30')

        # Get all components
        etr = q2.filter(pl.col('metric') == 'etr')['value'][0]
        op_inc = 35000.0  # From sample data

        # NOPAT = operating income × (1 − effective tax rate)
        expected_nopat = op_inc * (1 - etr)
        nopat = q2.filter(pl.col('metric') == 'nopat')
        assert len(nopat) == 1
        assert nopat['value'][0] == pytest.approx(expected_nopat)

        # Invested Capital = total equity + total debt - cash
        # = 125000 + (11000 + 52000) - 22000 = 166000
        inv_cap = q2.filter(pl.col('metric') == 'inv_cap')
        assert len(inv_cap) == 1
        assert inv_cap['value'][0] == pytest.approx(166000.0)

        # ROIC = NOPAT / invested capital
        roic = q2.filter(pl.col('metric') == 'roic')
        assert len(roic) == 1
        assert roic['value'][0] == pytest.approx(expected_nopat / 166000.0)

    def test_missing_concepts_create_nulls(self):
        """Test that missing input concepts result in missing derived metrics"""
        # Only provide revenue and net income, missing other concepts
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 2,
            'as_of_date': ['2024-06-30'] * 2,
            'concept': ['rev', 'net_inc'],
            'value': [100000.0, 25000.0]
        })

        result = compute_derived(df)

        # Net margin should work
        net_mgn = result.filter(pl.col('metric') == 'net_mgn')
        assert len(net_mgn) == 1

        # But gross profit shouldn't (missing cor)
        grs_pft = result.filter(pl.col('metric') == 'grs_pft')
        assert grs_pft.is_empty()

        # EBITDA shouldn't work (missing op_inc and dna)
        ebitda = result.filter(pl.col('metric') == 'ebitda')
        assert ebitda.is_empty()
