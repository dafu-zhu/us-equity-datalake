# Derived Fundamental Data

This document describes the derived fundamental metrics computed from raw fundamental data.

## Overview

Derived fundamental data is computed from raw fundamental data collected from SEC EDGAR. These derived metrics provide additional insights into company performance, profitability, efficiency, and growth.

**Storage Structure:**
- **Input:** `data/raw/fundamental/{symbol}/fundamental.parquet`
- **Output:** `data/derived/fundamental/{symbol}/{YYYY}/fundamental.parquet`

## Derived Concepts

All formulas are defined in `data/xbrl/fundamental.xlsx` (Priority = 3).

### 1. Profitability Metrics

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `grs_pft` | Gross Profit | `revenue - cost of revenue` | Revenue minus direct costs |
| `grs_mgn` | Gross Margin | `gross profit / revenue` | Gross profit as % of revenue |
| `op_mgn` | Operating Margin | `operating income / revenue` | Operating efficiency metric |
| `net_mgn` | Net Margin | `net income / revenue` | Bottom-line profitability |
| `ebitda` | EBITDA | `operating income + depreciation and amortization` | Earnings before interest, taxes, depreciation, and amortization |

### 2. Cash Flow Metrics

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `fcf` | Free Cash Flow | `CFO - CapEx` | Cash available after capital expenditures |
| `fcf_mgn` | FCF Margin | `free cash flow / revenue` | FCF as % of revenue |
| `capex_ratio` | CapEx Ratio | `capex / total assets` | Capital intensity metric |

### 3. Balance Sheet Constructs

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `ttl_dbt` | Total Debt | `short term debt + long term debt` | Combined debt obligations |
| `net_dbt` | Net Debt | `total debt - cash` | Debt net of cash position |
| `wc` | Working Capital | `current assets - current liabilities` | Short-term liquidity metric |

### 4. Return Metrics

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `roa` | Return on Assets | `net income / avg assets` | Profitability relative to assets |
| `avg_ast` | Average Assets | `(total assets(t) + total assets(t-1)) / 2` | Two-period average assets |
| `roe` | Return on Equity | `net income / avg equity` | Profitability relative to equity |
| `avg_eqt` | Average Equity | `(total equity(t) + total equity(t-1)) / 2` | Two-period average equity |
| `roic` | Return on Invested Capital | `NOPAT / invested capital` | Return on capital employed |
| `nopat` | NOPAT | `operating income × (1 − effective tax rate)` | Net operating profit after tax |
| `etr` | Effective Tax Rate | `income tax expense / income before tax` | Actual tax rate paid |
| `inv_cap` | Invested Capital | `total equity + total debt - cash` | Capital invested in operations |

### 5. Growth Metrics

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `rev_grw` | Revenue Growth | `revenue(t) - revenue(t-1)` | Absolute revenue change |
| `ast_grw` | Asset Growth | `total assets(t) - total assets(t-1)` | Absolute asset change |
| `inv_rt` | Investment Rate | `capex / total assets` | Rate of capital reinvestment |

### 6. Accrual Metrics

| Code | Concept | Formula | Description |
|------|---------|---------|-------------|
| `acc` | Accruals | `net income - CFO` | Difference between earnings and cash flow |
| `wc_acc` | Working Capital Accruals | `Delta(working capital) - depreciation and amortization` | Working capital changes |

## Usage

### Single Symbol Computation

```python
from src.derived.fundamental import DerivedFundamental

# Compute derived metrics for a single symbol
calc = DerivedFundamental(symbol='AAPL', year=2024)
derived_df = calc.run()

print(derived_df)
```

### Batch Computation

```python
from src.derived.fundamental import batch_compute_derived

# Compute for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
year = 2024

batch_compute_derived(symbols, year)
```

### Custom Paths

```python
from pathlib import Path
from src.derived.fundamental import DerivedFundamental

calc = DerivedFundamental(
    symbol='AAPL',
    year=2024,
    raw_data_dir=Path("custom/raw/path"),
    output_dir=Path("custom/output/path")
)
derived_df = calc.run()
```

### Load and Inspect

```python
from src.derived.fundamental import DerivedFundamental
import polars as pl

# Load raw data
calc = DerivedFundamental(symbol='AAPL', year=2024)
calc.load_raw_data()

# Compute specific metric groups
df = calc.raw_df.clone()
df = calc.compute_profitability_metrics(df)
df = calc.compute_cash_flow_metrics(df)

print(df.select(['timestamp', 'rev', 'grs_pft', 'grs_mgn', 'fcf']))
```

## Data Quality Considerations

### Lagged Values

Some metrics require previous period data (t-1):
- `roa`, `roe`: Require average assets/equity (t and t-1)
- `rev_grw`, `ast_grw`: Require previous period values

**First row values:** For metrics requiring lagged data, the first row will be `None` since there's no t-1 data.

### Missing Data Handling

The computation uses **safe arithmetic operations**:
- Division by zero returns `None`
- Operations with `None` values return `None`
- No automatic imputation or forward-filling

### Data Dependencies

Ensure raw fundamental data exists before computing derived metrics:
```bash
# Check if raw data exists
ls data/raw/fundamental/AAPL/fundamental.parquet

# If missing, collect raw data first
python -c "from src.collection.fundamental import Fundamental; ..."
```

## Implementation Details

### Formula Implementation

All formulas are hard-coded in `src/derived/fundamental.py` based on the definitions in `data/xbrl/fundamental.xlsx` (Priority = 3 rows). The formulas are implemented as methods in the `DerivedFundamental` class:

- `compute_profitability_metrics()`: Gross profit, margins, EBITDA
- `compute_cash_flow_metrics()`: FCF, FCF margin, capex ratio
- `compute_balance_sheet_constructs()`: Total debt, net debt, working capital
- `compute_returns()`: ROA, ROE, ROIC, effective tax rate
- `compute_growth_metrics()`: Revenue growth, asset growth
- `compute_accruals()`: Accrual metrics

### Logging

The implementation uses the centralized logger from `src/utils/logger.py`:

```python
from utils.logger import setup_logger

logger = setup_logger(
    name='derived.fundamental.AAPL',
    log_dir='data/logs/derived/fundamental',
    level=logging.INFO,
    console_output=True
)
```

**Log files:** `data/logs/derived/fundamental/logs_{YYYY-MM-DD}.log`

**Console output:** INFO level messages displayed during computation

### Computation Pipeline

The `DerivedFundamental` class computes metrics in dependency order:

1. **Load raw data** → from `data/raw/fundamental/{symbol}/`
2. **Compute profitability** → margins, EBITDA
3. **Compute balance sheet** → debt, working capital (needed for ROIC)
4. **Compute cash flow** → FCF, FCF margin
5. **Compute returns** → ROA, ROE, ROIC (depends on balance sheet)
6. **Compute growth** → revenue/asset growth
7. **Compute accruals** → earnings quality metrics
8. **Save** → to `data/derived/fundamental/{symbol}/{YYYY}/fundamental.parquet`

### Storage Format

Output files use **Parquet format** with the following structure:

```
Columns:
- timestamp: date (filing date)
- [all raw fundamental fields]: float64
- [all derived fields]: float64

Rows: One row per filing (typically quarterly)
```

## Examples

### Example 1: Profitability Analysis

```python
import polars as pl
from src.derived.fundamental import DerivedFundamental

calc = DerivedFundamental(symbol='AAPL', year=2024)
df = calc.run()

# Analyze profitability trends
profitability = df.select([
    'timestamp',
    'rev',
    'grs_mgn',
    'op_mgn',
    'net_mgn'
])

print(profitability)
```

### Example 2: Cash Flow Analysis

```python
# Compare FCF to net income
cash_quality = df.select([
    'timestamp',
    'net_inc',
    'cfo',
    'fcf',
    'fcf_mgn'
])

print(cash_quality)
```

### Example 3: Return Analysis

```python
# Analyze returns (ROA, ROE, ROIC)
returns = df.select([
    'timestamp',
    'roa',
    'roe',
    'roic',
    'etr'
])

print(returns)
```

### Example 4: Growth Analysis

```python
# Track growth metrics
growth = df.select([
    'timestamp',
    'rev',
    'rev_grw',
    'ta',
    'ast_grw'
]).with_columns([
    # Calculate growth rates
    (pl.col('rev_grw') / pl.col('rev').shift(1) * 100).alias('rev_grw_pct'),
    (pl.col('ast_grw') / pl.col('ta').shift(1) * 100).alias('ast_grw_pct')
])

print(growth)
```

## Testing

Run the test suite to verify computation:

```bash
# Run simple test
python test_derived_simple.py

# Run pytest tests (if pytest installed)
pytest tests/test_derived_fundamental.py -v
```

## Troubleshooting

### FileNotFoundError: Raw data not found

**Cause:** Raw fundamental data hasn't been collected yet.

**Solution:** Collect raw data first using `src/collection/fundamental.py`.

### Missing columns in derived output

**Cause:** Raw data is missing required fields.

**Solution:** Check that raw data contains all required base concepts (rev, cor, op_inc, etc.).

### All derived values are None

**Cause:** Raw data has None/null values for base concepts.

**Solution:** Verify data quality of raw fundamental data. Check SEC EDGAR data availability.

### Lagged metrics (ROA, ROE) are None

**Cause:** This is expected for the first row (no t-1 data).

**Solution:** Normal behavior. Lagged metrics require at least 2 data points.

## Future Enhancements

Potential improvements:

1. **Additional Metrics:**
   - Altman Z-Score
   - Piotroski F-Score
   - DuPont decomposition
   - Valuation ratios (P/E, P/B, etc.)

2. **Performance Optimization:**
   - Parallel processing for batch computation
   - Incremental updates (compute only new periods)
   - Caching intermediate results

3. **Data Quality:**
   - Outlier detection
   - Consistency checks
   - Cross-validation with industry benchmarks

4. **Integration:**
   - Automatic daily updates alongside raw data collection
   - API endpoints for derived metrics
   - Dashboard visualization

## References

- Raw fundamental collection: `src/collection/fundamental.py`
- Formula definitions: `data/xbrl/fundamental.xlsx`
- Field mappings: `configs/approved_mapping.yaml`
- Test suite: `tests/test_derived_fundamental.py`
