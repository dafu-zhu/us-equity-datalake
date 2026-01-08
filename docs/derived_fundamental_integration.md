# Derived Fundamental Integration with DataCollectors

## Overview

The `compute_derived_fundamental()` method in `DataCollectors` provides **in-memory** computation of 24 derived fundamental metrics from raw SEC EDGAR data.

**Key Features:**
- ✅ All operations in RAM (no file I/O)
- ✅ Seamless integration with `collect_fundamental_year()`
- ✅ 24 derived metrics: profitability, cash flow, returns, growth, accruals
- ✅ Safe arithmetic (handles None, division by zero)
- ✅ Hard-coded formulas based on `data/xbrl/fundamental.xlsx`

## Usage

### Basic Workflow

```python
from storage.data_collectors import DataCollectors

# Initialize DataCollectors
collector = DataCollectors(
    crsp_ticks=crsp_ticks,
    alpaca_ticks=alpaca_ticks,
    alpaca_headers=alpaca_headers,
    logger=logger
)

# Step 1: Collect raw fundamental data
raw_df = collector.collect_fundamental_year(
    cik='0001819994',
    year=2024,
    symbol='RKLB'
)

# Step 2: Compute derived metrics
derived_df = collector.compute_derived_fundamental(
    raw_df=raw_df,
    symbol='RKLB'
)

# Result: DataFrame with 32 raw + 24 derived = 56 columns
print(derived_df.columns)
```

### Combined Pipeline

```python
# Fetch and compute in one go
cik = '0001819994'
symbol = 'RKLB'
year = 2024

# Collect raw data
raw_df = collector.collect_fundamental_year(cik, year, symbol)

# Compute derived (handles empty DataFrame gracefully)
if len(raw_df) > 0:
    full_df = collector.compute_derived_fundamental(raw_df, symbol)
else:
    print(f"No data available for {symbol}")
```

### Batch Processing

```python
# Process multiple symbols
symbols_ciks = [
    ('AAPL', '0000320193'),
    ('MSFT', '0000789019'),
    ('GOOGL', '0001652044'),
]

results = {}
for symbol, cik in symbols_ciks:
    raw_df = collector.collect_fundamental_year(cik, 2024, symbol)
    if len(raw_df) > 0:
        results[symbol] = collector.compute_derived_fundamental(raw_df, symbol)
```

## Method Signature

```python
def compute_derived_fundamental(
    self,
    raw_df: pl.DataFrame,
    symbol: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute derived fundamental metrics from raw fundamental data.

    :param raw_df: Raw fundamental DataFrame (from collect_fundamental_year)
    :param symbol: Optional symbol for logging
    :return: DataFrame with both raw and derived columns
    """
```

## Input Requirements

### Raw DataFrame Schema

The input DataFrame must have these columns (output of `collect_fundamental_year`):

**Income Statement:**
- `rev` (revenue)
- `cor` (cost of revenue)
- `op_inc` (operating income)
- `net_inc` (net income)
- `ibt` (income before tax)
- `inc_tax_exp` (income tax expense)
- `dna` (depreciation and amortization)

**Balance Sheet:**
- `cce` (cash and cash equivalents)
- `ca` (current assets)
- `ta` (total assets)
- `std` (short term debt)
- `cl` (current liabilities)
- `ltd` (long term debt)
- `te` (total equity)

**Cash Flow:**
- `cfo` (cash flow from operations)
- `capex` (capital expenditures)

**Other:**
- `timestamp` (filing date)

## Output Schema

The output DataFrame contains **all raw columns** plus **24 derived columns**:

### Profitability Metrics (5)
| Code | Name | Formula |
|------|------|---------|
| `grs_pft` | Gross Profit | `revenue - cost of revenue` |
| `grs_mgn` | Gross Margin | `gross profit / revenue` |
| `op_mgn` | Operating Margin | `operating income / revenue` |
| `net_mgn` | Net Margin | `net income / revenue` |
| `ebitda` | EBITDA | `operating income + depreciation and amortization` |

### Cash Flow Metrics (3)
| Code | Name | Formula |
|------|------|---------|
| `fcf` | Free Cash Flow | `CFO - CapEx` |
| `fcf_mgn` | FCF Margin | `free cash flow / revenue` |
| `capex_ratio` | CapEx Ratio | `capex / total assets` |

### Balance Sheet Constructs (3)
| Code | Name | Formula |
|------|------|---------|
| `ttl_dbt` | Total Debt | `short term debt + long term debt` |
| `net_dbt` | Net Debt | `total debt - cash` |
| `wc` | Working Capital | `current assets - current liabilities` |

### Return Metrics (8)
| Code | Name | Formula |
|------|------|---------|
| `avg_ast` | Average Assets | `(total assets(t) + total assets(t-1)) / 2` |
| `avg_eqt` | Average Equity | `(total equity(t) + total equity(t-1)) / 2` |
| `etr` | Effective Tax Rate | `income tax expense / income before tax` |
| `roa` | Return on Assets | `net income / avg assets` |
| `roe` | Return on Equity | `net income / avg equity` |
| `nopat` | NOPAT | `operating income × (1 − effective tax rate)` |
| `inv_cap` | Invested Capital | `total equity + total debt - cash` |
| `roic` | Return on Invested Capital | `NOPAT / invested capital` |

### Growth Metrics (3)
| Code | Name | Formula |
|------|------|---------|
| `rev_grw` | Revenue Growth | `revenue(t) - revenue(t-1)` |
| `ast_grw` | Asset Growth | `total assets(t) - total assets(t-1)` |
| `inv_rt` | Investment Rate | `capex / total assets` |

### Accrual Metrics (2)
| Code | Name | Formula |
|------|------|---------|
| `acc` | Accruals | `net income - CFO` |
| `wc_acc` | WC Accruals | `Delta(working capital) - depreciation and amortization` |

## Lagged Data Handling

Some metrics require previous period data (t-1):

**First Row Behavior:**
- Lagged metrics return `None` for the first row (no t-1 data)
- Examples: `roa`, `roe`, `rev_grw`, `ast_grw`, `avg_ast`, `avg_eqt`

**Subsequent Rows:**
- Lagged metrics computed using `.shift(1)`
- Polars automatically handles time series operations

## Error Handling

### Empty Input
```python
raw_df = pl.DataFrame()  # Empty
derived_df = collector.compute_derived_fundamental(raw_df, 'AAPL')
# Returns: pl.DataFrame() (empty)
```

### Missing Columns
If required columns are missing:
- Method returns original DataFrame
- Error logged via `self.logger.error()`
- No exception raised (graceful degradation)

### Division by Zero
Safe arithmetic functions handle edge cases:
```python
# Example: revenue is 0
# net_mgn = net_inc / revenue → returns None (not error)
```

## Logging

The method uses the existing logger from `DataCollectors`:

```python
# Log messages during computation
RKLB: Computing profitability metrics
RKLB: Computing balance sheet constructs
RKLB: Computing cash flow metrics
RKLB: Computing return metrics
RKLB: Computing growth metrics
RKLB: Computing accruals
RKLB: Derived computation complete: 4 rows, 56 columns (24 derived)
```

**Log Level:** `DEBUG` (quiet by default)

## Performance Characteristics

**Time Complexity:** O(n) where n = number of rows (typically 4-5 per year)

**Memory:** In-memory only, no disk I/O

**Typical Performance:**
- 4 quarters of data: <10ms
- No file reads or writes
- Vectorized Polars operations

## Integration Example

### Full Storage Pipeline

```python
from storage.data_collectors import DataCollectors
from storage.s3_uploader import S3Uploader

# 1. Initialize components
collector = DataCollectors(...)
uploader = S3Uploader(...)

# 2. Collect raw fundamental data
cik = '0001819994'
symbol = 'RKLB'
year = 2024

raw_df = collector.collect_fundamental_year(cik, year, symbol)

# 3. Compute derived metrics
full_df = collector.compute_derived_fundamental(raw_df, symbol)

# 4. Upload to S3
# Option A: Store raw only
uploader.upload_fundamental_year(raw_df, symbol, year, path='raw/fundamental')

# Option B: Store derived only (includes raw columns)
uploader.upload_fundamental_year(full_df, symbol, year, path='derived/fundamental')

# Option C: Store both separately
uploader.upload_fundamental_year(raw_df, symbol, year, path='raw/fundamental')
uploader.upload_fundamental_year(full_df, symbol, year, path='derived/fundamental')
```

## Comparison with File-Based Approach

### Old Approach (src/derived/fundamental.py)
```python
# File-based: read from disk, compute, write to disk
calc = DerivedFundamental(symbol='AAPL', year=2024)
calc.load_raw_data()  # reads data/raw/fundamental/AAPL/fundamental.parquet
derived_df = calc.compute_all_derived()
calc.save(derived_df)  # writes data/derived/fundamental/AAPL/2024/fundamental.parquet
```

### New Approach (DataCollectors integration)
```python
# In-memory: fetch → compute → return
raw_df = collector.collect_fundamental_year('0000320193', 2024, 'AAPL')
derived_df = collector.compute_derived_fundamental(raw_df, 'AAPL')
# No file I/O, ready for storage pipeline
```

**Benefits:**
- ✅ No intermediate file storage
- ✅ Faster (no disk I/O)
- ✅ More flexible (can choose to store or not)
- ✅ Integrates with existing DataCollectors pattern
- ✅ Consistent with other collector methods

## Validation

### Formula Verification

```python
# Get latest quarter
latest_idx = derived_df.shape[0] - 1

# Verify gross profit
rev = derived_df['rev'][latest_idx]
cor = derived_df['cor'][latest_idx]
grs_pft = derived_df['grs_pft'][latest_idx]

assert abs((rev - cor) - grs_pft) < 0.01, "Gross profit formula error"

# Verify gross margin
grs_mgn = derived_df['grs_mgn'][latest_idx]
assert abs((grs_pft / rev) - grs_mgn) < 0.0001, "Gross margin formula error"
```

### Data Quality Checks

```python
# Check for unrealistic values
assert (derived_df['grs_mgn'] <= 1.0).all(), "Gross margin > 100%"
assert (derived_df['grs_mgn'] >= -1.0).all(), "Gross margin < -100%"

# Check logical consistency
# Assets = Liabilities + Equity (accounting identity)
balance_check = (
    (derived_df['ta'] - (derived_df['tl'] + derived_df['te'])).abs() < 0.01
).all()
```

## Troubleshooting

### Issue: All derived values are None

**Cause:** Raw data columns are None/null

**Solution:** Check data quality from `collect_fundamental_year()`

### Issue: Lagged metrics always None

**Cause:** Only 1 row of data (need at least 2 for t-1)

**Solution:** Normal behavior, collect more periods

### Issue: ROIC is None but other metrics work

**Cause:** Missing intermediate columns (e.g., `ttl_dbt`, `etr`)

**Solution:** Check that raw data has required fields: `std`, `ltd`, `inc_tax_exp`, `ibt`

## References

- **Implementation:** `src/storage/data_collectors.py:323-478`
- **Raw Data Collection:** `src/storage/data_collectors.py:235-321`
- **Formula Definitions:** `data/xbrl/fundamental.xlsx` (Priority = 3 rows)
- **Field Mappings:** `configs/approved_mapping.yaml`
- **Legacy File-Based Approach:** `src/derived/fundamental.py` (deprecated for storage pipeline)
