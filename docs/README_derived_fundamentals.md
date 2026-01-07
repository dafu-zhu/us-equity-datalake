# Derived Fundamentals Documentation Index

## Overview

This directory contains documentation for the derived fundamental metrics computation system.

## Documents

### 1. **derived_fundamental_integration.md** ⭐ RECOMMENDED
**For:** Storage pipeline integration (in-memory, no file I/O)

Complete guide to using `DataCollectors.compute_derived_fundamental()`:
- In-memory computation workflow
- Integration with `collect_fundamental_year()`
- All 24 derived metrics explained
- Usage examples and best practices
- Performance characteristics

**Use this when:**
- Building the storage pipeline
- Working with DataCollectors
- Need in-memory processing
- Want to avoid file I/O

### 2. **derived_fundamentals.md**
**For:** File-based computation (standalone use)

Documentation for the file-based approach using `src/derived/fundamental.py`:
- Complete formula reference
- File storage structure
- Batch processing
- Testing and validation

**Use this when:**
- Need standalone file-based computation
- Processing historical data in batches
- Working outside the storage pipeline

## Quick Start

### Storage Pipeline (Recommended)

```python
from storage.data_collectors import DataCollectors

# Initialize
collector = DataCollectors(crsp, alpaca, headers, logger)

# Collect raw data
raw_df = collector.collect_fundamental_year(cik='0001819994', year=2024, symbol='RKLB')

# Compute derived metrics (in-memory)
derived_df = collector.compute_derived_fundamental(raw_df, symbol='RKLB')

# Result: 32 raw + 24 derived = 56 columns
# All operations in RAM, ready for S3 upload
```

### File-Based (Legacy)

```python
from src.derived.fundamental import DerivedFundamental

# File-based: read → compute → write
calc = DerivedFundamental(symbol='AAPL', year=2024)
derived_df = calc.run()  # Reads/writes parquet files
```

## Derived Metrics Summary

### Categories (24 total)

1. **Profitability** (5): gross profit, margins, EBITDA
2. **Cash Flow** (3): free cash flow, FCF margin, capex ratio
3. **Balance Sheet** (3): total debt, net debt, working capital
4. **Returns** (8): ROA, ROE, ROIC, NOPAT, effective tax rate
5. **Growth** (3): revenue growth, asset growth, investment rate
6. **Accruals** (2): accruals, working capital accruals

### Formula Source

All formulas defined in: `data/xbrl/fundamental.xlsx` (Priority = 3 rows)

Hard-coded in:
- `src/storage/data_collectors.py:compute_derived_fundamental()` (storage pipeline)
- `src/derived/fundamental.py` (file-based)

## Implementation Comparison

| Feature | DataCollectors (Recommended) | File-Based (Legacy) |
|---------|------------------------------|---------------------|
| **File I/O** | None (in-memory) | Read + Write |
| **Speed** | Fast (no disk) | Slower (disk I/O) |
| **Use Case** | Storage pipeline | Standalone batches |
| **Integration** | Seamless with collectors | Requires setup |
| **Logging** | Uses existing logger | Creates own logger |
| **Location** | `src/storage/data_collectors.py` | `src/derived/fundamental.py` |

## Architecture

```
Storage Pipeline Flow:
┌─────────────────────────────────────────────────────────┐
│ 1. collect_fundamental_year()                           │
│    └─> Fetch from SEC EDGAR API                         │
│        Return: DataFrame with 32 raw columns            │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 2. compute_derived_fundamental()                        │
│    └─> Compute 24 derived metrics in-memory             │
│        Return: DataFrame with 56 columns (32+24)        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Upload to S3                                         │
│    └─> Store in data lake                               │
│        Path: s3://bucket/derived/fundamental/           │
└─────────────────────────────────────────────────────────┘
```

## Reference

### Code Locations

- **Storage Pipeline:** `src/storage/data_collectors.py:323-478`
- **File-Based:** `src/derived/fundamental.py`
- **Tests:** `tests/test_derived_fundamental.py`

### Data Sources

- **Raw Data:** SEC EDGAR JSON API
- **Formulas:** `data/xbrl/fundamental.xlsx` (Sheet: fields, Priority: 3)
- **Field Mappings:** `configs/approved_mapping.yaml`

### Related Documentation

- **Collection:** `src/collection/fundamental.py` (raw data extraction)
- **Models:** `src/collection/models.py` (data structures)
- **Storage:** `src/storage/` (S3 upload/download)

## FAQs

**Q: Which approach should I use?**
A: Use `DataCollectors.compute_derived_fundamental()` for the storage pipeline. Use file-based only for standalone batch processing.

**Q: Can I modify the formulas?**
A: Yes, edit the hard-coded formulas in `src/storage/data_collectors.py`. Formulas are based on `data/xbrl/fundamental.xlsx` but implemented directly in code.

**Q: Why are some values None?**
A: Lagged metrics (ROA, ROE, growth rates) require previous period data. First row will be None.

**Q: How do I add new derived metrics?**
A: Add the formula to the appropriate section in `compute_derived_fundamental()`. Follow the safe arithmetic pattern.

**Q: What if raw data is missing columns?**
A: The method handles missing columns gracefully - returns original DataFrame and logs error.

## Support

For questions or issues:
1. Check the documentation in this directory
2. Review code comments in `src/storage/data_collectors.py`
3. Run test suite: `pytest tests/test_derived_fundamental.py`
4. See example usage in integration test
