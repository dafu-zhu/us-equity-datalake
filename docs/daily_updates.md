# Daily Updates System

## Overview

Daily updates are triggered via CLI:
```bash
uv run quantdl-update [--date YYYY-MM-DD]  # defaults to yesterday
uv run quantdl-update --no-ticks           # skip ticks, only fundamentals
uv run quantdl-update --lookback 14        # check 14 days for new filings
```

## Update Flow

```
run_daily_update(target_date)
    |
    v
1. SecurityMaster.update_from_sec()   <-- First step, always runs
    |
    v
2. Check if market was open on target_date
    |
    v
3A. If market open + update_ticks:
    |-- update_daily_ticks()
    |   Uses security_master.get_security_id(sym, date)
    |
    +-- update_minute_ticks()
        Uses security_master.get_security_id(sym, date)
    |
    v
3B. If update_fundamentals:
    |-- get_symbols_with_recent_filings()
    |   Checks EDGAR for recent 10-K/10-Q (7-day lookback)
    |
    +-- update_fundamental()
        Fetches & publishes for symbols with new filings
```

## SecurityMaster Update

### When it runs

SecurityMaster updates **at the start of every daily update**, before any ticks or fundamentals processing.

### Data source

SEC's official company tickers JSON:
- URL: `https://www.sec.gov/files/company_tickers.json`
- Content: Current snapshot of `[ticker, cik, title]`
- Limitation: No historical data (current only)

### What `update_from_sec()` does

**Operation 1: Extend end_dates**

For existing securities still in SEC list:
```
AAPL end_date: 2024-12-31 -> 2025-01-13
```

**Operation 2: Add new securities (IPOs)**

For tickers in SEC not in master_tb:
```python
{
    'security_id': max_sid + 1,
    'symbol': 'NEWIPO',
    'cik': '0001234567',
    'company': 'New IPO Corp',
    'start_date': today,
    'end_date': today,
    'permno': NULL  # No CRSP mapping yet
}
```

### Return stats

```python
{
    'extended': N,    # Rows with updated end_dates
    'added': N,       # New securities added
    'unchanged': N    # Rows unchanged
}
```

### Storage

Auto-exports to S3 after update:
- Path: `data/master/security_master.parquet`
- Metadata: `crsp_end_date`, `export_timestamp`, `version`, `row_count`

## SecurityMaster Initialization

Two paths depending on environment:

**Path A: WRDS available**
```python
crsp_ticks = CRSPDailyTicks(...)  # Connects to WRDS
security_master = crsp_ticks.security_master  # Built from CRSP
```

**Path B: S3-only (no WRDS, used in CI/CD)**
```python
security_master = SecurityMaster(
    s3_client=s3_client,
    bucket_name='us-equity-datalake'
)
# Loads from S3 cache (~1-2 sec vs ~30 sec WRDS rebuild)
```

## SecurityMaster Usage in Updates

| Component | Usage |
|-----------|-------|
| CIKResolver | symbol -> CIK for fundamental fetching |
| UniverseManager | Validates symbol existence at year |
| DataPublishers | symbol -> security_id for S3 paths |
| Daily Ticks | `get_security_id(sym, date)` for storage path |
| Minute Ticks | Same security_id resolution |

## Key Design Points

### Security ID assignment logic

- PERMNO changes -> new security_id
- PERMNO same but both symbol AND CIK change -> new security_id
- Otherwise -> same security_id (handles mergers, symbol changes)

### S3 caching

- SecurityMaster cached as Parquet with metadata
- Metadata includes `crsp_end_date` for staleness detection
- Load from S3: ~1-2 sec
- Rebuild from WRDS: ~30 sec

### Limitations

1. **SEC JSON is current-only**: No historical ticker mappings
2. **CRSP updates monthly**: New IPOs get `permno = NULL` until next CRSP refresh
3. **New securities lack CRSP data**: Daily ticks unavailable until CRSP catches up

## Filing Tracking

Separate from SecurityMaster update, `get_symbols_with_recent_filings()`:
- Checks EDGAR for recent 10-K/10-Q/8-K filings
- Default lookback: 7 days
- Returns symbols needing fundamental data update
- Triggered independently after SecurityMaster update

## File Locations

| File | Purpose |
|------|---------|
| `src/quantdl/master/security_master.py` | Core SecurityMaster class |
| `src/quantdl/update/app.py` | DailyUpdateApp with WRDS |
| `src/quantdl/update/app_no_wrds.py` | DailyUpdateApp without WRDS |
| `src/quantdl/update/cli.py` | CLI entry point |
