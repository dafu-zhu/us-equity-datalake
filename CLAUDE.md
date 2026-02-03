# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

US Equity Data Lake: Self-hosted, automated market data infrastructure for US equities using official/authoritative sources. Data stored in flat-file structure on AWS S3.

**Data Coverage:**
- Daily ticks (OHLCV): CRSP via WRDS (2009+)
- Minute ticks: Alpaca API (2016+)
- Fundamentals: SEC EDGAR JSON API (2009+)
- Derived metrics: ROA, ROE, TTM calculations
- Sentiment: NLP analysis of MD&A sections (FinBERT, 2009+)

## Commands

### Environment Setup
```bash
uv sync

# Required .env variables:
# WRDS_USERNAME, WRDS_PASSWORD (for CRSP)
# ALPACA_API_KEY, ALPACA_API_SECRET (for minute ticks)
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for S3)
# SEC_USER_AGENT (for EDGAR API, e.g., your_name@example.com)
```

### Testing
```bash
uv run pytest --cov=src/quantdl     # All tests with coverage
uv run pytest -m unit               # Unit tests only (fast)
uv run pytest -m integration        # Integration tests only
uv run pytest tests/unit/collection/test_fundamental.py  # Single file
uv run pytest -n auto               # Parallel execution
```

### Data Operations
```bash
# Initial upload (full backfill)
uv run quantdl-storage --run-all --start-year 2009 --end-year 2025

# Upload specific data types
uv run quantdl-storage --run-fundamental
uv run quantdl-storage --run-daily-ticks
uv run quantdl-storage --run-minute-ticks
uv run quantdl-storage --run-derived-fundamental
uv run quantdl-storage --run-sentiment            # NLP sentiment from SEC filings

# Daily update (incremental)
uv run quantdl-update --date 2025-01-10
uv run quantdl-update                    # defaults to yesterday
uv run quantdl-update --no-ticks         # fundamentals only
uv run quantdl-update --no-wrds          # WRDS-free mode (GitHub Actions)

# Year consolidation (run on Jan 1)
uv run quantdl-consolidate --year 2025   # merge monthly → history.parquet
uv run quantdl-consolidate --year 2025 --force  # overwrite if exists

# Security master export
uv run quantdl-export-security-master --export
uv run quantdl-export-security-master --export --force-rebuild  # skip S3 cache
```

## Architecture

### Core Components

**1. Collection Layer** (`src/quantdl/collection/`)
- `crsp_ticks.py`: Fetches daily OHLCV from CRSP via WRDS
- `alpaca_ticks.py`: Fetches minute-level data from Alpaca API
- `fundamental.py`: Fetches SEC EDGAR XBRL data (JSON API)
- `models.py`: Data models (TickField, FndDataPoint, DataSource)

**2. Storage Layer** (`src/quantdl/storage/`)
- `data_collectors.py`: Collects data from sources, handles rate limiting
- `data_publishers.py`: Publishes collected data to S3 in Parquet format
- `s3_client.py`: S3 client wrapper with retry logic
- `validation.py`: Validates uploaded data completeness
- `cik_resolver.py`: Maps tickers to SEC CIK codes
- `rate_limiter.py`: Rate limiting for API calls

**3. Universe Management** (`src/quantdl/universe/`)
- `current.py`: Fetches current stock universe from Nasdaq Trader
- `historical.py`: Fetches historical universe from CRSP
- `manager.py`: Manages universe state, handles symbol changes

**4. Security Master** (`src/quantdl/master/`)
- `security_master.py`: Tracks stocks across symbol changes, mergers, delistings
- Contains `SymbolNormalizer` for deterministic ticker format conversion

**5. Derived Data** (`src/quantdl/derived/`)
- `ttm.py`: Computes trailing-twelve-month (TTM) fundamentals
- `metrics.py`: Computes derived metrics (ROA, ROE, leverage ratios, etc.)
- `sentiment.py`: Computes sentiment metrics from MD&A text using FinBERT

**5a. Models** (`src/quantdl/models/`)
- `base.py`: SentimentModel ABC, SentimentResult dataclass
- `finbert.py`: FinBERT implementation (ProsusAI/finbert) with CUDA support

**6. Update Apps** (`src/quantdl/update/`)
- `app.py`: `DailyUpdateApp` orchestrates daily incremental updates
- `app_no_wrds.py`: WRDS-free version using Nasdaq FTP + SEC API (for GitHub Actions)
- `cli.py`: CLI entry point for daily updates

**7. Upload Apps** (`src/quantdl/storage/`)
- `app.py`: `UploadApp` orchestrates full backfill uploads
- `cli.py`: CLI entry point for initial uploads

**8. Consolidate** (`src/quantdl/consolidate/`)
- `cli.py`: Merges monthly Parquet files into history.parquet for completed years

### Data Flow

```
Data Sources → Collection → Validation → S3 Storage
                    ↓
            Derived Metrics
                    ↓
                S3 Storage
```

**Initial Upload:**
1. `UploadApp` fetches universe from Nasdaq/CRSP
2. `DataCollectors` fetch data from sources (CRSP, Alpaca, SEC)
3. `DataPublishers` write Parquet files to S3
4. `Validator` checks completeness

**Daily Update:**
1. `DailyUpdateApp` checks if market was open yesterday
2. For ticks: Fetch yesterday's data, append to current year monthly Parquet files
3. For fundamentals: Check EDGAR filings in last N days, update if new 10-K/10-Q
4. Upload updated files to S3
5. Year consolidation (Jan 1): Merge previous year monthly files into history.parquet

### Storage Paths

```
data/raw/
├── ticks/
│   ├── daily/{security_id}/
│   │   ├── history.parquet                    # All completed years consolidated
│   │   └── {current_year}/{MM}/ticks.parquet  # Current year (monthly partitions)
│   └── minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet
├── fundamental/{cik}/fundamental.parquet
data/derived/
├── features/fundamental/{cik}/
│   ├── ttm.parquet
│   └── metrics.parquet
└── features/sentiment/{cik}/sentiment.parquet
```

**Storage Strategy:**
- Daily ticks: Keyed by security_id; current year monthly, history consolidated
- Minute ticks: Partitioned by symbol, year, month, day (unchanged)
- Fundamentals: Keyed by CIK (one file per legal entity)
- Derived: Keyed by CIK (one file per metric type)

### Key Design Decisions

**1. Security ID-Based Storage (Daily Ticks)**
- Daily ticks stored under security_id (not symbol) to prevent collisions
- SecurityMaster resolves symbol+date → security_id (tracks business continuity)
- Current year uses monthly partitions (optimizes daily updates, ~5 KB files)
- Completed years consolidated in history.parquet (optimizes multi-year queries, ~18 MB)
- TicksClient provides symbol-based API with transparent security_id resolution
- Session-based caching for symbol lookups (cleared on client reinitialization)
- Year consolidation runs on Jan 1: merges 12 monthly → append to history

**2. CIK-Based Storage (Fundamentals)**
- Fundamentals stored under SEC CIK codes, not tickers
- `CIKResolver` maps tickers to CIKs using SEC company tickers JSON
- Prevents data loss during ticker changes/mergers

**3. Symbol Normalization**
- `SymbolNormalizer` converts CRSP format (BRKB) to Nasdaq format (BRK.B)
- Uses `SecurityMaster` to verify same security_id before conversion
- Keeps delisted stocks in original format

**4. Parallel Processing**
- Daily ticks: Batch processing with rate limiting (200 symbols/batch)
- Minute ticks: High parallelization (50 workers, 100+ concurrent S3 reads)
- Fundamentals: Sequential with retry logic (EDGAR API limits)

**5. Update Strategy**
- Fundamentals: Check EDGAR for new 10-K/10-Q filings (lookback 7 days)
- Daily ticks: Update only if trading day, append to current year monthly files
- Read-modify-write pattern for monthly Parquet files (~5 KB, fast updates)
- Year consolidation: Jan 1 consolidates previous year into history.parquet

**6. WRDS-Free Mode**
- `--no-wrds` flag enables GitHub Actions deployment without WRDS access
- Uses Nasdaq FTP for current universe + SEC API for CIK mapping
- `SimpleCIKResolver` in `app_no_wrds.py` handles ticker→CIK resolution
- Suitable for recent data (2025+) where current SEC mapping is accurate

## Testing

### Test Structure
```
tests/
├── unit/          # Fast, isolated tests (mocked external calls)
├── integration/   # Tests with real S3/WRDS/SEC (slower)
└── conftest.py    # Shared fixtures, auto-marks unit/integration
```

### Test Markers
- `@pytest.mark.unit`: Fast unit tests (auto-applied to tests/unit/**)
- `@pytest.mark.integration`: Integration tests (auto-applied to tests/integration/**)
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.external`: Requires external API access

### Common Fixtures (conftest.py)
- `sample_ticker`, `sample_tickers`: Test tickers
- `sample_date`, `sample_date_range`: Test dates
- `sample_year`: Test year (2024)
- `sample_cik`: Apple's CIK (0000320193)

## Critical Files & Configs

**configs/approved_mapping.yaml**
- Maps standardized field names to XBRL tags (SEC EDGAR)
- Used by `fundamental.py` to extract XBRL concepts
- Multiple candidate tags per concept (handles deprecated tags)

**DURATION_CONCEPTS (fundamental.py)**
- Duration concepts (income statement): rev, net_inc, cfo, etc.
- Instant concepts (balance sheet): assets, liab, equity, etc.
- Determines quarterly filtering logic

## Common Patterns

### Adding New Data Collector
1. Create collector class in `collection/` inheriting `DataCollector`
2. Add collector initialization in `DataCollectors` class
3. Add publisher method in `DataPublishers` class
4. Add CLI flags to `storage/cli.py` and `storage/app.py`

### Adding New Derived Metric
1. Add metric function to `derived/metrics.py` or `derived/ttm.py`
2. Update `compute_derived()` or `compute_ttm_long()` to include metric
3. Add tests to `tests/unit/derived/test_metrics.py`

### Modifying Storage Paths
1. Update path constants in `UploadConfig` (storage/config_loader.py)
2. Update corresponding publisher methods in `data_publishers.py`
3. Update validation logic in `validation.py`

## Known Limitations

1. **Fundamentals:** ~75% small-cap coverage (SEC filing requirements)
2. **Minute data:** 2016+ only (Alpaca limitation)
3. **Historical index constituents:** Not available (survivorship bias)
4. **Fundamental lag:** 45-90 days (SEC filing deadlines)
5. **CRSP data:** Updated monthly by WRDS (latest: 2024-12-31)

## Edge Cases

### Symbol Changes
- SecurityMaster tracks permno (CRSP ID) across ticker changes
- SymbolNormalizer prevents false matches (e.g., delisted ABCD ≠ ABC.D)
- Historical data kept under original ticker for delisted stocks

### Corporate Actions
- Fundamentals handle share splits via XBRL context (shares outstanding)
- Daily ticks include both raw and adjusted close prices

### Missing Data
- Trading halts: Skip (don't interpolate)
- Data source unavailable: Retry with exponential backoff
- Fundamental filing delays: Expected, tracked separately

## Dependencies

Key libraries:
- **polars**: DataFrame processing (faster than pandas for large datasets)
- **boto3**: AWS S3 client
- **wrds**: CRSP data access
- **requests**: HTTP client for Alpaca, SEC EDGAR
- **arelle**: XBRL processing for SEC filings
- **pytest**: Testing framework
- **pyarrow**: Parquet I/O

## Performance Notes

- Daily ticks: ~1.2 GB for 5000 symbols × 15 years
- Minute ticks: ~283 GB for 5000 symbols × 9 years
- Fundamentals: ~7.5 GB for 5000 symbols × 15 years
- S3 costs: ~$7/month storage + API request costs
- Use threading for I/O-bound operations (S3 reads/writes)
- Rate limiting: CRSP via WRDS (throttled by DB), SEC EDGAR (10 req/sec limit)
