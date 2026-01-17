# US Equity Data Lake

[![Tests](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml/badge.svg)](https://github.com/dafu-zhu/us-equity-datalake/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/dafu-zhu/us-equity-datalake/branch/main/graph/badge.svg)](https://codecov.io/gh/dafu-zhu/us-equity-datalake)

A self-hosted, automated market data infrastructure for US equities using official/authoritative sources, with daily updates and programmatic access via Python API.

## Overview

This project provides comprehensive data collection, storage, and query capabilities for US equity markets:

- **Daily tick data (OHLCV)** from CRSP via WRDS (2009+)
- **Minute-level tick data** from Alpaca Market Data API (2016+)
- **Fundamental data** from SEC EDGAR JSON API (2009+)
- **Derived financial indicators** (ROA, ROE, etc.)

All data is stored in a flat-file structure on AWS S3, optimized for fast querying and minimal storage costs.

## Features

- **Official Data Sources**: All data from authoritative sources (SEC, CRSP, Alpaca)
- **Automated Updates**: Daily scheduled updates with error handling and retry logic
- **Flat File Storage**: Organized by symbol and time period for efficient querying
- **Security Master**: Track stocks across symbol changes, mergers, and corporate actions
- **No Survivorship Bias**: Delisted and inactive stocks are retained for unbiased backtesting
- **Alpha Research API**: Structured for quantitative research via [quantdl-api](https://github.com/dafu-zhu/quantdl-api)

## Installation

### Prerequisites

- Python 3.12+
- AWS account with S3 access
- WRDS account (for CRSP data)
- Alpaca account (for minute-level data)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dafu-zhu/us-equity-datalake.git
   cd us-equity-datalake
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Configure environment variables (create `.env` file):
   ```bash
   # WRDS Credentials
   WRDS_USERNAME=your_username
   WRDS_PASSWORD=your_password

   # Alpaca API Credentials
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_secret_key

   # AWS Credentials (for S3 storage)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key

   # SEC EDGAR API (required User-Agent)
   SEC_USER_AGENT=your_name@example.com
   ```

## Commands

### Upload (Initial Backfill)

```bash
uv run quantdl-storage [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--start-year` | 2009 | Start year for data collection |
| `--end-year` | 2025 | End year for data collection |
| `--overwrite` | false | Overwrite existing data |
| `--resume` | false | Resume from last checkpoint (minute ticks) |
| `--run-all` | false | Run all upload workflows |
| `--run-fundamental` | false | Upload fundamental data only |
| `--run-derived-fundamental` | false | Upload derived metrics only |
| `--run-ttm-fundamental` | false | Upload TTM fundamentals only |
| `--run-daily-ticks` | false | Upload daily ticks only |
| `--run-minute-ticks` | false | Upload minute ticks only |
| `--run-top-3000` | false | Upload top 3000 stocks only |
| `--alpaca-start-year` | 2025 | Alpaca data start year |
| `--minute-start-year` | 2017 | Minute ticks start year |
| `--daily-chunk-size` | 200 | Batch size for daily ticks |
| `--daily-sleep-time` | 0.2 | Sleep between daily batches (seconds) |
| `--max-workers` | 50 | Max parallel workers |
| `--minute-workers` | 50 | Workers for minute ticks |
| `--minute-chunk-size` | 500 | Batch size for minute ticks |
| `--minute-sleep-time` | 0.0 | Sleep between minute batches (seconds) |

**Examples:**
```bash
# Full backfill
uv run quantdl-storage --run-all --start-year 2009 --end-year 2025

# Upload specific data types
uv run quantdl-storage --run-fundamental
uv run quantdl-storage --run-daily-ticks
uv run quantdl-storage --run-minute-ticks
```

### Update (Daily Incremental)

```bash
uv run quantdl-update [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--date` | yesterday | Target date (YYYY-MM-DD) |
| `--backfill-from` | - | Backfill from date to --date (max 30 days) |
| `--no-ticks` | false | Skip all ticks (daily + minute) |
| `--no-daily-ticks` | false | Skip daily ticks only |
| `--no-minute-ticks` | false | Skip minute ticks only |
| `--no-fundamental` | false | Skip raw fundamental update |
| `--no-ttm` | false | Skip TTM fundamental update |
| `--no-derived` | false | Skip derived metrics update |
| `--lookback` | 7 | Days to look back for EDGAR filings |
| `--no-wrds` | false | WRDS-free mode (Nasdaq universe + SEC CIK mapping) |

**Examples:**
```bash
# Update yesterday's data
uv run quantdl-update

# Update specific date
uv run quantdl-update --date 2025-01-10

# Backfill a date range
uv run quantdl-update --backfill-from 2025-01-01 --date 2025-01-10

# Skip ticks, only fundamentals
uv run quantdl-update --no-ticks
```

### Consolidate (Year-End)

```bash
uv run quantdl-consolidate --year YYYY
```

Consolidates monthly Parquet files into `history.parquet` for completed years. Run annually on Jan 1.

## Data Storage Format

| Data Type | Path | Format | Coverage |
|-----------|------|--------|----------|
| Daily Ticks | `data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet` | Parquet (OHLCV) | 2009+ (CRSP) |
| Minute Ticks | `data/raw/ticks/minute/{security_id}/{YYYY}/{MM}/{DD}/ticks.parquet` | Parquet (OHLCV) | 2016+ (Alpaca) |
| Fundamentals | `data/raw/fundamental/{cik}/fundamental.parquet` | Parquet (long table) | 2009+ (SEC EDGAR) |
| TTM Derived | `data/derived/features/fundamental/{cik}/ttm.parquet` | Parquet (long table) | 2009+ |
| Metrics Derived | `data/derived/features/fundamental/{cik}/metrics.parquet` | Parquet (long table) | 2009+ |

## Project Structure

```
us-equity-datalake/
├── src/quantdl/              # Main package
│   ├── collection/           # Data collectors (CRSP, Alpaca, SEC)
│   ├── storage/              # Upload and validation logic
│   ├── master/               # Security master (symbol tracking)
│   ├── update/               # Daily data update logic
│   ├── universe/             # Universe and stock filtering
│   ├── derived/              # Technical indicators
│   └── utils/                # Logging, mapping, rate limiting
├── scripts/                  # One-off scripts and utilities
├── docs/                     # Additional documentation
└── tests/                    # Unit and integration tests
```

## GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **Tests** | Push, PR | Runs pytest with coverage, uploads to Codecov |
| **Daily Update** | Daily at 9:00 UTC (4am ET) | Incremental data update (WRDS-free mode) |
| **Manual Daily Update** | Manual dispatch | On-demand update with backfill support |
| **Year Consolidation** | Jan 1 at 10:00 UTC | Consolidates previous year's monthly files |

### Required Secrets

| Secret | Description |
|--------|-------------|
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 |
| `SEC_USER_AGENT` | User-Agent for SEC EDGAR API |
| `MAIL_SERVER` | SMTP server for notifications |
| `MAIL_PORT` | SMTP port |
| `MAIL_USERNAME` | SMTP username |
| `MAIL_PASSWORD` | SMTP password |
| `MAIL_TO` | Notification recipient |
| `MAIL_FROM` | Notification sender |

## Development

### Running Tests

```bash
uv run pytest --cov=src/quantdl
```

## Data Quality and Known Limitations

| Data Type | Coverage | Notes |
|-----------|----------|-------|
| Daily ticks | ~99% | Complete market coverage (2009+) |
| Fundamentals | ~75% | Small-cap limited by SEC filing requirements |
| Minute data | 2016+ | Alpaca API limitation |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **WRDS** for CRSP database access
- **SEC** for EDGAR API and official filings
- **Alpaca** for minute-level market data
- **Nasdaq** for reference data

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [documentation](docs/)
