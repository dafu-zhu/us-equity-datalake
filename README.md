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
4. Run
   ```python
   # Upload
   uv run quantdl-storage

   # Update
   uv run quantdl-update
   ```

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

## Data Storage Format

### Daily Ticks
- **Path**: `data/raw/ticks/daily/{symbol}/{YYYY}/{MM}/ticks.parquet`
- **Format**: Parquet with OHLCV fields
- **Coverage**: 2009+ (CRSP)

### Minute Ticks
- **Path**: `data/raw/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet`
- **Format**: Parquet with OHLCV fields
- **Coverage**: 2016+ (Alpaca)

### Fundamentals
- **Path**: `data/raw/fundamental/{symbol}/fundamental.parquet`
- **Format**: Parquet with long table
- **Coverage**: 2009+ (SEC EDGAR)

### Derived
- **Path**:
   - `data/derived/features/fundamental/{symbol}/ttm.parquet`
   - `data/derived/features/fundamental/{symbol}/metrics.parquet`
- **Format**: Parquet with long table
- **Coverage**: 2009+ (SEC EDGAR)

## Development

### Running Tests

```bash
uv run pytest --cov=src/quantdl
```

## Data Quality & Known Limitations

### Coverage
- **Daily ticks**: ~99% coverage for complete market (2009+)
- **Fundamentals**: ~75% coverage for small-cap stocks (SEC filing requirements)
- **Minute data**: 2016+ only (Alpaca API limitation)

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
