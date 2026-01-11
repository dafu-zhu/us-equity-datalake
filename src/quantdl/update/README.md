# Daily Update Module

## Overview

The Daily Update module provides automated daily data updates for the US Equity Data Lake. It intelligently updates:

1. **Daily Ticks** - OHLCV data from Alpaca
2. **Minute Ticks** - Intraday minute-level data from Alpaca
3. **Fundamental Data** - Financial statements from SEC EDGAR

## Key Features

### Smart Update Logic

- **Market Calendar Awareness**: Only updates ticks data if the market was open
- **EDGAR Filing Detection**: Only updates fundamental data if new filings are detected
- **Incremental Updates**: Efficiently updates existing year files instead of reprocessing everything
- **Concurrent Processing**: Uses threading for faster uploads
- **Error Handling**: Graceful error handling with detailed logging

### Data Updated

#### Daily Ticks
- **Storage**: `data/raw/ticks/daily/{symbol}/{YYYY}/{MM}/ticks.parquet` (monthly partitions)
- **Update Strategy**: Fetches yesterday's data and merges with current month file only
- **Optimization**: Only downloads/uploads current month (~5 KB) instead of entire year (~64 KB)
- **Savings**: 92% reduction in S3 data transfer, 13x faster updates
- **Source**: Alpaca Markets API

#### Minute Ticks
- **Storage**: `data/raw/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet`
- **Update Strategy**: Creates new daily parquet files
- **Source**: Alpaca Markets API

#### Fundamental Data
- **Raw**: `data/raw/fundamental/{symbol}/fundamental.parquet`
- **TTM**: `data/derived/features/fundamental/{symbol}/ttm.parquet`
- **Metrics**: `data/derived/features/fundamental/{symbol}/metrics.parquet`
- **Update Strategy**: Checks EDGAR for recent filings, only updates if new data available
- **Source**: SEC EDGAR API

## Usage

### Python API

```python
from quantdl.update.app import DailyUpdateApp
import datetime as dt

# Initialize the app
app = DailyUpdateApp()

# Run full daily update (updates yesterday's data)
app.run_daily_update()

# Update specific date
target_date = dt.date(2025, 1, 9)
app.run_daily_update(target_date=target_date)

# Update only ticks (skip fundamentals)
app.run_daily_update(update_fundamentals=False)

# Update only fundamentals (skip ticks)
app.run_daily_update(update_ticks=False)

# Customize EDGAR lookback period
app.run_daily_update(fundamental_lookback_days=14)
```

### Command Line Interface

```bash
# Update yesterday's data (default)
python -m quantdl.update.app

# Update specific date
python -m quantdl.update.app --date 2025-01-09

# Skip ticks update
python -m quantdl.update.app --no-ticks

# Skip fundamental update
python -m quantdl.update.app --no-fundamentals

# Customize EDGAR lookback period (default: 7 days)
python -m quantdl.update.app --lookback 14

# Combine options
python -m quantdl.update.app --date 2025-01-08 --no-ticks --lookback 14
```

## Update Workflow

The daily update follows this workflow:

```
1. Initialize
   ├── Load configuration
   ├── Setup S3 client
   ├── Initialize data collectors
   └── Load current universe

2. Check Market Status
   └── Query trading calendar for target date

3. Update Ticks (if market was open)
   ├── Update Daily Ticks (Monthly Partitions)
   │   ├── Download current month file from S3 (only ~5 KB)
   │   ├── Fetch yesterday's data from Alpaca
   │   ├── Merge new data with existing month
   │   └── Upload updated month file to S3 (saves 92% transfer cost!)
   │
   └── Update Minute Ticks
       ├── Fetch minute data for all symbols
       ├── Parse into daily DataFrames
       └── Upload new daily parquet files to S3

4. Update Fundamentals
   ├── Check EDGAR for Recent Filings
   │   ├── Query SEC submissions endpoint
   │   └── Filter for 10-K, 10-Q, 8-K forms
   │
   ├── Identify Symbols with New Filings
   │
   └── Update Fundamental Data
       ├── Fetch raw fundamental data
       ├── Compute TTM features
       ├── Compute derived metrics
       └── Upload all three to S3

5. Complete
   └── Log summary statistics
```

## Configuration

The update app uses the same configuration as the main upload app:

- **Storage Config**: `configs/storage.yaml`
- **Environment Variables**: `.env` file
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `ALPACA_API_KEY`
  - `ALPACA_API_SECRET`
  - `SEC_USER_AGENT`

## Scheduling

### Cron Job (Linux/Mac)

```bash
# Run daily at 6 PM ET (after market close)
0 18 * * 1-5 cd /path/to/us-equity-datalake && /usr/bin/python3 -m quantdl.update.app
```

### Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 6:00 PM
4. Set action: Run program
   - Program: `python`
   - Arguments: `-m quantdl.update.app`
   - Start in: `/path/to/us-equity-datalake`

### AWS Lambda (Serverless)

Deploy as Lambda function with EventBridge trigger:

```yaml
# serverless.yml
functions:
  dailyUpdate:
    handler: src/quantdl/update/lambda_handler.handler
    timeout: 900  # 15 minutes
    events:
      - schedule: cron(0 22 ? * MON-FRI *)  # 6 PM ET = 10 PM UTC
```

## Error Handling

The update app handles errors gracefully:

- **Network Errors**: Retries with exponential backoff (via rate limiters)
- **Missing Data**: Logs warning and continues with next symbol
- **API Rate Limits**: Respects Alpaca (200/min) and SEC (10/sec) limits
- **S3 Errors**: Logs error and continues with next upload

All errors are logged to:
- Console output (INFO level)
- Log files in `data/logs/update/`

## Monitoring

Check update status:

```bash
# View recent logs
tail -f data/logs/update/*.log

# Check S3 for updated files
aws s3 ls s3://us-equity-datalake/data/raw/ticks/daily/AAPL/2025/

# Verify update timestamps
aws s3api head-object --bucket us-equity-datalake \
  --key data/raw/ticks/daily/AAPL/2025/ticks.parquet \
  | jq '.LastModified'
```

## Performance

Typical daily update times:

- **Daily Ticks**: ~10-15 minutes for 5000 symbols
- **Minute Ticks**: ~20-30 minutes for 5000 symbols
- **Fundamentals**: ~5-10 minutes for 50-100 symbols with new filings

Total: **~30-60 minutes** per day

## Troubleshooting

### Market was closed but I want to force update

```python
app.update_daily_ticks(target_date, symbols)
app.update_minute_ticks(target_date, symbols)
```

### Update failed for specific symbol

```python
# Retry single symbol
app.update_daily_ticks(target_date, symbols=['AAPL'])
```

### EDGAR rate limit errors

Increase lookback days to spread requests:

```bash
python -m quantdl.update.app --lookback 30
```

### S3 upload timeouts

Reduce concurrency in code:

```python
app.update_minute_ticks(target_date, symbols, max_workers=10)
```

## Related Modules

- `quantdl.storage.app`: Full historical data upload
- `quantdl.collection`: Data collection from various sources
- `quantdl.universe`: Stock universe management
- `quantdl.utils.calendar`: Trading calendar utilities
