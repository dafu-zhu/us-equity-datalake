## Objective

Build a self-hosted, automated market data infrastructure for US equities using official/authoritative sources, with daily updates and programmatic access via Python API.

**Scope:**
- All US-listed stocks
- Post-2009 data only
- Daily tick data (OHLCV, both adj Close and unadj Close)
- Quarterly fundamental data (ROI, Capitalization, BM, ...)
- Corporate actions (dividends, splits)
- Index constituents (S&P 500, major indices)
- Minute level tick data (OHLCV)
- Derived daily common technical indicators (MACD, RSI, ...)

## Definition of Done / Deliverables

### 1. Flat file system

```
data/
├───ticks/
│   ├───daily/
│   │   ├───AAPL/
│   │   │   ├───2024/
│   │   │   │   └───ticks.json
│   │   │   ├───2023/
│   │   │   │   └───ticks.json
│   │   │   ...
│   ├───minute/
│   │   ├───AAPL/
│   │   │   ├───2024/
│   │   │   │   ├───01/
│   │   │   │   │   ├───15/
│   │   │   │   │   │   └───ticks.parquet
│   │   │   │   │   ...
│   │   │   ...
├───fundamental/
│   ├───AAPL/
│   │   ├───2024/
│   │   │   └───fundamental.json
│   │   │   ...
├───corporate_actions/
│   ├───AAPL/
│   │   └───actions.json
├───reference/
│   ├───ticker_metadata.parquet
│   └───index_constituents.parquet
...
```

**Storage Strategy:**
- Daily ticks: `data/ticks/daily/{symbol}/{YYYY}/ticks.json`
- Minute ticks: `data/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet`
- Fundamentals: `data/fundamental/{symbol}/{YYYY}/fundamental.json`
- Corporate actions: `data/corporate_actions/{symbol}/actions.json` (all dates in single file per symbol)
- Reference data: `data/reference/` (shared files for metadata and index constituents)

### 2. Data Collection Pipeline
- Python scripts for initial data collection
- Automated daily update mechanism
- Error handling and retry logic
- Progress logging and monitoring

### 3. Query API
- Python module for data lake access
- Support for date ranges, multi-symbol queries
- Export capabilities (pandas, CSV, Parquet)

### 4. Data Coverage Report
- Comprehensive analysis document
- Coverage by data type, source, completeness
- Known gaps and limitations documented
- Data quality metrics and validation results

### 5. Documentation
- Setup and installation guide
- Data dictionary (all fields explained)
- API reference with examples
- Maintenance procedures (daily updates, backfills)
- Troubleshooting guide

### 6. Cloud Deployment
- Server configured and running with AWS S3
- Daily automated updates operational
- Backup strategy implemented
- Monitoring/alerting configured

## Background and Assumptions

### Key Assumptions

1. Data Sources (Confirmed):
- Daily Ticks Data (OHLCV): yfinance (2009+)
- Minute Ticks Data (OHLCV): Alpaca Market Data API (2016+)
- Fundamental Data: SEC EDGAR JSON API (official, authoritative, 2009+)
- Corporate Actions: To be determined (candidates: NASDAQ FTP, SEC Form 8-K, yfinance backup)
- Reference Data: NASDAQ FTP + SEC EDGAR for ticker universe (official)
- Index Constituents: To be determined (candidates: S&P website, index provider APIs)

2. Constraints:
- Post-2009 data for daily ticks, fundamentals, corporate actions, reference data
- 2016+ data for minute-level ticks (Alpaca API limitation)
- Daily and minute-level frequency for ticks data
- Quarterly/annual frequency for fundamental data
- US markets only (NYSE, NASDAQ, AMEX)

3. Dependencies:
- Data Fetching Libraries: yfinance (daily ticks), alpaca-py or alpaca-trade-api (minute ticks)
- Data Processing Libraries: pandas, pyarrow (Parquet I/O), open to polars/duckdb for query optimization
- Official APIs: SEC EDGAR JSON API, NASDAQ FTP (for reference data), Alpaca Market Data API
- Cloud Services: AWS S3 (storage), boto3 (S3 client)
- Network Access: Reliable internet for daily API calls
- Rate Limits: yfinance ~2 req/sec, Alpaca free tier has data limits (verify quota)

4. Known Limitations:
- No historical index constituents (survivorship bias present, will build forward from 2024)
- Fundamental data lag: 45-90 days (filing deadlines)
- Small cap coverage: ~75% have fundamental data
- yfinance rate limits: ~2 requests/second

5. Query API
- Do not have rate limitation for my own personal use
- Fetch data directly from cloud service


## Technical Approach and Milestones

### Architecture Overview

```
Data Flow:
Official Sources (Primary) → Python Collectors → Cloud → Query API → Analysis

Components:
1. Data Collection Layer (Python + stdlib + yfinance)
2. Data Storage Layer (Parquet)
3. Update Automation (cron jobs)
4. Query API
    - Use threading to accelerate querying, exceeding the I/O bound
    - Use Polars, DuckDB or PySpark to accelerate querying
5. Cloud Infrastructure (AWS S3)
```

### Roadmap

**Phase 0: Validation & Foundation (Day 1-2)**
1. Prototype with sample data (10-20 symbols, 1 year)
2. Validate storage structure and measure file sizes
3. Setup S3 bucket with lifecycle policies and access controls
4. Create schema definitions for all data types
5. Build logging and monitoring framework

**Phase 1: Reference Data & Infrastructure (Day 3-4)**
6. Implement ticker universe collector (SEC EDGAR + NASDAQ FTP)
7. Build corporate actions collector (determine source: NASDAQ FTP vs SEC vs yfinance)
8. Implement index constituents collector (determine source)
9. Create data validation framework (completeness, accuracy, consistency checks)
10. Setup automated testing infrastructure

**Phase 2: Daily Data Collection (Day 5-8)**
11. Implement daily ticks collector (yfinance) with rate limiting
12. Write unit tests for daily ticks module
13. Backfill daily ticks for recent period (30-90 days) and validate
14. Implement fundamentals collector (SEC EDGAR JSON API)
15. Write unit tests for fundamentals module
16. Backfill fundamentals for recent period (4-8 quarters) and validate
17. Implement technical indicators calculator (MACD, RSI, etc.)

**Phase 3: Minute Data Collection (Day 9-11)**
18. Implement minute ticks collector (Alpaca API)
19. Write unit tests for minute ticks module
20. Backfill minute data for recent period (1-3 months) and validate
21. Monitor storage costs and optimize partition strategy if needed

**Phase 4: Full Historical Backfill (Day 12-15)**
22. Execute full backfill for all data types with progress tracking
23. Implement retry logic and error recovery
24. Run data quality checks and generate coverage report
25. Document known gaps and limitations

**Phase 5: Query API (Day 16-19)**
26. Design and implement core query API (date ranges, multi-symbol)
27. Add caching layer (local file cache)
28. Implement export formats (pandas DataFrame, CSV, Parquet)
29. Write API unit and integration tests
30. Create usage examples and cookbook

**Phase 6: Automation & Production (Day 20-22)**
31. Implement daily update pipeline (scheduled jobs)
32. Add error handling, retry logic, and alerting
33. Setup monitoring dashboard (coverage, latency, errors)
34. Implement backup strategy (S3 versioning)
35. Write operational runbook

**Phase 7: Documentation & Handoff (Day 23-25)**
36. Complete setup and installation guide
37. Document data dictionary (all fields explained)
38. Write API reference documentation
39. Create troubleshooting guide
40. Deliver final coverage report

## Implementation Considerations

### Storage Estimates

**Daily Ticks (EOD, stored by year):**
- Per symbol-year file: ~252 trading days × 8 fields (O/H/L/C/adj_C/V/date/symbol) × 8 bytes = ~16 KB compressed
- Total data: 5000 symbols × 15 years × 16 KB = ~1.2 GB
- Total files: 5000 symbols × 15 years = **75,000 files**

**Minute Ticks (stored by day):**
- Per symbol-day file: 390 minutes × 8 fields × 8 bytes = ~25 KB compressed
- Total data: 5000 symbols × 9 years × 252 days × 25 KB = ~283 GB
- Total files: 5000 symbols × 9 years × 252 days = **11.34 million files**

**Fundamentals (stored by quarter):**
- Per symbol-quarter: ~50 metrics × 500 bytes = ~25 KB
- Total data: 5000 symbols × 60 quarters × 25 KB = ~7.5 GB
- Total files: 5000 symbols × 15 years × 4 quarters = **300,000 files**

**Total Storage:** ~292 GB | **Total Files:** ~11.7 million
**Estimated S3 Costs:** ~$7/month (storage) + API request costs (important with millions of files!)

### Query Optimization Strategies

**Symbol-based partitioning characteristics:**

✅ **Advantages:**
- Single-symbol queries are extremely efficient (direct file access)
- Easy to add/update individual symbols incrementally
- Clear data organization and debugging

⚠ **Challenges:**
- Multi-symbol queries require many file reads (especially minute data)
- S3 LIST operations expensive with millions of files
- Cross-symbol analysis requires aggregation layer

**Recommended Query Patterns:**

1. **Daily Data Queries (by year files):**
   - Single symbol, date range: Read only needed year files
   - Multi-symbol, same period: Parallelize reads with threading (10-50 symbols per batch)

2. **Minute Data Queries (by day files):**
   - Use aggressive parallelization (100+ concurrent S3 reads)
   - Implement local cache for frequently accessed symbol-days
   - Consider creating aggregated views for cross-symbol analysis

3. **Query Engine Recommendation:**
   - **DuckDB** (recommended): Efficient Parquet scanning, SQL interface, handles partial reads
   - Implement result caching at query level (cache popular date ranges)

### Data Quality Framework

**Validation checks per data type:**

```python
Daily Ticks Validation:
- Row count matches trading days (compare vs NYSE calendar)
- No missing OHLC values
- High >= Low >= 0, Close within [Low, High]
- Volume >= 0
- No gaps in date sequence for trading days
- Adjusted close reconciles with corporate actions

Minute Ticks Validation:
- 390 rows per trading day (9:30 AM - 4:00 PM ET)
- Same OHLC sanity checks as daily
- First minute of day: Open should align with previous day close (±5%)
- Aggregate minute data should match daily data (±1%)

Fundamentals Validation:
- Required fields present (revenue, net_income, assets, liabilities, equity)
- Balance sheet equation: Assets = Liabilities + Equity (±1%)
- Filing date <= 90 days after period end
- No negative values for absolute metrics (total assets, market cap)
```

### Edge Case Handling

**1. Corporate Actions (splits, dividends):**
- Store raw prices and adjusted prices separately
- Maintain `corporate_actions/{symbol}/actions.parquet` with adjustment factors
- Fields: symbol, date, action_type (split/dividend), value, adjustment_factor

**2. Symbol Changes / Ticker Renames:**
- Primary data stored under current symbol
- Maintain `reference/ticker_history.parquet`: old_symbol, new_symbol, effective_date
- Query API should support both old and new symbols

**3. Delistings:**
- Keep historical data in same structure
- Flag in `reference/ticker_metadata.parquet`: delisting_date, delisting_reason
- Include in backfills but exclude from daily updates after delisting

**4. Missing Data:**
- Trading halts: Mark as NULL, don't interpolate
- Data source unavailable: Retry with exponential backoff, alert if > 24h gap
- Fundamental filing delays: Expected, track filing date separately
