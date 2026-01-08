"""
Example: Fundamental Data Collection and Upload

This example demonstrates:
1. Using DataCollectors.collect_fundamental_long() to fetch concept-based fundamental data
2. Using DataCollectors.collect_ttm_long() to fetch TTM data
3. Using DataCollectors.collect_ttm_long_range() to fetch TTM data by date range
4. Using DataCollectors.collect_derived_long() to fetch derived metrics in long format
5. Using UploadApp to upload fundamental data to S3

The system uses approved_mapping.yaml for concept-based extraction of all 29 financial metrics.
"""

from pathlib import Path
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.data_collectors import DataCollectors
from src.collection.alpaca_ticks import Ticks
from src.collection.crsp_ticks import CRSPDailyTicks
from src.storage.upload_app import UploadApp
from utils.logger import setup_logger


def _init_collector() -> DataCollectors:
    logger = setup_logger(
        name='example_fundamental',
        log_dir=Path('data/logs/examples'),
        level=logging.INFO,
        console_output=True
    )
    alpaca_ticks = Ticks()
    crsp_ticks = CRSPDailyTicks()
    return DataCollectors(
        crsp_ticks=crsp_ticks,
        alpaca_ticks=alpaca_ticks,
        alpaca_headers={},
        logger=logger
    )


def example_1_collect_fundamental_long():
    """
    Example 1: Use DataCollectors to fetch long-format fundamental data for a single company.

    This method:
    - Fetches all concepts from approved_mapping.yaml
    - Returns long-format data with one row per filing/concept
    - Returns a Polars DataFrame with columns: [symbol, as_of_date, accn, form, concept, value, start, end, fp]
    """
    print("=" * 80)
    print("Example 1: DataCollectors.collect_fundamental_long()")
    print("=" * 80)

    collector = _init_collector()

    # Example: Fetch fundamental data for Apple (AAPL)
    symbol = 'AAPL'
    cik = '0000320193'  # Apple's CIK
    year = 2024
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"\nFetching fundamental data for {symbol} (CIK: {cik}) in {year}...")
    print("Using concept-based extraction with approved_mapping.yaml\n")

    # Fetch data
    df = collector.collect_fundamental_long(
        cik=cik,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol
    )

    # Display results
    if len(df) > 0:
        print(f"✓ Successfully fetched {len(df)} rows")
        print(f"  Columns ({len(df.columns)}): {df.columns}")
        print("\n  First 5 rows:")
        print(df.head(5))
    else:
        print("✗ No data found")

    return df


def example_3_collect_ttm_long_range():
    """
    Example 3: Use DataCollectors to fetch TTM long-format data for a date range.

    This method:
    - Computes TTM in memory
    - Filters by end date within the provided range
    - Returns a Polars DataFrame with columns:
      [symbol, as_of_date, accn, form, concept, value, start, end, fp]
    """
    print("\n" + "=" * 80)
    print("Example 3: DataCollectors.collect_ttm_long_range()")
    print("=" * 80)

    collector = _init_collector()

    symbol = 'AAPL'
    cik = '0000320193'
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    print(f"\nComputing TTM data for {symbol} (CIK: {cik}) from {start_date} to {end_date}...")
    print("Using concept-based extraction with approved_mapping.yaml\n")

    df = collector.collect_ttm_long_range(
        cik=cik,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol
    )

    if len(df) > 0:
        print(f"✓ Successfully computed {len(df)} TTM rows")
        print(f"  Columns ({len(df.columns)}): {df.columns}")
        print("\n  First 5 rows:")
        print(df.head(5))
    else:
        print("✗ No data found")

    return df


def example_4_collect_derived_long():
    """
    Example 4: Use DataCollectors to fetch derived metrics in long format.

    This method:
    - Computes derived metrics from TTM and balance-sheet data
    - Returns a Polars DataFrame with columns:
      [symbol, as_of_date, start, end, accn, form, fp, concept, value]
    """
    print("\n" + "=" * 80)
    print("Example 4: DataCollectors.collect_derived_long()")
    print("=" * 80)

    collector = _init_collector()

    symbol = 'AAPL'
    cik = '0000320193'
    start_date = "2023-01-01"
    end_date = "2024-12-31"

    print(f"\nComputing derived metrics for {symbol} (CIK: {cik}) from {start_date} to {end_date}...")
    print("Using concept-based extraction with approved_mapping.yaml\n")

    df = collector.collect_derived_long(
        cik=cik,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol
    )

    if len(df) > 0:
        print(f"✓ Successfully computed {len(df)} derived rows")
        print(f"  Columns ({len(df.columns)}): {df.columns}")
        print("\n  First 5 rows:")
        print(df.head(5))
    else:
        print("✗ No data found")

    return df


def example_2_upload_fundamental():
    """
    Example 2: Use UploadApp to upload fundamental data to S3.

    This demonstrates:
    - Uploading fundamental data for a specific date range
    - Batch CIK pre-fetching for performance
    - Concept-based extraction with approved_mapping.yaml
    - Concurrent processing with rate limiting
    """
    print("\n" + "=" * 80)
    print("Example 2: UploadApp.fundamental()")
    print("=" * 80)

    # Initialize UploadApp
    app = UploadApp()

    try:
        # Upload fundamental data for 2024
        # This will:
        # - Load all symbols for 2024
        # - Batch pre-fetch CIKs
        # - Fetch data using concept-based extraction
        # - Upload to S3: data/raw/fundamental/{symbol}/fundamental.parquet

        year = 2024
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        max_workers = 50  # Concurrent workers (rate limited to 9.5 req/sec)
        overwrite = False  # Set to True to overwrite existing data

        print(f"\nUploading fundamental data for {year}...")
        print(f"  Max workers: {max_workers} (rate limited to 9.5 req/sec)")
        print(f"  Overwrite: {overwrite}")
        print(f"  Using: approved_mapping.yaml (29 concepts)")
        print("  Storage: data/raw/fundamental/{symbol}/fundamental.parquet\n")

        # CAUTION: This will upload data for ALL symbols in the universe!
        # For testing, you may want to modify the universe in upload_app.py
        # or test with a smaller year/subset

        app.upload_fundamental(
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            overwrite=overwrite
        )

        print(f"\n✓ Upload completed for {year}")

    finally:
        app.close()


def example_3_upload_single_symbol():
    """
    Example 3: Upload fundamental data for a single symbol (manual approach).

    Demonstrates how to upload data for just one company without processing the entire universe.
    """
    print("\n" + "=" * 80)
    print("Example 3: Upload Single Symbol")
    print("=" * 80)

    app = UploadApp()

    try:
        # Manually process a single symbol
        symbol = 'RKLB'  # Rocket Lab USA Inc.
        cik = '1819994'
        year = 2024
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        overwrite = True

        print(f"\nUploading fundamental data for {symbol} (CIK: {cik}) in {year}...")

        result = app._process_symbol_fundamental(
            sym=symbol,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            cik=cik
        )

        # Display result
        print(f"\nResult:")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Status: {result['status']}")
        print(f"  Error: {result.get('error', 'None')}")

        if result['status'] == 'success':
            print(f"\n✓ Successfully uploaded to: data/raw/fundamental/{symbol}/fundamental.parquet")

    finally:
        app.close()


if __name__ == "__main__":
    # Run examples
    print("\n")
    print("=" * 80)
    print("FUNDAMENTAL DATA COLLECTION & UPLOAD EXAMPLES")
    print("=" * 80)
    print()

    # Example 1: Fetch fundamental long-format data using DataCollectors
    df = example_1_collect_fundamental_long()

    # Example 3: Compute TTM long-format data for a date range using DataCollectors
    ttm_range_df = example_3_collect_ttm_long_range()

    # Example 4: Compute derived metrics in long format using DataCollectors
    derived_df = example_4_collect_derived_long()

    # Example 5: Upload single symbol (safer for testing)
    # example_3_upload_single_symbol()

    # Example 6: Upload all symbols for a year (COMMENTED OUT FOR SAFETY)
    # CAUTION: This will process ALL symbols in the universe!
    # Uncomment only when ready to do a full upload
    # example_2_upload_fundamental()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
