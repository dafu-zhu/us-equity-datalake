#!/usr/bin/env python3
"""
Search for BBBY (Bed Bath & Beyond) data using CRSP and SEC EDGAR
BBBY was delisted in 2023 after bankruptcy - good test case for delisted securities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collection.wrds_daily import WRDSDailyTicks
from collection.fundamental import Fundamental

def search_crsp_daily_data():
    """Search for BBBY daily data using CRSP"""
    print("=" * 80)
    print("SEARCHING CRSP DAILY DATA FOR BBBY")
    print("=" * 80)

    try:
        wrds = WRDSDailyTicks()

        # Get PERMNO for BBBY
        print("\n1. Looking up PERMNO for ticker BBBY...")
        permno = wrds.get_permno_from_ticker('BBBY')

        if permno:
            print(f"   ✓ Found PERMNO: {permno}")

            # Fetch recent data (2022-2023, before and around delisting)
            print("\n2. Fetching daily data for 2022-2023 (around delisting period)...")
            df = wrds.get_daily_ohlcv_by_ticker(
                ticker='BBBY',
                start_date='2022-01-01',
                end_date='2023-12-31',
                adjusted=True
            )

            if not df.is_empty():
                print(f"   ✓ Retrieved {len(df)} trading days")
                print(f"\n   First 5 rows:")
                print(df.head(5))
                print(f"\n   Last 5 rows:")
                print(df.tail(5))
                print(f"\n   Schema: {df.schema}")

                # Show statistics
                print(f"\n   Price Range:")
                print(f"   - Highest Close: ${df['close'].max():.2f}")
                print(f"   - Lowest Close: ${df['close'].min():.2f}")
                print(f"   - Last Close: ${df['close'][-1]:.2f}")
            else:
                print("   ✗ No data retrieved")
        else:
            print("   ✗ Could not find PERMNO for BBBY")

        wrds.close()

    except Exception as e:
        print(f"   ✗ Error accessing CRSP: {e}")
        print("   Make sure WRDS_USERNAME and WRDS_PASSWORD are set in .env")

def search_sec_fundamental_data():
    """Search for BBBY fundamental data using SEC EDGAR"""
    print("\n" + "=" * 80)
    print("SEARCHING SEC EDGAR FUNDAMENTAL DATA FOR BBBY")
    print("=" * 80)

    # BBBY CIK (Bed Bath & Beyond)
    cik = "0000886158"
    symbol = "BBBY"

    try:
        fund = Fundamental(cik, symbol=symbol)

        print(f"\n1. Connected to SEC EDGAR for CIK {cik}")
        print(f"   Company: {fund.req_response.get('entityName', 'N/A')}")

        # Define key fundamental fields
        fields = [
            'Assets',
            'Liabilities',
            'StockholdersEquity',
            'Revenues',
            'NetIncomeLoss',
            'CashAndCashEquivalentsAtCarryingValue',
            'OperatingIncomeLoss'
        ]

        print(f"\n2. Fetching {len(fields)} fundamental fields for 2022...")
        df = fund.collect_fields(year=2022, fields=fields, location='us-gaap')

        if not df.is_empty():
            print(f"   ✓ Retrieved {len(df)} calendar days with fundamental data")

            # Show non-null data points
            print(f"\n   Sample data (non-null rows):")
            non_null_df = df.filter(pl.col('Assets').is_not_null())
            print(non_null_df.tail(5))

            print(f"\n   Available fields:")
            for field in fields:
                non_null_count = df.filter(pl.col(field).is_not_null()).height
                if non_null_count > 0:
                    latest_value = df.filter(pl.col(field).is_not_null()).tail(1)[field][0]
                    print(f"   - {field}: {non_null_count} data points (latest: ${latest_value:,.0f})")
                else:
                    print(f"   - {field}: No data available")
        else:
            print("   ✗ No fundamental data retrieved")

    except Exception as e:
        print(f"   ✗ Error accessing SEC EDGAR: {e}")

def main():
    # Import polars here for the second function
    global pl
    import polars as pl

    # Search CRSP daily data
    search_crsp_daily_data()

    # Search SEC fundamental data
    search_sec_fundamental_data()

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()