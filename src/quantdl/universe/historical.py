import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
import wrds
import os
from dotenv import load_dotenv
from quantdl.master.security_master import SymbolNormalizer, SecurityMaster
from quantdl.utils.validation import validate_date_string, validate_year, validate_month

load_dotenv()


def get_hist_universe_crsp(year: int, month: int, db: Optional[wrds.Connection] = None) -> pl.DataFrame:
    """
    Historical universe common stock list from CRSP database for a given year.
    Queries CRSP directly using year-end "as of" date to get all active stocks.

    Ticker name has no '-' or '.', e.g. BRK.B in alpaca, BRK-B in SEC, BRKB in CRSP

    :param year: Year (e.g., 2024)
    :param month: Month (e.g., 12)
    :param db: Optional WRDS connection (creates new one if not provided)
    :return: DataFrame with columns: Ticker (CRSP format), Name, PERMNO
    """
    # Create WRDS connection if not provided
    close_db = False
    if db is None:
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        db = wrds.Connection(wrds_username=username, wrds_password=password)
        close_db = True

    try:
        # Use end-of-month as "as of" date
        # Validate year and month to ensure they are integers within valid ranges
        validated_year = validate_year(year)
        validated_month = validate_month(month)

        asof = f"{validated_year}-{validated_month:02d}-28"
        validated_asof = validate_date_string(asof)

        sql = f"""
        SELECT DISTINCT
            ticker, tsymbol, permno, comnam, shrcd, exchcd
        FROM crsp_a_stock.dsenames
        WHERE namedt <= '{validated_asof}'
          AND nameendt >= '{validated_asof}'
          AND ticker IS NOT NULL
          AND shrcd IN (10, 11)
          AND exchcd IN (1, 2, 3)
        ORDER BY ticker;
        """

        # Execute query
        df = db.raw_sql(sql)

        # Convert to Polars DataFrame with proper schema
        result = pl.DataFrame({
            'Ticker': df['tsymbol'].str.upper().tolist(),  # Ensure uppercase
            'Name': df['comnam'].tolist(),
            'PERMNO': df['permno'].tolist()
        }).unique(subset=['Ticker'], maintain_order=True)

        return result

    finally:
        # Close connection only if we created it
        if close_db:
            db.close()


def get_hist_universe_nasdaq(
    year: int,
    with_validation: bool = True,
    security_master: Optional[SecurityMaster] = None,
    db: Optional[wrds.Connection] = None
) -> pl.DataFrame:
    """
    Historical universe with symbols converted to Nasdaq format (with periods/hyphens).

    Uses SymbolNormalizer with SecurityMaster validation to prevent false matches
    between delisted stocks and new stocks with similar symbols.

    :param year: Year (e.g., 2024)
    :param with_validation: Use SecurityMaster to validate symbol conversions
    :param security_master: Optional pre-initialized SecurityMaster instance (recommended for batch processing)
    :param db: Optional WRDS connection (for performance when calling multiple times)
    :return: DataFrame with columns: Ticker (Nasdaq format), Name, PERMNO

    Note: Returns all stocks that were active on the last trading day of the year

    Example:
        get_hist_universe_nasdaq(2022)
        # Returns: BRK.B, AAPL, GOOGL, etc. (Nasdaq format)
    """
    # Get historical universe in CRSP format
    crsp_df = get_hist_universe_crsp(year, month=12, db=db)
    crsp_symbols = crsp_df['Ticker'].to_list()

    # Initialize normalizer with optional SecurityMaster validation
    sm_to_close = None
    if with_validation:
        # Use provided security_master or create a new one
        if security_master is not None:
            sm = security_master
        else:
            sm = SecurityMaster()
            sm_to_close = sm  # Mark for closing later

        normalizer = SymbolNormalizer(security_master=sm)
    else:
        normalizer = SymbolNormalizer()

    # Convert to Nasdaq format with validation
    # Use July 1st of the year as reference date for validation
    reference_day = f"{year}-12-31" if with_validation else None
    nasdaq_symbols = normalizer.batch_normalize(crsp_symbols, day=reference_day)

    # Close SecurityMaster connection only if we created it
    if sm_to_close is not None:
        sm_to_close.close()

    # Create result DataFrame
    result = pl.DataFrame({
        'Ticker': nasdaq_symbols,
        'Name': crsp_df['Name'].to_list(),
        'PERMNO': crsp_df['PERMNO'].to_list()
    })

    return result


if __name__ == "__main__":
    year = 2021

    print("=" * 70)
    print(f"Example 1: Historical Universe - CRSP Format ({year})")
    print("=" * 70)

    crsp_df = get_hist_universe_crsp(year, 2)
    crsp_symbols = crsp_df['Ticker'].to_list()
    print(f"\nTotal symbols (CRSP format): {len(crsp_symbols)}")
    print(f"First 10 symbols: {crsp_symbols[:10]}")
    print(f"Sample data:")
    print(crsp_df.head(10))

    print("\n" + "=" * 70)
    print(f"Example 2: Transformation Analysis - Before vs After")
    print("=" * 70)

    # Get both CRSP and Nasdaq format (reuse db connection)
    nasdaq_df = get_hist_universe_nasdaq(year, with_validation=False)
    nasdaq_symbols = nasdaq_df['Ticker'].to_list()

    # Find symbols that changed
    crsp_set = set(crsp_symbols)
    nasdaq_set = set(nasdaq_symbols)

    print(f"\nBefore transformation (CRSP format): {len(crsp_symbols)} symbols")
    print(f"After transformation (Nasdaq format): {len(nasdaq_symbols)} symbols")

    print("\n" + "=" * 70)
    print(f"Example 3: Symbols with Periods (Class Shares)")
    print("=" * 70)

    # Find symbols with periods in Nasdaq format
    symbols_with_periods = []
    for crsp_sym, nasdaq_sym in zip(crsp_symbols, nasdaq_symbols):
        if '.' in nasdaq_sym:
            symbols_with_periods.append((crsp_sym, nasdaq_sym))

    print(f"\nTotal symbols with periods: {len(symbols_with_periods)}")
    print(f"\nCRSP Format -> Nasdaq Format:")
    print("-" * 40)

    # Show first 20 examples
    for crsp_sym, nasdaq_sym in symbols_with_periods[:20]:
        print(f"{crsp_sym:15} -> {nasdaq_sym}")

    if len(symbols_with_periods) > 20:
        print(f"... and {len(symbols_with_periods) - 20} more")

    print("\n" + "=" * 70)
    print(f"Example 4: Summary Statistics")
    print("=" * 70)

    total_crsp = len(crsp_symbols)
    total_nasdaq = len(nasdaq_symbols)
    total_with_periods = len(symbols_with_periods)
    retention_rate = (total_nasdaq / total_crsp * 100) if total_crsp > 0 else 0

    print(f"\nCRSP symbols (original):        {total_crsp:5}")
    print(f"Nasdaq symbols (after transform): {total_nasdaq:5}")
    print(f"Symbols with periods (class shares): {total_with_periods:5}")
    print(f"Retention rate:                  {retention_rate:5.1f}%")

    print("\n" + "=" * 70)
