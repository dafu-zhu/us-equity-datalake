import polars as pl
from pathlib import Path
from typing import List, Optional
from master.security_master import SymbolNormalizer, SecurityMaster

year = 2010
month = 1


def get_hist_universe_crsp(year: int, month: int) -> pl.DataFrame:
    """
    Historical universe common stock list from CRSP database
    Ticker name has no '-' or '.', e.g. BRK.B in alpaca, BRK-B in SEC, BRKB in CRSP

    :param year: Year to fetch
    :param month: Month to fetch
    :return: DataFrame with columns: Ticker (CRSP format), Name
    """
    file_path = Path("data/symbols/history_symbols.csv")

    q = (
        pl.scan_csv(file_path)
        .with_columns(
            pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'),
            pl.col('TSYMBOL').alias('Ticker'),
            pl.col('COMNAM').alias('Name')
        )
        .filter(
            pl.col('date').dt.year().eq(year),
            pl.col('date').dt.month().eq(month)
        )
        .select(['Ticker', 'Name'])
        .drop_nulls()
    )

    return q.collect()


def get_hist_universe_nasdaq(
    year: int,
    month: int,
    with_validation: bool = True
) -> pl.DataFrame:
    """
    Historical universe with symbols converted to Nasdaq format (with periods/hyphens).

    Uses SymbolNormalizer with SecurityMaster validation to prevent false matches
    between delisted stocks and new stocks with similar symbols.

    :param year: Year to fetch
    :param month: Month to fetch
    :param with_validation: Use SecurityMaster to validate symbol conversions
    :return: DataFrame with columns: Ticker (Nasdaq format), Name

    Example:
        get_hist_universe_nasdaq_format(2022, 1)
        # Returns: BRK.B, AAPL, GOOGL, etc. (Nasdaq format)
    """
    # Get historical universe in CRSP format
    crsp_df = get_hist_universe_crsp(year, month)
    crsp_symbols = crsp_df['Ticker'].to_list()

    # Build date string for validation
    day = f"{year}-{month:02d}-15"  # Use mid-month date

    # Initialize normalizer with optional SecurityMaster validation
    if with_validation:
        sm = SecurityMaster()
        normalizer = SymbolNormalizer(security_master=sm)
    else:
        normalizer = SymbolNormalizer()

    # Convert to Nasdaq format with validation
    nasdaq_symbols = normalizer.batch_normalize(crsp_symbols, day=day if with_validation else None)

    # Create result DataFrame
    result = pl.DataFrame({
        'Ticker': nasdaq_symbols,
        'Name': crsp_df['Name'].to_list()
    })

    return result


if __name__ == "__main__":
    print("=" * 70)
    print(f"Example 1: Historical Universe - CRSP Format ({year}-{month:02d})")
    print("=" * 70)

    crsp_df = get_hist_universe_crsp(year, month)
    crsp_symbols = crsp_df['Ticker'].to_list()
    print(f"\nTotal symbols (CRSP format): {len(crsp_symbols)}")
    print(f"First 10 symbols: {crsp_symbols[:10]}")

    print("\n" + "=" * 70)
    print(f"Example 2: Transformation Analysis - Before vs After")
    print("=" * 70)

    # Get both CRSP and Nasdaq format
    nasdaq_df = get_hist_universe_nasdaq(year, month, with_validation=False)
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