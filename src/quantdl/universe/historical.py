import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional
import wrds
import os
from dotenv import load_dotenv
from quantdl.master.security_master import SymbolNormalizer, SecurityMaster
from quantdl.utils.wrds import raw_sql_with_retry
from quantdl.utils.validation import validate_date_string, validate_year, validate_month

load_dotenv()

def get_hist_universe_crsp(year: int, month: int = 12, db: Optional[wrds.Connection] = None) -> pl.DataFrame:
    """
    Historical universe common stock list from CRSP database for a given year.
    Returns ALL stocks that were active at ANY POINT during the year to avoid
    survivorship bias (includes mid-year IPOs and delistings).

    Ticker name has no '-' or '.', e.g. BRK.B in alpaca, BRK-B in SEC, BRKB in CRSP

    :param year: Year (e.g., 2024)
    :param month: Month (deprecated, kept for backward compatibility - now uses full year range)
    :param db: Optional WRDS connection (creates new one if not provided)
    :return: DataFrame with columns: Ticker (CRSP format), Name, PERMNO
    """
    from sqlalchemy.exc import OperationalError

    # Create WRDS connection if not provided
    close_db = False
    if db is None:
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        if not username or not password:
            raise ValueError(
                "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables."
            )
        db = wrds.Connection(wrds_username=username, wrds_password=password)
        close_db = True

    try:
        # Validate year
        validated_year = validate_year(year)

        # Use full year range to capture ALL stocks active at any point during the year
        # This eliminates survivorship bias by including:
        # - Stocks that IPO'd mid-year (namedt during year)
        # - Stocks that delisted mid-year (nameendt during year)
        # - Stocks active all year
        year_start = f"{validated_year}-01-01"
        year_end = f"{validated_year}-12-31"

        sql = f"""
        SELECT DISTINCT
            ticker, tsymbol, permno, comnam, shrcd, exchcd
        FROM crsp_a_stock.dsenames
        WHERE namedt <= '{year_end}'
          AND nameendt >= '{year_start}'
          AND ticker IS NOT NULL
          AND shrcd IN (10, 11)
          AND exchcd IN (1, 2, 3)
        ORDER BY ticker;
        """

        # Execute query with retry
        try:
            df = raw_sql_with_retry(db, sql)
        except OperationalError as e:
            # If connection is stale and we own it, recreate it
            if close_db and ("closed the connection" in str(e) or "server closed" in str(e)):
                username = os.getenv('WRDS_USERNAME')
                password = os.getenv('WRDS_PASSWORD')
                try:
                    db.close()
                except:
                    pass
                db = wrds.Connection(wrds_username=username, wrds_password=password)
                df = raw_sql_with_retry(db, sql)
            else:
                raise

        if df.empty:
            return pl.DataFrame({
                'Ticker': [],
                'Name': [],
                'PERMNO': []
            })

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
