"""
WRDS CRSP Daily Data Collection
Fetches OHLCV data from CRSP and converts to Polars DataFrame format matching ticks.py
"""
import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
from typing import Optional
import datetime as dt
from pathlib import Path
from master.security_master import SecurityMaster

load_dotenv()


class CRSPDailyTicks:
    def __init__(self, conn: Optional[wrds.Connection]=None):
        """Initialize WRDS connection and load master calendar"""
        if conn is None:
            username = os.getenv('WRDS_USERNAME')
            password = os.getenv('WRDS_PASSWORD')

            self.conn = wrds.Connection(
                wrds_username=username,
                wrds_password=password
            )
        else:
            self.conn = conn

        # Setup calendar directory
        self.calendar_dir = Path("data/calendar")
        self.calendar_path = self.calendar_dir / "master.parquet"

        self.security_master = SecurityMaster(db=self.conn)

    def get_daily(self, symbol: str, day: str, adjusted: bool=True, auto_resolve: bool=True) -> dict:
        """
        Find daily ticks for one symbol on a given day

        :param adjusted: Adjustment only applies on stock split
        :param auto_resolve: Enable auto_resolve to find nearest available security_id if symbol not found in the given day; Otherwise throw an ValueError

        Example: get_daily('META', '2021-12-31')
        - 'META' not active on 2021-12-31 (exact match fails)
        - Find security that ever used 'META' → security_id=1234
        - Check if security_id=1234 was active on 2021-12-31 (as 'FB') → YES
        - Fetch data using permno for that security
        """
        sid = self.security_master.get_security_id(symbol, day, auto_resolve=auto_resolve)
        permno = self.security_master.sid_to_permno(sid)

        # Write query to access CRSP database
        # CRSP DSF fields:
        # - date: trading date
        # - openprc: opening price
        # - askhi: high ask price (proxy for high)
        # - bidlo: low bid price (proxy for low)
        # - prc: closing price (negative if bid/ask average, take absolute)
        # - vol: volume (in hundreds of shares, multiply by 100)
        # - cfacpr: cumulative price adjustment factor (for splits/dividends)
        # - cfacshr: cumulative share adjustment factor
        if adjusted:
            # Fetch with adjustment factors
            query = f"""
                SELECT
                    date,
                    openprc / cfacpr as open,
                    askhi / cfacpr as high,
                    bidlo / cfacpr as low,
                    abs(prc) / cfacpr as close,
                    vol * cfacshr as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date = '{day}'
                    AND prc IS NOT NULL
            """
        else:
            # Fetch raw unadjusted prices
            query = f"""
                SELECT
                    date,
                    openprc as open,
                    askhi as high,
                    bidlo as low,
                    abs(prc) as close,
                    vol as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date = '{day}'
                    AND prc IS NOT NULL
            """
        
        df_pandas = self.conn.raw_sql(query, date_cols=['date'])

        # Handle empty data case
        if df_pandas.empty:
            return {}
        
        row = df_pandas.iloc[0]

        result = {
            'timestamp': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }      

        return result

    def get_daily_range(self, symbol: str, start_date: str, end_date: str, adjusted: bool=True, auto_resolve: bool=True) -> list[dict]:
        """
        Fetch daily ticks for a symbol across a date range

        Handles symbol changes automatically (e.g., FB -> META)
        Returns one dict per trading day with OHLCV data

        :param symbol: Ticker symbol (can be current or historical)
        :param start_date: Start date in 'YYYY-MM-DD' format
        :param end_date: End date in 'YYYY-MM-DD' format (inclusive)
        :param adjusted: If True, apply split/dividend adjustments
        :param auto_resolve: Enable auto_resolve to find security across symbol changes
        :return: List of dicts, one per trading day

        Example:
            get_daily_range('META', '2021-01-01', '2023-01-01')
            # Returns data for FB (2021-2022) and META (2022-2023)
        """
        # Resolve symbol to security_id (use end_date as reference point)
        sid = self.security_master.get_security_id(symbol, end_date, auto_resolve=auto_resolve)
        permno = self.security_master.sid_to_permno(sid)

        # Build query for date range
        if adjusted:
            query = f"""
                SELECT
                    date,
                    openprc / cfacpr as open,
                    askhi / cfacpr as high,
                    bidlo / cfacpr as low,
                    abs(prc) / cfacpr as close,
                    vol * cfacshr as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date >= '{start_date}'
                    AND date <= '{end_date}'
                    AND prc IS NOT NULL
                ORDER BY date ASC
            """
        else:
            query = f"""
                SELECT
                    date,
                    openprc as open,
                    askhi as high,
                    bidlo as low,
                    abs(prc) as close,
                    vol as volume
                FROM crsp.dsf
                WHERE permno = {permno}
                    AND date >= '{start_date}'
                    AND date <= '{end_date}'
                    AND prc IS NOT NULL
                ORDER BY date ASC
            """

        # Execute query
        df_pandas = self.conn.raw_sql(query, date_cols=['date'])

        # Handle empty data case
        if df_pandas.empty:
            return []

        # Convert each row to dict
        result = []
        for _, row in df_pandas.iterrows():
            day_data = {
                'timestamp': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
            result.append(day_data)

        return result

    def close(self):
        """Close WRDS connection"""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    wrds_ticks = CRSPDailyTicks()

    # Close connection
    wrds_ticks.close()