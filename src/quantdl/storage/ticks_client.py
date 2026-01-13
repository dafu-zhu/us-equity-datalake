"""
Ticks Client for Querying Daily Ticks Data
===========================================

Provides user-friendly API for querying daily ticks with:
- Symbol-based queries (transparent security_id resolution)
- Session-based caching for performance
- Automatic routing to history or current year data
"""

import datetime as dt
from typing import Optional, Dict, Tuple
import polars as pl
import logging

from quantdl.master.security_master import SecurityMaster


class TicksClient:
    """
    Client for querying daily ticks data with symbol resolution.

    Features:
    - Transparent symbol → security_id resolution
    - Session-based cache for lookups
    - Automatic routing to history.parquet (completed years) or monthly files (current year)
    - Date range filtering

    Storage structure:
    - History: data/raw/ticks/daily/{security_id}/history.parquet (all completed years)
    - Current: data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet (current year)
    """

    def __init__(
        self,
        s3_client,
        bucket_name: str,
        security_master: SecurityMaster,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TicksClient.

        :param s3_client: Boto3 S3 client
        :param bucket_name: S3 bucket name
        :param security_master: SecurityMaster for symbol resolution
        :param logger: Optional logger
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.security_master = security_master
        self.logger = logger or logging.getLogger(__name__)

        # Session-based cache: (symbol, year) → security_id
        self._cache: Dict[Tuple[str, int], int] = {}

    def get_daily_ticks(
        self,
        symbol: str,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch daily ticks for a symbol and year.

        :param symbol: Ticker symbol (e.g., 'AAPL', 'BRK.B')
        :param year: Year to fetch
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Resolve symbol to security_id
        security_id = self._resolve_symbol(symbol, year)

        # Fetch data
        return self._fetch_by_security_id(security_id, year, start_date, end_date)

    def get_daily_ticks_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch full historical daily ticks for a symbol.

        Reads from history.parquet file (all completed years).

        :param symbol: Ticker symbol
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :return: Polars DataFrame with historical data
        """
        # Use end_date year for security_id resolution, or current year
        if end_date:
            year = int(end_date[:4])
        else:
            year = dt.date.today().year

        security_id = self._resolve_symbol(symbol, year)

        # Read history file
        s3_key = f"data/raw/ticks/daily/{security_id}/history.parquet"

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pl.read_parquet(response['Body'])

            # Apply date filters
            if start_date or end_date:
                df = self._apply_date_filter(df, start_date, end_date)

            return df

        except self.s3_client.exceptions.NoSuchKey:
            raise ValueError(
                f"No historical data found for {symbol} (security_id={security_id}). "
                f"Check if data has been uploaded."
            )

    def _resolve_symbol(self, symbol: str, year: int) -> int:
        """
        Resolve symbol to security_id with session caching.

        :param symbol: Ticker symbol
        :param year: Year for resolution
        :return: security_id
        """
        cache_key = (symbol, year)

        if cache_key not in self._cache:
            # Lookup using year-end date
            date = f"{year}-12-31"
            self._cache[cache_key] = self.security_master.get_security_id(symbol, date)
            self.logger.debug(f"Resolved {symbol} ({year}) → security_id={self._cache[cache_key]}")

        return self._cache[cache_key]

    def _fetch_by_security_id(
        self,
        security_id: int,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch data by security_id, automatically routing to history or monthly files.

        :param security_id: Security ID
        :param year: Year to fetch
        :param start_date: Optional start date filter
        :param end_date: Optional end date filter
        :return: Polars DataFrame
        """
        current_year = dt.date.today().year

        if year == current_year:
            # Current year: read from monthly files
            return self._fetch_monthly(security_id, year, start_date, end_date)
        else:
            # Completed year: read from history file
            return self._fetch_history_year(security_id, year, start_date, end_date)

    def _fetch_history_year(
        self,
        security_id: int,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch completed year from history.parquet file.

        :param security_id: Security ID
        :param year: Year to fetch
        :param start_date: Optional start date filter
        :param end_date: Optional end date filter
        :return: Polars DataFrame
        """
        s3_key = f"data/raw/ticks/daily/{security_id}/history.parquet"

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pl.read_parquet(response['Body'])

            # Filter to requested year
            df = df.filter(
                pl.col('timestamp').str.slice(0, 4) == str(year)
            )

            # Apply additional date filters if provided
            if start_date or end_date:
                df = self._apply_date_filter(df, start_date, end_date)

            return df

        except self.s3_client.exceptions.NoSuchKey:
            raise ValueError(
                f"No historical data found for security_id={security_id}. "
                f"Check if data has been uploaded and consolidated."
            )

    def _fetch_monthly(
        self,
        security_id: int,
        year: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch current year data from monthly partitions.

        :param security_id: Security ID
        :param year: Year to fetch
        :param start_date: Optional start date filter
        :param end_date: Optional end date filter
        :return: Polars DataFrame
        """
        # Determine which months to read
        start_month, end_month = self._determine_months(start_date, end_date)

        # Read monthly files
        monthly_dfs = []
        for month in range(start_month, end_month + 1):
            s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"

            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                monthly_dfs.append(pl.read_parquet(response['Body']))
            except self.s3_client.exceptions.NoSuchKey:
                # Month doesn't exist, skip
                self.logger.debug(f"Month file not found: {s3_key}")
                continue

        if not monthly_dfs:
            raise ValueError(
                f"No data found for security_id={security_id}, year={year}. "
                f"Check if data has been uploaded."
            )

        # Concatenate monthly data
        df = pl.concat(monthly_dfs).sort('timestamp')

        # Apply date filters
        if start_date or end_date:
            df = self._apply_date_filter(df, start_date, end_date)

        return df

    def _determine_months(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[int, int]:
        """
        Determine which months to read based on date filters.

        :param start_date: Start date (YYYY-MM-DD) or None
        :param end_date: End date (YYYY-MM-DD) or None
        :return: Tuple of (start_month, end_month) both inclusive (1-12)
        """
        start_month = 1
        end_month = 12

        if start_date:
            start_month = int(start_date[5:7])

        if end_date:
            end_month = int(end_date[5:7])

        return start_month, end_month

    def _apply_date_filter(
        self,
        df: pl.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pl.DataFrame:
        """
        Apply date range filter to DataFrame.

        :param df: Input DataFrame
        :param start_date: Start date (YYYY-MM-DD) or None
        :param end_date: End date (YYYY-MM-DD) or None
        :return: Filtered DataFrame
        """
        if start_date:
            df = df.filter(pl.col('timestamp') >= start_date)

        if end_date:
            df = df.filter(pl.col('timestamp') <= end_date)

        return df

    def clear_cache(self):
        """Clear the symbol resolution cache."""
        self._cache.clear()
        self.logger.debug("Symbol resolution cache cleared")
