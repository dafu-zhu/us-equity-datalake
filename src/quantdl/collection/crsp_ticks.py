"""
WRDS CRSP Daily Data Collection
Fetches OHLCV data from CRSP and converts to Polars DataFrame format matching ticks.py
"""
import wrds
import pandas as pd
import polars as pl
from tqdm import tqdm
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Any
import datetime as dt
from pathlib import Path
from quantdl.master.security_master import SecurityMaster
from quantdl.utils.logger import setup_logger
from quantdl.utils.mapping import align_calendar
from quantdl.utils.validation import validate_date_string, validate_permno
from quantdl.utils.wrds import raw_sql_with_retry
import logging

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

        # Setup logger
        self.logger = setup_logger(
            name="collection.crsp_ticks",
            log_dir="data/logs/ticks",
            level=logging.INFO
        )

    def get_daily(self, symbol: str, day: str, adjusted: bool=True, auto_resolve: bool=True) -> Dict[str, Any]:
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
        permno = validate_permno(self.security_master.sid_to_permno(sid))
        validated_day = validate_date_string(day)

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
                    AND date = '{validated_day}'
                    AND prc IS NOT NULL
                    AND cfacpr != 0
                    AND cfacshr != 0
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
                    AND date = '{validated_day}'
                    AND prc IS NOT NULL
            """
        
        df_pandas = raw_sql_with_retry(self.conn, query, date_cols=['date'])

        # Handle empty data case
        if df_pandas.empty:
            return {}

        row = df_pandas.iloc[0]

        # Check for missing OHLCV data
        if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']) or pd.isna(row['volume']):
            self.logger.warning(f"Missing OHLCV data for {symbol} on {day}")
            return {}

        result = {
            'timestamp': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }

        return result

    def get_daily_range(
            self,
            symbol: str,
            start_day: str,
            end_day: str,
            adjusted: bool=True,
            auto_resolve: bool=True
        ) -> List[Dict[str, Any]]:
        """
        Fetch daily ticks for a symbol across a date range

        Handles symbol changes automatically (e.g., FB -> META)
        Returns one dict per trading day with OHLCV data

        :param symbol: Ticker symbol (can be current or historical)
        :param start_day: Start date in 'YYYY-MM-DD' format
        :param end_day: End date in 'YYYY-MM-DD' format (inclusive)
        :param adjusted: If True, apply split adjustments
        :param auto_resolve: Enable auto_resolve to find security across symbol changes
        :return: List of dicts, one per trading day

        Example:
            get_daily_range('META', '2021-01-01', '2023-01-01')
            # Returns data for FB (2021-2022) and META (2022-2023)
        """
        # Resolve symbol to security_id (use end_date as reference point)
        sid = self.security_master.get_security_id(symbol, end_day, auto_resolve=auto_resolve)

        # Check if symbol resolution returned None (symbol not found or has null security_id)
        if sid is None:
            raise ValueError(
                f"Symbol '{symbol}' not active on {end_day} (security_id is None). "
                "This may indicate the symbol doesn't exist in the security master or has invalid data."
            )

        permno = validate_permno(self.security_master.sid_to_permno(sid))
        validated_start_day = validate_date_string(start_day)
        validated_end_day = validate_date_string(end_day)

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
                    AND date >= '{validated_start_day}'
                    AND date <= '{validated_end_day}'
                    AND prc IS NOT NULL
                    AND cfacpr != 0
                    AND cfacshr != 0
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
                    AND date >= '{validated_start_day}'
                    AND date <= '{validated_end_day}'
                    AND prc IS NOT NULL
                ORDER BY date ASC
            """

        # Execute query
        df_pandas = raw_sql_with_retry(self.conn, query, date_cols=['date'])

        # Handle empty data case
        if df_pandas.empty:
            return []

        # Convert each row to dict
        result = []
        skipped_count = 0
        for _, row in df_pandas.iterrows():
            # Skip rows with missing OHLCV data
            if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']) or pd.isna(row['volume']):
                skipped_count += 1
                continue

            day_data = {
                'timestamp': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            }
            result.append(day_data)

        if skipped_count > 0:
            self.logger.warning(f"Skipped {skipped_count} rows with missing OHLCV data in date range {start_day} to {end_day}")

        return result

    def recent_daily_ticks(
        self,
        symbols: List[str],
        end_day: str,
        window: int = 90,
        adjusted: bool = True,
        auto_resolve: bool = True,
        chunk_size: int = 2000
    ) -> Dict[str, pl.DataFrame]:
        """
        Fetches recent daily data for a list of symbols from CRSP

        :param symbols: List of symbols to fetch
        :param end_day: End date in format 'YYYY-MM-DD'
        :param window: Number of calendar days to fetch (default 90)
        :param adjusted: If True, apply split adjustments
        :param auto_resolve: Enable auto_resolve to handle symbol changes
        :param chunk_size: Number of permnos per SQL query (default 2000)
        :return: Dictionary {symbol: DataFrame} with timestamp, close, volume data
        """
        # Calculate start date
        end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
        start_dt = end_dt - dt.timedelta(days=int(window))
        start_day = start_dt.strftime('%Y-%m-%d')

        self.logger.info(f"Fetching {len(symbols)} symbols from {start_day} to {end_day} (window: {window} days)")

        # Step 1: Bulk resolve symbols to permnos (single lookup)
        self.logger.info("Step 1/3: Resolving symbols to permnos...")
        symbol_to_permno = {}
        failed_symbols = []

        # Pre-resolve security_id from master table for this date (fast path)
        date_check = dt.datetime.strptime(end_day, '%Y-%m-%d').date()
        crsp_keys = [symbol.replace('.', '').replace('-', '').upper() for symbol in symbols]
        resolved_df = (
            self.security_master.master_tb.filter(
                pl.col('symbol').is_in(crsp_keys),
                pl.col('start_date').le(date_check),
                pl.col('end_date').ge(date_check)
            )
            .select(['symbol', 'security_id'])
            .unique(subset=['symbol'], maintain_order=True)
            .filter(pl.col('security_id').is_not_null())
        )
        resolved_map = dict(zip(resolved_df['symbol'].to_list(), resolved_df['security_id'].to_list()))

        auto_resolved = 0
        symbol_to_sid = {}
        for symbol, crsp_key in tqdm(
            list(zip(symbols, crsp_keys)),
            desc="Resolving symbols",
            unit="sym"
        ):
            try:
                sid = resolved_map.get(crsp_key)
                if sid is None and auto_resolve:
                    sid = self.security_master.get_security_id(
                        crsp_key,
                        end_day,
                        auto_resolve=True
                    )
                    auto_resolved += 1
                if sid is None:
                    raise ValueError("security_id is None")
                symbol_to_sid[symbol] = sid
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to resolve {symbol}: {e}")

        if auto_resolve:
            self.logger.info(
                f"Resolved {len(resolved_map)}/{len(symbols)} via fast lookup, "
                f"auto_resolved {auto_resolved} symbols"
            )

        # Bulk map security_id -> permno
        if symbol_to_sid:
            unique_sids = list(set(symbol_to_sid.values()))
            sid_permno_df = (
                self.security_master.security_map()
                .filter(pl.col('security_id').is_in(unique_sids))
                .select(['security_id', 'permno'])
                .unique(subset=['security_id'], maintain_order=True)
            )
            sid_to_permno = dict(
                zip(sid_permno_df['security_id'].to_list(), sid_permno_df['permno'].to_list())
            )
            for symbol, sid in symbol_to_sid.items():
                permno = sid_to_permno.get(sid)
                if permno is None:
                    failed_symbols.append(symbol)
                    self.logger.warning(f"Failed to resolve permno for {symbol} (sid={sid})")
                    continue
                symbol_to_permno[symbol] = permno

        # Log failed symbols summary if any
        if failed_symbols:
            self.logger.warning(f"Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols[:20])}")
            if len(failed_symbols) > 20:
                self.logger.warning(f"... and {len(failed_symbols) - 20} more")

        if not symbol_to_permno:
            self.logger.error("No symbols could be resolved to permnos")
            return {}

        self.logger.info(f"Resolved {len(symbol_to_permno)}/{len(symbols)} symbols to permnos")

        # Step 2: Bulk SQL query for all permnos (chunked)
        self.logger.info("Step 2/3: Executing bulk SQL query...")
        permnos = list(symbol_to_permno.values())
        if chunk_size <= 0:
            chunk_size = len(permnos)
        chunks = [permnos[i:i + chunk_size] for i in range(0, len(permnos), chunk_size)]

        # Validate dates
        validated_start_day = validate_date_string(start_day)
        validated_end_day = validate_date_string(end_day)

        frames = []
        for chunk in tqdm(chunks, desc="Querying CRSP", unit="chunk"):
            # Validate each permno in the chunk
            validated_chunk = [validate_permno(p) for p in chunk]
            permno_list_str = ','.join(map(str, validated_chunk))

            if adjusted:
                query = f"""
                    SELECT
                        permno,
                        date,
                        openprc / cfacpr as open,
                        askhi / cfacpr as high,
                        bidlo / cfacpr as low,
                        abs(prc) / cfacpr as close,
                        vol * cfacshr as volume
                    FROM crsp.dsf
                    WHERE permno IN ({permno_list_str})
                        AND date >= '{validated_start_day}'
                        AND date <= '{validated_end_day}'
                        AND prc IS NOT NULL
                        AND cfacpr IS NOT NULL
                        AND cfacpr != 0
                        AND cfacshr IS NOT NULL
                        AND cfacshr != 0
                    ORDER BY permno, date ASC
                """
            else:
                query = f"""
                    SELECT
                        permno,
                        date,
                        openprc as open,
                        askhi as high,
                        bidlo as low,
                        abs(prc) as close,
                        vol as volume
                    FROM crsp.dsf
                    WHERE permno IN ({permno_list_str})
                        AND date >= '{validated_start_day}'
                        AND date <= '{validated_end_day}'
                        AND prc IS NOT NULL
                    ORDER BY permno, date ASC
                """

            frames.append(raw_sql_with_retry(self.conn, query, date_cols=['date']))

        if frames:
            df_pandas = pd.concat(frames, ignore_index=True)
        else:
            df_pandas = pd.DataFrame()

        self.logger.info(f"Fetched {len(df_pandas)} total rows from CRSP")

        # Step 3: Split results by symbol
        self.logger.info("Step 3/3: Splitting results by symbol...")

        # Create reverse mapping: permno -> symbol
        permno_to_symbol = {v: k for k, v in symbol_to_permno.items()}

        # Convert to polars and split by symbol
        result_dict = {}

        if not df_pandas.empty:
            # Add symbol column
            df_pandas['symbol'] = df_pandas['permno'].map(permno_to_symbol)

            # Convert to polars
            df_polars = pl.from_pandas(df_pandas)

            # Split by symbol
            for symbol in symbol_to_permno.keys():
                symbol_df = (
                    df_polars
                    .filter(pl.col('symbol') == symbol)
                    .with_columns([
                        pl.col('date').cast(pl.Date).alias('timestamp'),
                        pl.col('close').cast(pl.Float64),
                        pl.col('volume').cast(pl.Int64)
                    ])
                    .select(['timestamp', 'close', 'volume'])
                )

                if len(symbol_df) > 0:
                    result_dict[symbol] = symbol_df

        self.logger.info(f"Successfully fetched {len(result_dict)}/{len(symbols)} symbols")
        if failed_symbols:
            self.logger.warning(f"Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols[:20])}")
            if len(failed_symbols) > 20:
                self.logger.warning(f"... and {len(failed_symbols) - 20} more")

        return result_dict

    def collect_daily_ticks_year_bulk(
        self,
        symbols: List[str],
        year: int,
        adjusted: bool = True,
        auto_resolve: bool = True,
        chunk_size: int = 2000
    ) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily data for a list of symbols across a full year.

        Returns a dict mapping symbol -> DataFrame with daily OHLCV rows for the year.
        Data is not calendar-aligned; consumers can align per month if needed.
        """
        start_day = f"{year}-01-01"
        end_day = f"{year}-12-31"

        self.logger.info(f"Fetching {len(symbols)} symbols for {year} via bulk CRSP query")

        symbol_to_permno = {}
        failed_symbols = []

        date_check = dt.datetime.strptime(end_day, '%Y-%m-%d').date()
        crsp_keys = [symbol.replace('.', '').replace('-', '').upper() for symbol in symbols]
        resolved_df = (
            self.security_master.master_tb.filter(
                pl.col('symbol').is_in(crsp_keys),
                pl.col('start_date').le(date_check),
                pl.col('end_date').ge(date_check)
            )
            .select(['symbol', 'security_id'])
            .unique(subset=['symbol'], maintain_order=True)
            .filter(pl.col('security_id').is_not_null())
        )
        resolved_map = dict(zip(resolved_df['symbol'].to_list(), resolved_df['security_id'].to_list()))

        auto_resolved = 0
        symbol_to_sid = {}
        for symbol, crsp_key in tqdm(
            list(zip(symbols, crsp_keys)),
            desc="Resolving symbols",
            unit="sym"
        ):
            try:
                sid = resolved_map.get(crsp_key)
                if sid is None and auto_resolve:
                    sid = self.security_master.get_security_id(
                        crsp_key,
                        end_day,
                        auto_resolve=True
                    )
                    auto_resolved += 1
                if sid is None:
                    raise ValueError("security_id is None")
                symbol_to_sid[symbol] = sid
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to resolve {symbol}: {e}")

        if auto_resolve:
            self.logger.info(
                f"Resolved {len(resolved_map)}/{len(symbols)} via fast lookup, "
                f"auto_resolved {auto_resolved} symbols"
            )

        if symbol_to_sid:
            unique_sids = list(set(symbol_to_sid.values()))
            sid_permno_df = (
                self.security_master.security_map()
                .filter(pl.col('security_id').is_in(unique_sids))
                .select(['security_id', 'permno'])
                .unique(subset=['security_id'], maintain_order=True)
            )
            sid_to_permno = dict(
                zip(sid_permno_df['security_id'].to_list(), sid_permno_df['permno'].to_list())
            )
            for symbol, sid in symbol_to_sid.items():
                permno = sid_to_permno.get(sid)
                if permno is None:
                    failed_symbols.append(symbol)
                    self.logger.warning(f"Failed to resolve permno for {symbol} (sid={sid})")
                    continue
                symbol_to_permno[symbol] = permno

        # Log failed symbols summary if any
        if failed_symbols:
            self.logger.warning(f"Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols[:20])}")
            if len(failed_symbols) > 20:
                self.logger.warning(f"... and {len(failed_symbols) - 20} more")

        if not symbol_to_permno:
            self.logger.error("No symbols could be resolved to permnos")
            return {}

        self.logger.info(f"Resolved {len(symbol_to_permno)}/{len(symbols)} symbols to permnos")

        permnos = list(symbol_to_permno.values())
        if chunk_size <= 0:
            chunk_size = len(permnos)
        chunks = [permnos[i:i + chunk_size] for i in range(0, len(permnos), chunk_size)]

        validated_start_day = validate_date_string(start_day)
        validated_end_day = validate_date_string(end_day)

        frames = []
        for chunk in tqdm(chunks, desc="Querying CRSP", unit="chunk"):
            validated_chunk = [validate_permno(p) for p in chunk]
            permno_list_str = ','.join(map(str, validated_chunk))

            if adjusted:
                query = f"""
                    SELECT
                        permno,
                        date,
                        openprc / cfacpr as open,
                        askhi / cfacpr as high,
                        bidlo / cfacpr as low,
                        abs(prc) / cfacpr as close,
                        vol * cfacshr as volume
                    FROM crsp.dsf
                    WHERE permno IN ({permno_list_str})
                        AND date >= '{validated_start_day}'
                        AND date <= '{validated_end_day}'
                        AND prc IS NOT NULL
                        AND cfacpr IS NOT NULL
                        AND cfacpr != 0
                        AND cfacshr IS NOT NULL
                        AND cfacshr != 0
                    ORDER BY permno, date ASC
                """
            else:
                query = f"""
                    SELECT
                        permno,
                        date,
                        openprc as open,
                        askhi as high,
                        bidlo as low,
                        abs(prc) as close,
                        vol as volume
                    FROM crsp.dsf
                    WHERE permno IN ({permno_list_str})
                        AND date >= '{validated_start_day}'
                        AND date <= '{validated_end_day}'
                        AND prc IS NOT NULL
                    ORDER BY permno, date ASC
                """

            frames.append(raw_sql_with_retry(self.conn, query, date_cols=['date']))

        if frames:
            df_pandas = pd.concat(frames, ignore_index=True)
        else:
            df_pandas = pd.DataFrame()

        self.logger.info(f"Fetched {len(df_pandas)} total rows from CRSP for {year}")

        permno_to_symbol = {v: k for k, v in symbol_to_permno.items()}
        result_dict: Dict[str, pl.DataFrame] = {}

        if not df_pandas.empty:
            df_pandas['symbol'] = df_pandas['permno'].map(permno_to_symbol)
            df_polars = pl.from_pandas(df_pandas)

            for symbol in symbol_to_permno.keys():
                symbol_df = (
                    df_polars
                    .filter(pl.col('symbol') == symbol)
                    .with_columns([
                        pl.col('date').cast(pl.Date).alias('timestamp'),
                        pl.col('open').cast(pl.Float64),
                        pl.col('high').cast(pl.Float64),
                        pl.col('low').cast(pl.Float64),
                        pl.col('close').cast(pl.Float64),
                        pl.col('volume').cast(pl.Int64)
                    ])
                    .select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                )
                if len(symbol_df) > 0:
                    result_dict[symbol] = symbol_df.with_columns(
                        pl.col('timestamp').dt.strftime('%Y-%m-%d')
                    )
        else:
            result_dict = {}

        self.logger.info(f"Bulk fetch produced {len(result_dict)}/{len(symbols)} symbols with data")
        if failed_symbols:
            self.logger.warning(f"Failed symbols ({len(failed_symbols)}): {', '.join(failed_symbols[:20])}")
            if len(failed_symbols) > 20:
                self.logger.warning(f"... and {len(failed_symbols) - 20} more")

        return result_dict

    def collect_daily_ticks(self, symbol: str, year: int, month: int, adjusted: bool=True, auto_resolve: bool=True) -> List[Dict[str, Any]]:
        """
        Collect daily ticks for a specific month and return as DataFrame.

        :param symbol: Stock symbol (CRSP format, e.g., 'BRKB', 'AAPL')
        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param adjusted: If True, apply split adjustments (default: True)
        :param auto_resolve: Enable auto_resolve to handle symbol changes (default: True)
        :return: Polars DataFrame with daily OHLCV data
        """
        # Calculate month range
        start_date_obj = dt.date(year, month, 1)
        if month == 12:
            end_date_obj = dt.date(year, 12, 31)
        else:
            end_date_obj = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')

        # Fetch data using get_daily_range
        daily_data = self.get_daily_range(
            symbol=symbol,
            start_day=start_date,
            end_day=end_date,
            adjusted=adjusted,
            auto_resolve=auto_resolve
        )

        # Handle empty data case (symbol not active in this period)
        if not daily_data:
            return []

        result = align_calendar(daily_data, start_date_obj, end_date_obj, self.calendar_path)

        return result

    def close(self):
        """Close WRDS connection"""
        self.conn.close()
