import os
import logging
import requests
import time
import zoneinfo
import datetime as dt
import warnings
from typing import Tuple, List, Dict, Any
from dataclasses import asdict
from dotenv import load_dotenv
import polars as pl
from pathlib import Path

from collection.models import TickField, TickDataPoint
from utils.logger import LoggerFactory
from utils.mapping import align_calendar

load_dotenv()


class Ticks:
    def __init__(self) -> None:

        # Setup logger
        self.logger = LoggerFactory(
            log_dir='data/logs/ticks',
            level=logging.INFO,
            daily_rotation=True,
            console_output=False
        ).get_logger(name=f'collection.ticks')

        # Load Alpaca API key and secrets
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        self.calendar_dir = Path("data/calendar")
        self.calendar_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_trade_day_range(trade_day: str) -> Tuple[str, str]:
        """
        Get the full UTC time range for a trading day (9:30 AM - 4:00 PM ET)

        :param trade_day: Specify trade day
        :param type:
            str: format "YYYY-MM-DD"
            datetime.date:
        :return: start, end
        """
        trade_date = dt.datetime.strptime(trade_day, "%Y-%m-%d").date()

        # Declare ET time zone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = dt.datetime.combine(trade_date, dt.time(9, 30), tzinfo=eastern)
        end_et = dt.datetime.combine(trade_date, dt.time(16, 0), tzinfo=eastern)

        # Transform into UTC time
        start_str = start_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = end_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return start_str, end_str
    
    def recent_daily_ticks(self, symbols: List[str], end_day: str, window: int=90) -> Dict[str, pl.DataFrame]:
        """
        Fetches recent daily data for a list of symbols using Alpaca's multi-symbol API.

        Note: Alpaca's limit parameter applies to TOTAL data points across all symbols,
        not per symbol. This function handles pagination with next_page_token.

        :param symbols: List of symbols to fetch
        :param end_day: End date in format 'YYYY-MM-DD'
        :param window: Number of calendar days to fetch
        :return: Dictionary {symbol: DataFrame} with price and volume data
        """
        # Calculate start date (add buffer for weekends/holidays)
        end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
        start_dt = end_dt - dt.timedelta(days=int(window))

        # Convert to UTC timestamps for Alpaca API
        start_utc = dt.datetime.combine(start_dt.date(), dt.time(0, 0), tzinfo=dt.timezone.utc)
        end_utc = dt.datetime.combine(end_dt.date(), dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        start_str = start_utc.isoformat().replace("+00:00", "Z")
        end_str = end_utc.isoformat().replace("+00:00", "Z")

        self.logger.info(f"Fetching {len(symbols)} symbols from {start_dt.date()} to {end_day} (window: {window} days)")

        # Prepare symbol list for API
        symbols_str = ','.join(symbols)

        # Collect all data using pagination
        all_data = {}
        page_token = None
        page_count = 0

        while True:
            page_count += 1
            url = "https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": symbols_str,
                "timeframe": "1Day",
                "start": start_str,
                "end": end_str,
                "limit": 10000,
                "adjustment": "split",
                "feed": "sip",
                "sort": "asc"
            }

            if page_token:
                params["page_token"] = page_token

            try:
                response = requests.get(url, headers=self.headers, params=params)
                time.sleep(0.1)  # Rate limiting: 200/min = ~3/sec, use 10/sec to be safe

                if response.status_code != 200:
                    self.logger.error(f"API Error [{response.status_code}]: {response.text}")
                    break

                data = response.json()

                # Accumulate bars for each symbol
                if 'bars' in data and data['bars']:
                    for symbol, bars in data['bars'].items():
                        if symbol not in all_data:
                            all_data[symbol] = []
                        all_data[symbol].extend(bars)

                # Check for next page
                page_token = data.get('next_page_token')
                if not page_token:
                    self.logger.info(f"Completed fetching data in {page_count} page(s)")
                    break

            except Exception as e:
                self.logger.error(f"Request failed on page {page_count}: {e}")
                break

        # Convert collected data to DataFrames
        result_dict = {}
        for symbol, bars in all_data.items():
            if bars:
                try:
                    parsed_ticks = self.parse_ticks(bars)
                    ticks_data = [asdict(dp) for dp in parsed_ticks]

                    df = (
                        pl.DataFrame(ticks_data)
                        .with_columns([
                            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
                            pl.col('open').cast(pl.Float64),
                            pl.col('high').cast(pl.Float64),
                            pl.col('low').cast(pl.Float64),
                            pl.col('close').cast(pl.Float64),
                            pl.col('volume').cast(pl.Int64)
                        ])
                        .select(['timestamp', 'close', 'volume'])
                    )

                    result_dict[symbol] = df

                except Exception as e:
                    self.logger.error(f"Failed to process data for {symbol}: {e}")

        self.logger.info(f"Successfully fetched {len(result_dict)}/{len(symbols)} symbols")
        return result_dict

    def get_ticks(
        self,
        symbol: str,
        start_day: str,
        end_day: str,
        timeframe: str = "1Min",
        adjusted: bool = True
    ) -> List[dict]:
        """
        Get tick data from Alpaca API for one symbol, with specified timeframe and date range.

        :param symbols: A list of symbol names to fetch
        :param start_day: Start datetime in UTC (format: "2025-01-03T14:30:00Z")
        :param end_day: End datetime in UTC (format: "2025-01-03T21:00:00Z")
        :param timeframe: Alpaca timeframe (e.g., "1Min", "1Day", "1Hour")
        :return: List of dictionaries with OHLCV data
        Example:
        {"c": 184.25, "h": 185.88, "l": 183.43, "n": 656956, "o": 184.22, "t": "2024-01-03T05:00:00Z", "v": 58418916, "vw": 184.319693}
        """
        url = f"https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": str(symbol).upper(),
            "timeframe": timeframe,
            "start": start_day,
            "end": end_day,
            "limit": 10000,  # Max
            "adjustment": "split",
            "feed": "sip",
            "sort": "asc"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)

            # Rate limit: 200/m or 10/s
            time.sleep(0.03)

            if response.status_code != 200:
                self.logger.error(f"API Error [{response.status_code}] for {symbol}: {response.text}")
                return []

            data = response.json()

            if 'bars' not in data or data['bars'] is None:
                return []

            if symbol not in data['bars']:
                return []

            return data['bars'][symbol]

        except Exception as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
            return []

    def _get_month_range(self, year: int, month: int) -> tuple[str, str]:
        """
        Calculate UTC time range for a given month.

        :param year: Year
        :param month: Month (1-12)
        :return: Tuple of (start_str, end_str) in ISO format with 'Z' suffix
        """
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        start_str = dt.datetime.combine(
            start_date, dt.time(0, 0), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(
            end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")

        return start_str, end_str

    def _fetch_with_pagination(
        self,
        session: requests.Session,
        base_url: str,
        params: dict,
        symbols: List[str],
        sleep_time: float
    ) -> dict:
        """
        Fetch data with pagination handling from Alpaca API.

        :param session: Requests session with headers configured
        :param base_url: API endpoint URL
        :param params: Initial request parameters
        :param symbols: List of symbols being fetched
        :param sleep_time: Sleep time between requests
        :return: Dict mapping symbol -> list of bars
        """
        all_bars = {sym: [] for sym in symbols}
        page_count = 0

        while True:
            page_count += 1
            response = session.get(base_url, params=params)
            time.sleep(sleep_time)

            if response.status_code != 200:
                self.logger.error(
                    f"API error (page {page_count}): {response.status_code}, {response.text}"
                )
                break

            data = response.json()
            bars = data.get("bars", {})

            for sym in symbols:
                if sym in bars:
                    all_bars[sym].extend(bars[sym])

            # Check for next page
            next_token = data.get("next_page_token")
            if not next_token:
                self.logger.info(
                    f"Fetched {sum(len(v) for v in all_bars.values())} total bars "
                    f"for {len(symbols)} symbols ({page_count} pages)"
                )
                break

            params["page_token"] = next_token

        return all_bars

    def _fetch_minute_bulk_with_retry(
        self,
        symbols: List[str],
        start_str: str,
        end_str: str,
        period_desc: str,
        sleep_time: float = 0.2
    ) -> dict:
        """
        Generic method to fetch minute data with retry and pagination logic.

        :param symbols: List of symbols in Alpaca format
        :param start_str: Start time in UTC ISO format with 'Z' suffix
        :param end_str: End time in UTC ISO format with 'Z' suffix
        :param period_desc: Description for logging (e.g., "2024-12" or "2024-12-31")
        :param sleep_time: Sleep time between requests in seconds
        :return: Dict mapping symbol -> list of bars
        """
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "sip",
            "sort": "asc"
        }

        max_retries = 3
        all_bars = {sym: [] for sym in symbols}
        session = requests.Session()
        session.headers.update(self.headers)

        for retry in range(max_retries):
            try:
                base_url = "https://data.alpaca.markets/v2/stocks/bars"

                # Reset page_token for retries
                if "page_token" in params:
                    del params["page_token"]

                # Initial request
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)

                if response.status_code == 429:
                    wait_time = min(60, (2 ** retry) * 5)  # 5s, 10s, 20s (capped at 60s)
                    self.logger.warning(
                        f"Rate limit hit for {period_desc} (retry {retry + 1}/{max_retries}), "
                        f"waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                    continue
                elif response.status_code != 200:
                    self.logger.error(
                        f"Fetch error for {period_desc}: {response.status_code}, {response.text}"
                    )
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) * 2  # 2s, 4s, 8s
                        self.logger.warning(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    break

                data = response.json()
                bars = data.get("bars", {})

                # Collect initial bars
                for sym in symbols:
                    if sym in bars:
                        all_bars[sym].extend(bars[sym])

                # Handle pagination with rate limit retry
                page_count = 1
                next_token = data.get("next_page_token")

                while next_token:
                    page_count += 1
                    params["page_token"] = next_token
                    response = session.get(base_url, params=params)
                    time.sleep(sleep_time)

                    if response.status_code == 429:
                        wait_time = 5
                        self.logger.warning(
                            f"Rate limit hit during pagination on page {page_count}, "
                            f"waiting {wait_time}s"
                        )
                        time.sleep(wait_time)
                        response = session.get(base_url, params=params)
                        time.sleep(sleep_time)

                    if response.status_code != 200:
                        self.logger.error(
                            f"Pagination error on page {page_count} for {period_desc}: "
                            f"{response.status_code}"
                        )
                        break

                    data = response.json()
                    bars = data.get("bars", {})

                    for sym in symbols:
                        if sym in bars:
                            all_bars[sym].extend(bars[sym])

                    next_token = data.get("next_page_token")

                self.logger.info(
                    f"Fetched {sum(len(v) for v in all_bars.values())} total bars for "
                    f"{len(symbols)} symbols for {period_desc} ({page_count} pages)"
                )
                break

            except Exception as e:
                self.logger.error(
                    f"Exception during fetch for {period_desc} "
                    f"(retry {retry + 1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:
                    wait_time = (2 ** retry) * 2
                    self.logger.warning(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)

        session.close()
        return all_bars

    def fetch_minute_day_bulk(
        self,
        symbols: List[str],
        trade_day: str,
        sleep_time: float = 0.2
    ) -> dict:
        """
        Bulk fetch minute data for multiple symbols for a specific trading day.
        Useful for daily incremental updates without refetching entire month.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param trade_day: Trading day (format: "YYYY-MM-DD")
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        start_str, end_str = self.get_trade_day_range(trade_day)
        return self._fetch_minute_bulk_with_retry(
            symbols=symbols,
            start_str=start_str,
            end_str=end_str,
            period_desc=trade_day,
            sleep_time=sleep_time
        )

    def fetch_minute_month_single(
        self,
        symbol: str,
        year: int,
        month: int,
        sleep_time: float = 0.1
    ) -> List[dict]:
        """
        Fetch minute data for a single symbol for the specified month.

        :param symbol: Symbol in Alpaca format (e.g., 'AAPL')
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between requests in seconds (default: 0.1)
        :return: List of bars
        """
        start_str, end_str = self._get_month_range(year, month)

        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "sip",
            "sort": "asc"
        }

        session = requests.Session()
        session.headers.update(self.headers)

        try:
            base_url = "https://data.alpaca.markets/v2/stocks/bars"
            result = self._fetch_with_pagination(
                session=session,
                base_url=base_url,
                params=params,
                symbols=[symbol],
                sleep_time=sleep_time
            )
            return result.get(symbol, [])

        except Exception as e:
            self.logger.error(f"Exception during single fetch for {symbol}: {e}")
            return []
        finally:
            session.close()

    def fetch_minute_month_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> dict:
        """
        Bulk fetch minute data for multiple symbols for the specified month.
        If bulk fetch fails, retry by fetching symbols one by one.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        start_str, end_str = self._get_month_range(year, month)
        period_desc = f"{year}-{month:02d}"

        # Try bulk fetch with retry logic
        all_bars = self._fetch_minute_bulk_with_retry(
            symbols=symbols,
            start_str=start_str,
            end_str=end_str,
            period_desc=period_desc,
            sleep_time=sleep_time
        )

        # Check if bulk fetch succeeded (at least some data returned)
        bulk_success = any(len(bars) > 0 for bars in all_bars.values())

        # If bulk fetch failed completely, fetch symbols one by one
        if not bulk_success:
            self.logger.info(
                f"Bulk fetch returned no data, fetching symbols individually"
            )
            failed_symbols = []

            for sym in symbols:
                try:
                    bars = self.fetch_minute_month_single(sym, year, month, sleep_time=sleep_time)
                    all_bars[sym] = bars
                    if not bars:
                        self.logger.info(f"No data returned for {sym}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {sym} individually: {e}")
                    failed_symbols.append(sym)

            if failed_symbols:
                self.logger.warning(
                    f"Failed to fetch {len(failed_symbols)} symbols even after "
                    f"individual retry: {failed_symbols}"
                )

        return all_bars

    def get_daily(self, symbol: str, year: int, month: int, adjusted: bool=True) -> List[Dict[str, Any]]:
        """
        Get daily OHLCV data for a specific month with adjusted prices.

        :param symbol: Stock symbol
        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param adjusted: If True, apply split adjustments (default: True)
        :return: List of dictionaries with OHLCV data
        """
        # Calculate month range
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        # Convert to UTC timestamps for API
        start_utc = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
        end_utc = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        start = start_utc.isoformat().replace("+00:00", "Z")
        end = end_utc.isoformat().replace("+00:00", "Z")

        result = self.get_ticks(symbol, start, end, "1Day", adjusted=adjusted)

        return result

    def get_daily_year(self, symbol: str, year: int, adjusted: bool=True) -> pl.DataFrame:
        """
        Get daily OHLCV data for an entire year and return as Polars DataFrame.

        :param symbol: Stock symbol
        :param year: Year (e.g., 2024)
        :param adjusted: If True, apply split adjustments (default: True)
        :return: Polars DataFrame with daily OHLCV data
        """
        # Fetch entire year
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        # Convert to UTC timestamps for API
        start_utc = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
        end_utc = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        start = start_utc.isoformat().replace("+00:00", "Z")
        end = end_utc.isoformat().replace("+00:00", "Z")

        # Fetch ticks
        ticks = self.get_ticks(symbol, start, end, "1Day", adjusted=adjusted)

        # Handle empty data
        if not ticks:
            return pl.DataFrame({
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            })

        # Parse and convert to DataFrame
        parsed_ticks = self.parse_ticks(ticks)
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        df = pl.DataFrame(ticks_data).with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S').dt.date(),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64)
        ]).select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        return df

    @staticmethod
    def parse_ticks(ticks: List[Dict[str, Any]]) -> List[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        eastern = zoneinfo.ZoneInfo("America/New_York")
        datapoints = []

        for tick in ticks:
            if not tick:
                raise ValueError(f"tick is empty before parsing, get {tick}")
            
            # Parse UTC timestamp and convert to ET
            timestamp_utc = dt.datetime.fromisoformat(
                tick[TickField.TIMESTAMP.value].replace('Z', '+00:00')
            )
            # Convert to Eastern Time and remove timezone info for storage
            timestamp_et = timestamp_utc.astimezone(eastern).replace(tzinfo=None)

            dp = TickDataPoint(
                timestamp=timestamp_et.isoformat(),
                open=tick[TickField.OPEN.value],
                high=tick[TickField.HIGH.value],
                low=tick[TickField.LOW.value],
                close=tick[TickField.CLOSE.value],
                volume=tick[TickField.VOLUME.value],
                num_trades=tick[TickField.NUM_TRADES.value],
                vwap=tick[TickField.VWAP.value]
            )
            datapoints.append(dp)

        return datapoints
    
    def collect_daily_ticks(self, symbol: str, year: int, month: int, adjusted: bool=True) -> List[Dict[str, Any]]:
        """
        Collect daily ticks for a specific month and return as list of dicts (JSON format), align with trading days.

        :param symbol: Stock symbol
        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param adjusted: If True, apply split adjustments (default: True)
        :return: List of dictionaries with daily OHLCV data
        """
        ticks = self.get_daily(symbol, year, month, adjusted)
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(ticks)

        # Transform dataclass to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Define date range for the month
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        # Merge with master calendar
        calendar_path = self.calendar_dir / "master.parquet"
        result = align_calendar(ticks_data, start_date, end_date, calendar_path)

        return result
    



if __name__ == "__main__":
    ticks = Ticks()

    # Example 1: Fetch recent daily ticks for multiple symbols (returns dict of DataFrames)
    print("=" * 70)
    print("Example 1: recent_daily_ticks")
    print("=" * 70)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_day = "2024-12-31"
    window = 90  # 90 calendar days (~3 months)

    print(f"Fetching recent data for {len(symbols)} symbols...")
    recent_data = ticks.recent_daily_ticks(symbols=symbols, end_day=end_day, window=window)

    print(f"\nSuccessfully fetched {len(recent_data)} symbols")
    for symbol, df in recent_data.items():
        print(f"\n{symbol}: {len(df)} records")
        print(df.head(3))

    # Example 2: Fetch daily ticks for a full year (returns list of dictionaries)
    print("\n" + "=" * 70)
    print("Example 2: collect_daily_ticks")
    print("=" * 70)

    symbol = "AAPL"
    year = 2024
    month = 12

    print(f"Fetching daily ticks for {symbol} in {year}-{month:02d}...")
    daily_ticks = ticks.collect_daily_ticks(symbol=symbol, year=year, month=month, adjusted=True)

    print(f"\nTotal records: {len(daily_ticks)}")
    print("\nFirst 3 records:")
    print(daily_ticks[:3])

    # Example 3: Bulk fetch minute ticks for multiple symbols for a month
    print("\n" + "=" * 70)
    print("Example 3: fetch_minute_month_bulk")
    print("=" * 70)

    symbols = ["AAPL", "MSFT"]
    year = 2024
    month = 12

    print(f"Fetching minute data for {len(symbols)} symbols for {year}-{month:02d}...")
    minute_data = ticks.fetch_minute_month_bulk(symbols=symbols, year=year, month=month)

    for sym, bars in minute_data.items():
        print(f"\n{sym}: {len(bars)} minute bars")
        if bars:
            print(f"First bar: {bars[0]}")
            print(f"Last bar: {bars[-1]}")

    # Example 4: Bulk fetch minute ticks for a single day (for daily updates)
    print("\n" + "=" * 70)
    print("Example 4: fetch_minute_day_bulk")
    print("=" * 70)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    trade_day = "2024-12-31"

    print(f"Fetching minute data for {len(symbols)} symbols on {trade_day}...")
    day_data = ticks.fetch_minute_day_bulk(symbols=symbols, trade_day=trade_day)

    for sym, bars in day_data.items():
        print(f"\n{sym}: {len(bars)} minute bars")
        if bars:
            print(f"First bar: {bars[0]}")
            print(f"Last bar: {bars[-1]}")