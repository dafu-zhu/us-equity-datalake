import os
import logging
import requests
import time
import zoneinfo
import datetime as dt
import warnings
from typing import Tuple, List
from dataclasses import asdict
from dotenv import load_dotenv
import polars as pl
from pathlib import Path
import yfinance as yf

from collection.models import TickField, TickDataPoint
from utils.logger import LoggerFactory

load_dotenv()

# Suppress yfinance warnings and error messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*yfinance.*")

# Suppress yfinance logger output (errors printed to stderr)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Shared logger
_logger_factory = LoggerFactory(
    log_dir='data/logs/ticks',
    level=logging.INFO,
    daily_rotation=True,
    console_output=False
)

class Ticks:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        # Setup logger
        self.logger = _logger_factory.get_logger(name=f'collection.ticks.{symbol}')

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
        self.daily_ticks_df = None
        self.minute_ticks_df = None

        # Store path
        self.daily_dir = Path("data/raw/ticks/daily")
        self.minute_dir = Path("data/raw/ticks/minute")
        self.recent_dir = Path("data/raw/ticks/recent")
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.minute_dir.mkdir(parents=True, exist_ok=True)
        self.recent_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_trade_day_range(trade_day: str | dt.date) -> Tuple[str]:
        """
        Get the full UTC time range for a trading day (9:30 AM - 4:00 PM ET)

        :param trade_day: Specify trade day
        :param type:
            str: format "YYYY-MM-DD"
            datetime.date:
        :return: start, end
        """
        # Parse str day into dt.date object
        if isinstance(trade_day, str):
            trade_day = dt.datetime.strptime(trade_day, "%Y-%m-%d").date()

        # Declare ET time zone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = dt.datetime.combine(trade_day, dt.time(9, 30), tzinfo=eastern)
        end_et = dt.datetime.combine(trade_day, dt.time(16, 0), tzinfo=eastern)

        # Transform into UTC time
        start_str = start_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = end_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return start_str, end_str

    @staticmethod
    def get_year_range(year: str | int) -> Tuple[str]:
        """
        Get the UTC time range for a full year (Jan 1 - Dec 31)

        :param year: Year as string or integer
        :return: start, end in UTC format
        """
        if isinstance(year, str):
            year = int(year)

        # Create start and end dates in UTC
        start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(year, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)

        start_str = start.isoformat().replace("+00:00", "Z")
        end_str = end.isoformat().replace("+00:00", "Z")

        return start_str, end_str
    
    # ==========================================
    # 1. BULK METHODS (For Universe Selection)
    # ==========================================
    def fetch_and_store_bulk(self, symbols: List[str], end_day: str) -> bool:
        """
        Fetches the last 3 months of daily data for a list of symbols using yfinance batch download
        and saves them to the 'recent' folder.

        If success (for at least one symbol), return True; otherwise return False

        :param symbols: List of symbols to fetch in bulk
        :param end_day: Endpoint of the 3 month period
        """
        def _process_batch(batch_symbols: List[str], start_date: str, end_date: str, save_dir: Path) -> int:
            """
            Fetch and process data for a batch of symbols using yfinance batch download

            :param batch_symbols: List of symbols to fetch
            :param start_date: Start date (format: "YYYY-MM-DD")
            :param end_date: End date (format: "YYYY-MM-DD")
            :param save_dir: Directory to save parquet files
            :return: Number of successfully processed symbols
            """
            success_count = 0

            try:
                # Batch download from yfinance
                data = yf.download(
                    tickers=batch_symbols,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    show_errors=False,
                    threads=True
                )

                # Handle empty data
                if data.empty:
                    self.logger.warning(f"No data returned for batch of {len(batch_symbols)} symbols")
                    return 0

                # Process based on single vs multiple symbols
                if len(batch_symbols) == 1:
                    # Single symbol - data is a simple DataFrame
                    symbol = batch_symbols[0]
                    if not data.empty:
                        df = pl.DataFrame({
                            'Date': data.index.date.tolist(),
                            'symbol': [symbol] * len(data),
                            'close': data['Close'].astype(float).tolist(),
                            'volume': data['Volume'].astype(int).tolist()
                        }).with_columns([
                            pl.col('Date').cast(pl.Date),
                            pl.col('close').cast(pl.Float64),
                            pl.col('volume').cast(pl.Int64)
                        ]).select(["Date", "symbol", "close", "volume"])

                        save_path = save_dir / f"{symbol}.parquet"
                        df.write_parquet(save_path)
                        success_count += 1
                else:
                    # Multiple symbols - data is multi-index DataFrame
                    # Extract Close and Volume for each symbol
                    for symbol in batch_symbols:
                        try:
                            # Check if symbol has data
                            if symbol not in data['Close'].columns:
                                self.logger.warning(f"No data for {symbol}")
                                continue

                            # Extract symbol's data
                            symbol_close = data['Close'][symbol]
                            symbol_volume = data['Volume'][symbol]

                            # Remove NaN rows (missing data)
                            valid_mask = ~(symbol_close.isna() | symbol_volume.isna())

                            if not valid_mask.any():
                                self.logger.warning(f"No valid data for {symbol}")
                                continue

                            # Create DataFrame
                            df = pl.DataFrame({
                                'Date': data.index[valid_mask].date.tolist(),
                                'symbol': [symbol] * valid_mask.sum(),
                                'close': symbol_close[valid_mask].astype(float).tolist(),
                                'volume': symbol_volume[valid_mask].astype(int).tolist()
                            }).with_columns([
                                pl.col('Date').cast(pl.Date),
                                pl.col('close').cast(pl.Float64),
                                pl.col('volume').cast(pl.Int64)
                            ]).select(["Date", "symbol", "close", "volume"])

                            # Save to dated directory
                            save_path = save_dir / f"{symbol}.parquet"
                            df.write_parquet(save_path)
                            success_count += 1

                        except Exception as e:
                            self.logger.error(f"Failed to process {symbol}: {e}")
                            continue

                return success_count

            except Exception as e:
                self.logger.error(f"Exception processing batch: {e}")
                return 0

        try:
            # Define 3-month window
            end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
            start_dt = end_dt - dt.timedelta(weeks=12)

            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

            # Extract year and month from end_day for directory structure
            year = end_dt.strftime('%Y')
            month = end_dt.strftime('%m')

            # Create dated directory path
            recent_dated_dir = self.recent_dir / year / month
            recent_dated_dir.mkdir(parents=True, exist_ok=True)

            total_success = 0
            BATCH_SIZE = 100  # yfinance handles 100 symbols well

            self.logger.info(f"Starting bulk fetch for {len(symbols)} symbols from {start_date} to {end_date}")
            self.logger.info(f"Storing to: {recent_dated_dir}")

            # Process in batches
            for i in range(0, len(symbols), BATCH_SIZE):
                batch = symbols[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE

                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")

                batch_success = _process_batch(batch, start_date, end_date, recent_dated_dir)
                total_success += batch_success

                self.logger.info(f"Batch {batch_num} completed: {batch_success}/{len(batch)} successful")

                # Small delay between batches to be respectful to yfinance
                if i + BATCH_SIZE < len(symbols):
                    time.sleep(1)

            # Log final summary
            self.logger.info(f"Bulk fetch completed: {total_success}/{len(symbols)} symbols successful")

            return total_success > 0

        except Exception as e:
            self.logger.error(f"Error during bulk fetch: {e}")
            return False

    # ==========================================
    # 2. SINGLE SYMBOL METHODS (For Historical Data)
    # ==========================================
    def get_ticks(
        self,
        start: str,
        end: str,
        timeframe: str = "1Min"
    ) -> List[dict]:
        """
        Get tick data from Alpaca API for specified timeframe and date range.

        :param start: Start datetime in UTC (format: "2025-01-03T14:30:00Z")
        :param end: End datetime in UTC (format: "2025-01-03T21:00:00Z")
        :param timeframe: Alpaca timeframe (e.g., "1Min", "1Day", "1Hour")
        :return: List of dictionaries with OHLCV data
        Example:
        {"c": 184.25, "h": 185.88, "l": 183.43, "n": 656956, "o": 184.22, "t": "2024-01-03T05:00:00Z", "v": 58418916, "vw": 184.319693}
        """
        symbol = str(self.symbol).upper()
        url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe={timeframe}&start={start}&end={end}&limit=10000&adjustment=raw&feed=sip&sort=asc"

        try:
            response = requests.get(url, headers=self.headers)

            # Rate limit: 200/m or 10/s
            time.sleep(0.3)

            # 1. Check HTTP Status Code
            if response.status_code != 200:
                self.logger.error(f"API Error [{response.status_code}] for {symbol}: {response.text}")
                return []

            data = response.json()

            # 2. Check if 'bars' key exists and is not None
            if 'bars' not in data or data['bars'] is None:
                return []
            
            # 3. Check if the specific symbol exists in the bars
            if symbol not in data['bars']:
                return []

            return data['bars'][symbol]

        except Exception as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
            return []

    def get_minute(self, trade_day: str | dt.date) -> List[dict]:
        """
        Get minute-level OHLCV data for a specific trading day.

        :param trade_day: Trading day (format: "YYYY-MM-DD" or date object)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_trade_day_range(trade_day)
        return self.get_ticks(start, end, timeframe="1Min")

    def get_daily(self, year: str | int, adjustment: str = "all") -> List[dict]:
        """
        Get daily OHLCV data for a full year with adjusted prices.

        :param year: Year as string or integer (e.g., "2024" or 2024)
        :param adjustment: Price adjustment type - 'all' for split+dividend adjusted (default), 'raw' for unadjusted
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_year_range(year)

        # Override timeframe-specific adjustment for daily data
        symbol = str(self.symbol).upper()
        url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe=1Day&start={start}&end={end}&limit=10000&adjustment={adjustment}&feed=sip&sort=asc"

        try:
            response = requests.get(url, headers=self.headers)
            time.sleep(0.3)

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

    def get_daily_yf(self, year: str | int) -> pl.DataFrame:
        """
        Get daily OHLCV data for a full year from yfinance.
        Returns DataFrame with schema matching Alpaca format (with null num_trades and vwap).

        :param year: Year as string or integer (e.g., "2024" or 2024)
        :return: Polars DataFrame with daily ticks (Date, OHLCV, null num_trades/vwap)
        """
        import sys
        import io
        import yfinance as yf

        if isinstance(year, str):
            year = int(year)

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        try:
            # Fetch data from yfinance with stderr suppressed
            self.logger.info(f"Fetching daily data from yfinance for {self.symbol} (year {year})")

            # Redirect stderr to suppress yfinance error messages
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                ticker = yf.Ticker(self.symbol)
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            finally:
                sys.stderr = old_stderr

            if hist.empty:
                self.logger.warning(f"No data returned from yfinance for {self.symbol}")
                return pl.DataFrame()

            # Reset index to get Date as a column
            hist = hist.reset_index()

            # Create Polars DataFrame with schema matching Alpaca format
            df = pl.DataFrame({
                'Date': hist['Date'].dt.date.tolist(),
                'open': hist['Open'].astype(float).tolist(),
                'high': hist['High'].astype(float).tolist(),
                'low': hist['Low'].astype(float).tolist(),
                'close': hist['Close'].astype(float).tolist(),
                'volume': hist['Volume'].astype(int).tolist(),
                'num_trades': [None] * len(hist),  # yfinance doesn't provide this
                'vwap': [None] * len(hist)  # yfinance doesn't provide this
            }).with_columns([
                pl.col('Date').cast(pl.Date),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])

            self.logger.info(f"Successfully fetched {len(df)} trading days from yfinance for {self.symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch data from yfinance for {self.symbol}: {e}")
            return pl.DataFrame()

    @staticmethod
    def parse_ticks(ticks: List[dict]) -> List[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        eastern = zoneinfo.ZoneInfo("America/New_York")
        datapoints = []

        for tick in ticks:
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
    
    def collect_daily_ticks(self, year: int) -> pl.DataFrame:
        """
        Given a year, collect daily ticks of the symbol for that year with a dataframe

        :param year: Specify trade year
        :type year: int
        """
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(self.get_daily(year=year))

        # Transform dataclass to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Handle empty data case - return immediately without merging
        if not ticks_data:
            self.logger.warning(f"No data available for {self.symbol} in year {year}")
            self.daily_ticks_df = pl.DataFrame()
            return pl.DataFrame()

        # Define date range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        # Merge with master calendar
        calendar_path = self.calendar_dir / "master.parquet"
        calendar_lf = (
            pl.scan_parquet(calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .sort('Date')
            .lazy()
        )

        ticks_lf = (
            pl.DataFrame(ticks_data)
            .with_columns([
                pl.col('timestamp').str.to_date(format='%Y-%m-%dT%H:%M:%S').alias('Date'),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])
            .drop("timestamp")
            .lazy()
        )
        calendar_lf = calendar_lf.join_asof(ticks_lf, on='Date')

        result = calendar_lf.collect()
        self.daily_ticks_df = result

        return result

    def collect_daily_ticks_yf(self, year: int) -> pl.DataFrame:
        """
        Given a year, collect daily ticks from yfinance and merge with master calendar.

        :param year: Specify trade year
        :type year: int
        """
        # Fetch data from yfinance (returns DataFrame directly)
        ticks_df = self.get_daily_yf(year=year)

        # Handle empty data case
        if len(ticks_df) == 0:
            self.logger.warning(f"No data available from yfinance for {self.symbol} in year {year}")
            self.daily_ticks_df = pl.DataFrame()
            return pl.DataFrame()

        # Define date range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        # Merge with master calendar
        calendar_path = self.calendar_dir / "master.parquet"
        calendar_lf = (
            pl.scan_parquet(calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .sort('Date')
            .lazy()
        )

        # Join with calendar (yfinance data already has correct schema)
        ticks_lf = ticks_df.lazy()
        calendar_lf = calendar_lf.join_asof(ticks_lf, on='Date')

        result = calendar_lf.collect()
        self.daily_ticks_df = result

        return result
    
    def store_daily_ticks(self, year: int) -> None:
        """
        Store dataframe result from collect_daily_ticks with parquet
        
        :param year: Specify trade year
        :type year: int
        """
        if not isinstance(self.daily_ticks_df, pl.DataFrame):
            self.daily_ticks_df: pl.DataFrame = self.collect_daily_ticks(year=year)

        # Define storage path
        dir_path = self.daily_dir / f"{self.symbol}/{year}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write to Parquet file
        file_path = dir_path / 'ticks.parquet'
        self.daily_ticks_df.write_parquet(file_path, compression='zstd')

        self.logger.info(f"Stored {len(self.daily_ticks_df)} daily ticks to {file_path}")
    
    def collect_minute_ticks(self, trade_day: str) -> pl.DataFrame:
        """
        Given a trade day, collect the minute level tick data with dataframe

        :param trade_day: In format 'YYYY-MM-DD'
        :type trade_day: str
        """
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(self.get_minute(trade_day=trade_day))

        # Convert datapoints to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Handle empty data case
        if not ticks_data:
            self.logger.warning(f"No data available for {self.symbol} on {trade_day}")
            # Return empty DataFrame with correct schema
            df = pl.DataFrame({
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': [],
                'num_trades': [],
                'vwap': []
            }).with_columns([
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])
            self.minute_ticks_df = df
            return df

        # Create DataFrame with appropriate schema based on storage type
        df = pl.DataFrame(ticks_data).with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64),
            pl.col('num_trades').cast(pl.Int64),
            pl.col('vwap').cast(pl.Float64)
        ])

        # No need to merge with master cal
        self.minute_ticks_df = df

        return df

    def store_minute_ticks(self, trade_day: str) -> None:
        """
        Store dataframe result from collect_minute_ticks with parquet
        
        :param trade_day: In format 'YYYY-MM-DD'
        :type trade_day: str
        """
        if not isinstance(self.minute_ticks_df, pl.DataFrame):
            self.minute_ticks_df: pl.DataFrame = self.collect_minute_ticks(trade_day=trade_day)

        # Build directory path based on storage type
        date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        dir_path = self.minute_dir / f"{self.symbol}/{year}/{month}/{day}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write to Parquet file
        file_path = dir_path / 'ticks.parquet'
        self.minute_ticks_df.write_parquet(file_path, compression='zstd')

        self.logger.info(f"Stored {len(self.minute_ticks_df)} minute ticks to {file_path}")



if __name__ == "__main__":
    SYMBOL = "AAPL"

    ticks = Ticks(SYMBOL)

    # Example 1: Fetch and store minute data for a specific day
    print("=" * 50)
    print("Fetching minute data...")
    trade_day = "2025-01-03"
    minutes = ticks.collect_minute_ticks(trade_day)
    ticks.store_minute_ticks(trade_day)
    minute_df = pl.read_parquet(ticks.minute_dir/"AAPL/2025/01/03/ticks.parquet")
    print(minute_df)

    # Example 2: Fetch and store daily data for a full year
    print("\n" + "=" * 50)
    print("Fetching daily data...")
    year = 2024
    daily_bars = ticks.collect_daily_ticks(year)
    ticks.store_daily_ticks(year)
    daily_df = pl.read_parquet(ticks.daily_dir/"AAPL/2024/ticks.parquet")
    print(daily_df)