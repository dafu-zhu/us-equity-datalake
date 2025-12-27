import io
import os
import json
import datetime as dt
from typing import List
from pathlib import Path
import logging
import requests
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from boto3.s3.transfer import TransferConfig
import polars as pl
from dotenv import load_dotenv

from utils.logger import setup_logger
from utils.mapping import symbol_cik_mapping
from storage.config_loader import UploadConfig
from storage.s3_client import S3Client
from collection.fundamental import Fundamental
from collection.ticks import Ticks
from collection.models import TickField
from storage.validation import Validator

load_dotenv()


class UploadApp:
    def __init__(self, symbol_file: str="universe_top3000.txt"):
        # Load symbols
        self.symbol_path = Path(f"data/symbols/{symbol_file}")
        self.sec_symbols = self._load_symbols('sec')
        self.alpaca_symbols = self._load_symbols('alpaca')
        self.cik_map = symbol_cik_mapping()

        # Setup config and client
        self.config = UploadConfig()
        self.client = S3Client().client

        # Setup logger
        self.logger = setup_logger(
            name=f"uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.INFO,
            console_output = True
        )

        self.validator = Validator(self.client)

        # Load trading calendar
        self.calendar_path = Path("data/calendar/master.parquet")
        self.trading_days = self._load_trading_days()

        # Load Alpaca Key
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

    def _load_symbols(self, sym_type: str) -> List[str]:
        """
        Load symbol as list from file
        SEC uses '-' as separator ('BRK-B'), Alpaca uses '.' ('BRK.B')

        :param sym_type: "sec" or "alpaca"
        """
        symbols = []
        with open(self.symbol_path, 'r') as file:
            if sym_type == "alpaca":
                for line in file:
                    symbol = line.strip()
                    symbols.append(symbol)
            elif sym_type == "sec":
                for line in file:
                    symbol = line.strip().replace('.', '-')
                    symbols.append(symbol)
            else:
                msg = f"Expected sym_type: 'sec' or 'alpaca', get {sym_type}"
                raise ValueError(msg)

        return symbols

    def _load_trading_days(self, year: int = 2024) -> List[str]:
        """
        Load trading days from master calendar for a specific year.

        :param year: Year to load trading days for (default: 2024)
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .select('Date')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['Date'].to_list()]

    # ===========================
    # Upload ticks
    # ===========================
    def _process_symbol_daily_ticks(self, sym: str, year: int, overwrite: bool = False) -> dict:
        """
        Process daily ticks for a single symbol.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Convert to SEC format for S3 key (BRK.B -> BRK-B)
        sec_symbol = sym.replace('.', '-')

        if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            # Fetch from Alpaca API using Alpaca format (with '.')
            ticks = Ticks(sym)
            daily_df = ticks.collect_daily_ticks(year=year)

            # Check if all data columns are null (no actual data available)
            if daily_df['open'].is_null().all():
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message
            buffer = io.BytesIO()
            daily_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            # Use SEC format (with '-') in S3 key
            s3_key = f"data/raw/ticks/daily/{sec_symbol}/{year}/ticks.parquet"
            s3_metadata = {
                'symbol': sec_symbol,
                'year': str(year),
                'data_type': 'ticks'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def daily_ticks(self, year: int, overwrite: bool = False):
        """
        Upload daily ticks for all symbols sequentially (no concurrency to avoid rate limits).

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        total = len(self.alpaca_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        self.logger.info(f"Starting daily ticks upload for {total} symbols (sequential processing, overwrite={overwrite})")

        for sym in self.alpaca_symbols:
            result = self._process_symbol_daily_ticks(sym, year, overwrite=overwrite)
            completed += 1

            if result['status'] == 'success':
                success += 1
            elif result['status'] == 'canceled':
                canceled += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

            # Progress logging every 10 symbols
            if completed % 10 == 0:
                self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"Daily ticks upload completed: {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total")

    def _upload_minute_ticks_worker(
            self,
            data_queue: queue.Queue,
            stats: dict,
            stats_lock: threading.Lock
        ):
        """
        Worker thread that consumes fetched data and uploads to S3.

        :param data_queue: Queue containing (sym, trade_day, minute_df) tuples
        :param stats: Shared statistics dictionary
        :param stats_lock: Lock for updating statistics
        """
        while True:
            try:
                item = data_queue.get(timeout=1)
                if item is None:  # Poison pill to stop worker
                    break

                sym, trade_day, minute_df = item

                # Convert to SEC format for S3 key
                sec_symbol = sym.replace('.', '-')

                try:
                    # Skip if DataFrame is empty (overwrite check already done before fetching)
                    if len(minute_df) == 0:
                        with stats_lock:
                            stats['skipped'] += 1
                        continue

                    # Setup S3 message
                    buffer = io.BytesIO()
                    minute_df.write_parquet(buffer)
                    buffer.seek(0)

                    # Parse date for S3 key
                    date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')

                    s3_key = f"data/raw/ticks/minute/{sec_symbol}/{year}/{month}/{day}/ticks.parquet"
                    s3_metadata = {
                        'symbol': sec_symbol,
                        'trade_day': trade_day,
                        'data_type': 'ticks'
                    }
                    s3_metadata_prepared = {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                        for k, v in s3_metadata.items()
                    }

                    # Upload to S3
                    self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

                    with stats_lock:
                        stats['success'] += 1

                except Exception as e:
                    self.logger.error(f'Upload error for {sym} on {trade_day}: {e}')
                    with stats_lock:
                        stats['failed'] += 1

                finally:
                    data_queue.task_done()

            except queue.Empty:
                continue

    def _fetch_single_symbol_minute(self, symbol: str, year: int = 2024, sleep_time: float = 0.2) -> List[dict]:
        """
        Fetch minute data for a single symbol for entire year from Alpaca.

        :param symbol: Symbol in Alpaca format (e.g., 'AAPL')
        :param year: Year to fetch (default: 2024)
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: List of bars
        """
        # Get year range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"

        bars = []

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
            # Initial request
            response = session.get(base_url, params=params)
            time.sleep(sleep_time)  # Rate limiting

            if response.status_code != 200:
                self.logger.error(f"Single fetch error for {symbol}: {response.status_code}, {response.text}")
                return bars

            data = response.json()
            symbol_bars = data.get("bars", {}).get(symbol, [])
            bars.extend(symbol_bars)

            # Handle pagination
            page_count = 1
            while "next_page_token" in data and data["next_page_token"]:
                params["page_token"] = data["next_page_token"]
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)  # Rate limiting

                if response.status_code != 200:
                    self.logger.warning(f"Pagination error on page {page_count} for {symbol}: {response.status_code}")
                    break

                data = response.json()
                symbol_bars = data.get("bars", {}).get(symbol, [])
                bars.extend(symbol_bars)

                page_count += 1

            self.logger.info(f"Fetched {len(bars)} bars for {symbol} ({page_count} pages)")

        except Exception as e:
            self.logger.error(f"Exception during single fetch for {symbol}: {e}")

        return bars

    def _fetch_minute_bulk(self, symbols: List[str], year: int = 2024, sleep_time: float = 0.2) -> dict:
        """
        Bulk fetch minute data for multiple symbols for entire year from Alpaca.
        If bulk fetch fails, retry by fetching symbols one by one.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch (default: 2024)
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        # Get year range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        start_str = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")

        # Prepare request
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        symbols_str = ",".join(symbols)
        all_bars = {sym: [] for sym in symbols}

        params = {
            "symbols": symbols_str,
            "timeframe": "1Min",
            "start": start_str,
            "end": end_str,
            "limit": 10000,
            "adjustment": "raw",  # Raw prices for minute data
            "feed": "sip",
            "sort": "asc"
        }

        # Retry logic for bulk fetch
        max_retries = 3
        bulk_success = False

        # OPTIMIZATION: Use persistent session to reuse TCP connections (avoids handshake overhead)
        session = requests.Session()
        session.headers.update(self.headers)

        for retry in range(max_retries):
            try:
                # Initial request (using session for connection reuse)
                response = session.get(base_url, params=params)
                time.sleep(sleep_time)  # Rate limiting

                if response.status_code == 429:
                    # Rate limit error - use exponential backoff with longer waits
                    wait_time = min(60, (2 ** retry) * 5)  # 5s, 10s, 20s (capped at 60s)
                    self.logger.warning(f"Rate limit hit for bulk fetch (retry {retry + 1}/{max_retries}), waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                elif response.status_code != 200:
                    self.logger.error(f"Bulk fetch error for {symbols}: {response.status_code}, {response.text}")
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) * 2  # 2s, 4s, 8s
                        self.logger.warning(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    break

                data = response.json()
                bars = data.get("bars", {})

                # Collect bars from initial response
                for sym in symbols:
                    if sym in bars:
                        all_bars[sym].extend(bars[sym])

                # Handle pagination (session reuses TCP connection - much faster!)
                page_count = 1
                while "next_page_token" in data and data["next_page_token"]:
                    params["page_token"] = data["next_page_token"]
                    response = session.get(base_url, params=params)
                    time.sleep(sleep_time)  # Rate limiting

                    if response.status_code == 429:
                        # Rate limit during pagination - wait and retry this page
                        wait_time = 5
                        self.logger.warning(f"Rate limit hit during pagination on page {page_count}, waiting {wait_time}s")
                        time.sleep(wait_time)
                        response = session.get(base_url, params=params)
                        time.sleep(sleep_time)

                    if response.status_code != 200:
                        self.logger.warning(f"Pagination error on page {page_count} for {symbols}: {response.status_code}")
                        break

                    data = response.json()
                    bars = data.get("bars", {})

                    for sym in symbols:
                        if sym in bars:
                            all_bars[sym].extend(bars[sym])

                    page_count += 1

                self.logger.info(f"Fetched {sum(len(v) for v in all_bars.values())} total bars for {len(symbols)} symbols ({page_count} pages)")
                bulk_success = True
                break

            except Exception as e:
                self.logger.error(f"Exception during bulk fetch for {symbols} (retry {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    wait_time = (2 ** retry) * 2
                    self.logger.warning(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)

        # Close the session after retry loop
        session.close()

        # If bulk fetch failed after retries, fetch symbols one by one
        if not bulk_success:
            self.logger.warning(f"Bulk fetch failed after {max_retries} retries, fetching symbols individually")
            failed_symbols = []

            for sym in symbols:
                try:
                    bars = self._fetch_single_symbol_minute(sym, year, sleep_time=sleep_time)
                    all_bars[sym] = bars
                    if not bars:
                        self.logger.warning(f"No data returned for {sym}")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {sym} individually: {e}")
                    failed_symbols.append(sym)

            if failed_symbols:
                self.logger.error(f"Failed to fetch {len(failed_symbols)} symbols even after individual retry: {failed_symbols}")

        return all_bars

    def minute_ticks(self, year: int, overwrite: bool = False, num_workers: int = 50, chunk_size: int = 30, sleep_time: float = 0.2):
        """
        Upload minute ticks using bulk fetch + concurrent processing.
        Fetches 30 symbols at a time for full year, then processes concurrently.

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        :param num_workers: Number of concurrent processing workers (default: 50)
        :param chunk_size: Number of symbols to fetch at once (default: 30)
        :param sleep_time: Sleep time between API requests in seconds (default: 0.2)
        """
        # Update trading days for the specified year
        self.trading_days = self._load_trading_days(year=year)

        total_symbols = len(self.alpaca_symbols)
        total_days = len(self.trading_days)
        total_tasks = total_symbols * total_days

        # Shared statistics
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()

        # Queue for passing data from parser to upload workers
        data_queue = queue.Queue(maxsize=200)

        self.logger.info(f"Starting minute ticks upload for {total_symbols} symbols × {total_days} days = {total_tasks} tasks")
        self.logger.info(f"Bulk fetching {chunk_size} symbols at a time | {num_workers} concurrent processors | sleep_time={sleep_time}s")

        # Start consumer threads
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self._upload_minute_ticks_worker,
                args=(data_queue, stats, stats_lock),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        # Producer: Bulk fetch and parse
        try:
            for i in range(0, total_symbols, chunk_size):
                chunk = self.alpaca_symbols[i:i + chunk_size]
                chunk_num = i // chunk_size + 1

                # Pre-filter symbols that need data (skip if all days exist and overwrite=False)
                if not overwrite:
                    symbols_to_fetch = []
                    for sym in chunk:
                        sec_symbol = sym.replace('.', '-')
                        # Check if any day is missing for this symbol
                        needs_fetch = False
                        for day in self.trading_days:
                            if not self.validator.data_exists(sec_symbol, 'ticks', day=day):
                                needs_fetch = True
                                break
                        if needs_fetch:
                            symbols_to_fetch.append(sym)
                        else:
                            # All days exist, mark them all as canceled
                            for day in self.trading_days:
                                with stats_lock:
                                    stats['canceled'] += 1
                                    stats['completed'] += 1

                    if not symbols_to_fetch:
                        self.logger.info(f"Skipping chunk {chunk_num}: all data already exists")
                        continue

                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(symbols_to_fetch)}/{len(chunk)} symbols (skipped {len(chunk) - len(symbols_to_fetch)})")
                    chunk = symbols_to_fetch
                else:
                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(chunk)} symbols")

                # Bulk fetch for entire year
                start = time.perf_counter()
                symbol_bars = self._fetch_minute_bulk(chunk, year=year, sleep_time=sleep_time)
                print(f"_fetch_minute_bulk: {time.perf_counter()-start:.2f}s")
                start = time.perf_counter()
                # Parse and organize by (symbol, day) - OPTIMIZED with vectorized operations
                for sym in chunk:
                    bars = symbol_bars.get(sym, [])

                    if not bars:
                        # No data for this symbol
                        for day in self.trading_days:
                            with stats_lock:
                                stats['skipped'] += 1
                                stats['completed'] += 1
                        continue

                    try:
                        # OPTIMIZATION: Convert all bars to DataFrame at once using vectorized operations
                        timestamps = [bar[TickField.TIMESTAMP.value] for bar in bars]
                        opens = [bar[TickField.OPEN.value] for bar in bars]
                        highs = [bar[TickField.HIGH.value] for bar in bars]
                        lows = [bar[TickField.LOW.value] for bar in bars]
                        closes = [bar[TickField.CLOSE.value] for bar in bars]
                        volumes = [bar[TickField.VOLUME.value] for bar in bars]
                        num_trades_list = [bar[TickField.NUM_TRADES.value] for bar in bars]
                        vwaps = [bar[TickField.VWAP.value] for bar in bars]

                        # Create DataFrame and process with vectorized operations (very fast)
                        all_bars_df = pl.DataFrame({
                            'timestamp_utc': timestamps,
                            'open': opens,
                            'high': highs,
                            'low': lows,
                            'close': closes,
                            'volume': volumes,
                            'num_trades': num_trades_list,
                            'vwap': vwaps
                        }, strict=False).with_columns([
                            # Parse timestamp: UTC -> ET, remove timezone (all vectorized, no Python loops)
                            # Use strptime with explicit format and timezone to handle 'Z' marker
                            pl.col('timestamp_utc')
                                .str.strptime(pl.Datetime('us', 'UTC'), format='%Y-%m-%dT%H:%M:%SZ')
                                .dt.convert_time_zone('America/New_York')
                                .dt.replace_time_zone(None)
                                .alias('timestamp'),
                            # Extract trade date for filtering by day (fast string slice)
                            pl.col('timestamp_utc')
                                .str.slice(0, 10)  # Just extract 'YYYY-MM-DD' from '2024-01-03T14:30:00Z'
                                .alias('trade_date'),
                            # Cast types (vectorized)
                            pl.col('open').cast(pl.Float64),
                            pl.col('high').cast(pl.Float64),
                            pl.col('low').cast(pl.Float64),
                            pl.col('close').cast(pl.Float64),
                            pl.col('volume').cast(pl.Int64),
                            pl.col('num_trades').cast(pl.Int64),
                            pl.col('vwap').cast(pl.Float64)
                        ]).drop('timestamp_utc')

                        # Process each trading day by filtering
                        for day in self.trading_days:
                            day_df = all_bars_df.filter(pl.col('trade_date') == day)

                            if len(day_df) > 0:
                                # Select final columns for upload
                                minute_df = day_df.select([
                                    'timestamp', 'open', 'high', 'low', 'close',
                                    'volume', 'num_trades', 'vwap'
                                ])
                            else:
                                # Empty DataFrame for days with no data
                                minute_df = pl.DataFrame()

                            # Put in queue for upload
                            data_queue.put((sym, day, minute_df))

                            with stats_lock:
                                stats['completed'] += 1
                                completed = stats['completed']

                            # Progress logging
                            if completed % 100 == 0:
                                with stats_lock:
                                    self.logger.info(
                                        f"Progress: {completed}/{total_tasks} "
                                        f"({stats['success']} success, {stats['failed']} failed, "
                                        f"{stats['canceled']} canceled, {stats['skipped']} skipped)"
                                    )

                    except Exception as e:
                        self.logger.error(f"Error processing bars for {sym}: {e}", exc_info=True)
                        # Mark all days as failed for this symbol
                        for day in self.trading_days:
                            data_queue.put((sym, day, pl.DataFrame()))
                            with stats_lock:
                                stats['failed'] += 1
                                stats['completed'] += 1
                print(f"loop: {time.perf_counter()-start:.2f}s")
        except Exception as e:
            self.logger.error(f"Error in bulk fetch/parse: {e}", exc_info=True)

        finally:
            # Signal workers to stop
            for _ in range(num_workers):
                data_queue.put(None)

            # Wait for all workers to finish
            for worker in workers:
                worker.join()

        self.logger.info(
            f"Minute ticks upload completed: {stats['success']} success, {stats['failed']} failed, "
            f"{stats['canceled']} canceled, {stats['skipped']} skipped out of {total_tasks} total"
        )

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(self, sym: str, year: int, dei_fields: List[str], gaap_fields: List[str], overwrite: bool = False) -> dict:
        """
        Process fundamental data for a single symbol.
        Returns dict with status for progress tracking.

        :param overwrite: If True, skip existence check and overwrite existing data
        """
        if not overwrite and self.validator.data_exists(sym, 'fundamental', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            cik = self.cik_map[sym]

            # Fetch from SEC EDGAR API
            fnd = Fundamental(cik, sym)

            # Load data on RAM
            dei_df = fnd.collect_fields(year=year, fields=dei_fields, location='dei')
            gaap_df = fnd.collect_fields(year=year, fields=gaap_fields, location='us-gaap')

            # Merge on Date column
            combined_df = dei_df.join(
                gaap_df,
                on='Date',
                how='inner'
            )

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/raw/fundamental/{sym}/{year}/fundamental.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'fundamental'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except KeyError as e:
            self.logger.warning(f'Invalid symbol ({sym}) for CIK mapping: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except ValueError as e:
            self.logger.error(f'Invalid data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym} (CIK {cik}): {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def fundamental(self, year: int, max_workers: int = 20, overwrite: bool = False):
        """
        Upload fundamental data for all symbols using threading.

        :param max_workers: Number of concurrent threads (default: 20)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Fields
        dei_fields = self.config.dei_fields
        gaap_fields = self.config.us_gaap_fields

        total = len(self.sec_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0

        self.logger.info(f"Starting fundamental upload for {total} symbols with {max_workers} workers (overwrite={overwrite})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_fundamental, sym, year, dei_fields, gaap_fields, overwrite): sym
                for sym in self.sec_symbols
            }

            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                else:
                    failed += 1

                # Progress logging every 10 symbols
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled)")

        self.logger.info(f"Fundamental upload completed: {success} success, {failed} failed, {canceled} canceled out of {total} total")

    def upload_fileobj(
            self, 
            data: io.BytesIO, 
            key: str, 
            metadata: dict=None
        ) -> None:
        """Upload file object to S3 with proper configuration"""

        # Define transfer config
        cfg = self.config.transfer
        transfer_config = TransferConfig(
            multipart_threshold=int(eval(cfg.get('multipart_threshold', 10*1024*1024))),
            max_concurrency=int(cfg.get('max_concurrency', 5)),
            multipart_chunksize=int(eval(cfg.get('multipart_chunksize', 10*1024*1024))),
            num_download_attempts=int(cfg.get('num_download_attempts', 5)),
            max_io_queue=int(cfg.get('max_io_queue', 100)),
            io_chunksize=int(cfg.get('io_chunksize', 262144)),
            use_threads=(str(cfg.get('use_threads', True)).lower() == "true")
        )

        # Determine content type
        content_type_map = {
            '.parquet': 'application/x-parquet',
            '.json': 'application/json',
            '.csv': 'text/csv'
        }
        file_ext = Path(key).suffix
        content_type = content_type_map.get(file_ext, 'application/octet-stream')

        # Build ExtraArgs
        extra_args = {
            'ContentType': content_type,
            'ServerSideEncryption': 'AES256',
            'StorageClass': 'INTELLIGENT_TIERING',
            'Metadata': metadata or {}
        }

        # Upload
        self.client.upload_fileobj(
            Fileobj=data,
            Bucket='us-equity-datalake',
            Key=key,
            Config=transfer_config,
            ExtraArgs=extra_args
        )
    
    def run(
            self, 
            start_year: int, 
            end_year: int, 
            max_workers: int=50, 
            overwrite: bool=False, 
            chunk_size: int=30, 
            sleep_time: float=0.02
        ) -> None:
        """
        Run the complete workflow, fetch and upload fundamental, daily ticks and minute ticks data within the period
        """
        year = start_year
        while year <= end_year:
            self.fundamental(year, max_workers, overwrite)
            self.daily_ticks(year, overwrite)
            if year >= 2016:
                self.minute_ticks(year, overwrite, max_workers, chunk_size, sleep_time)

            year += 1

if __name__ == "__main__":
    app = UploadApp()
    app.run(start_year=2010, end_year=2025)