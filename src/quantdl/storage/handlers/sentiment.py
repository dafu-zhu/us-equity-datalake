"""
Sentiment upload handler.

Handles parallel SEC fetches + sequential GPU inference for sentiment analysis.
"""

import io
import time
import logging
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any

from tqdm import tqdm

from quantdl.storage.rate_limiter import RateLimiter


class SentimentHandler:
    """
    Handles sentiment data upload with parallel SEC fetches and sequential GPU inference.

    Architecture:
    - Producer thread: Fetches SEC filings in parallel (5 workers, rate-limited)
    - Consumer (main): Runs FinBERT inference sequentially (GPU bottleneck)
    - Prefetch queue: Buffers up to 20 symbols ahead
    """

    def __init__(
        self,
        s3_client,
        bucket: str,
        cik_resolver,
        universe_manager,
        logger: logging.Logger
    ):
        self.s3_client = s3_client
        self.bucket = bucket
        self.cik_resolver = cik_resolver
        self.universe_manager = universe_manager
        self.logger = logger

        # Stats
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.skipped_exists = 0

    def upload(
        self,
        start_date: str,
        end_date: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Upload sentiment data for all symbols in date range.

        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param overwrite: If True, overwrite existing data
        :return: Dict with stats
        """
        from quantdl.collection.sentiment import SentimentCollector
        from quantdl.derived.sentiment import compute_sentiment_for_cik
        from quantdl.models.finbert import FinBERTModel

        start_time = time.time()

        # Step 1: Build symbol list and prefetch CIKs
        cik_map, symbols_with_cik = self._prefetch_ciks(start_date, end_date)

        if not symbols_with_cik:
            self.logger.warning("No symbols with CIKs found, skipping sentiment upload")
            return self._build_stats(start_time, 0, 0)

        # Step 2: Load FinBERT model
        self.logger.info("Loading FinBERT model...")
        model_start = time.time()
        model = FinBERTModel(logger=self.logger)
        model.load()
        model_time = time.time() - model_start
        self.logger.info(f"Model loaded in {model_time:.1f}s")

        # Step 3: Filter existing and process
        symbols_to_process = self._filter_existing(symbols_with_cik, cik_map, overwrite)

        if not symbols_to_process:
            self.logger.info("No symbols to process")
            model.unload()
            return self._build_stats(start_time, model_time, 0)

        # Step 4: Process with parallel SEC fetches + sequential GPU inference
        fetch_start = time.time()
        rate_limiter = RateLimiter(max_rate=10.0)
        collector = SentimentCollector(rate_limiter=rate_limiter, logger=self.logger)

        try:
            self._process_symbols(
                symbols_to_process=symbols_to_process,
                collector=collector,
                model=model,
                compute_fn=compute_sentiment_for_cik,
                start_date=start_date,
                end_date=end_date
            )
        finally:
            model.unload()
            self.logger.info("Model unloaded")

        fetch_time = time.time() - fetch_start
        return self._build_stats(start_time, model_time, fetch_time, len(symbols_to_process))

    def _prefetch_ciks(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[Dict[str, str], List[str]]:
        """Prefetch CIKs for all symbols in date range."""
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        # Build symbol list from universe
        symbol_reference_year = {}
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
            for sym in symbols:
                if sym not in symbol_reference_year:
                    symbol_reference_year[sym] = year

        alpaca_symbols = list(symbol_reference_year.keys())
        total = len(alpaca_symbols)

        self.logger.info(f"Starting sentiment upload for {total} symbols from {start_date} to {end_date}")
        self.logger.info("Storage: data/derived/features/sentiment/{cik}/sentiment.parquet")

        # Pre-fetch CIKs
        self.logger.info("Pre-fetching CIKs...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_symbols = [
                sym for sym, ref_year in symbol_reference_year.items()
                if ref_year == year
            ]
            if year_symbols:
                cik_map.update(
                    self.cik_resolver.batch_prefetch_ciks(year_symbols, year, batch_size=100)
                )
        self.logger.info(f"CIK pre-fetch completed in {time.time() - prefetch_start:.1f}s")

        # Filter symbols with valid CIKs
        symbols_with_cik = [sym for sym in alpaca_symbols if cik_map.get(sym)]
        self.logger.info(f"Found {len(symbols_with_cik)}/{total} symbols with CIKs")

        return cik_map, symbols_with_cik

    def _filter_existing(
        self,
        symbols_with_cik: List[str],
        cik_map: Dict[str, str],
        overwrite: bool
    ) -> List[Tuple[str, str]]:
        """Filter out already-processed symbols."""
        self.logger.info("Checking for existing sentiment files...")
        symbols_to_process = []

        for sym in tqdm(symbols_with_cik, desc="Check existing", unit="sym"):
            cik = cik_map.get(sym)
            if not cik:
                self.skipped += 1
                continue

            key = f"data/derived/features/sentiment/{cik}/sentiment.parquet"
            if not overwrite:
                try:
                    self.s3_client.head_object(Bucket=self.bucket, Key=key)
                    self.skipped_exists += 1
                    continue
                except:
                    pass

            symbols_to_process.append((sym, cik))

        self.logger.info(
            f"Symbols to process: {len(symbols_to_process)} "
            f"(skipped {self.skipped_exists} existing, {self.skipped} no CIK)"
        )
        return symbols_to_process

    def _process_symbols(
        self,
        symbols_to_process: List[Tuple[str, str]],
        collector,
        model,
        compute_fn,
        start_date: str,
        end_date: str
    ):
        """Process symbols with parallel SEC fetches + sequential GPU inference."""
        self.logger.info(f"Processing {len(symbols_to_process)} symbols...")
        self.logger.info("Strategy: Parallel SEC fetches (10 req/sec) + Sequential FinBERT (GPU)")

        # Queue for prefetched filing texts
        prefetch_queue: Queue = Queue(maxsize=20)
        fetch_done = threading.Event()

        def fetch_worker(sym: str, cik: str) -> Tuple[str, str, Any, str]:
            """Fetch filing texts for a symbol (runs in thread pool)."""
            try:
                filings = collector.get_filings_metadata(
                    cik=cik,
                    start_date=start_date,
                    end_date=end_date
                )
                if not filings:
                    return (sym, cik, None, 'no_filings')

                filing_texts = collector.collect_filing_texts(cik=cik, filings=filings)
                if not filing_texts:
                    return (sym, cik, None, 'no_mda')

                return (sym, cik, filing_texts, 'ok')
            except Exception as e:
                return (sym, cik, None, f'error: {e}')

        def producer_thread():
            """Producer: fetch SEC filings in parallel."""
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(fetch_worker, sym, cik): (sym, cik)
                    for sym, cik in symbols_to_process
                }
                for future in as_completed(futures):
                    prefetch_queue.put(future.result())
            fetch_done.set()

        # Start producer
        producer = threading.Thread(target=producer_thread, daemon=True)
        producer.start()

        # Consumer: process with FinBERT (sequential GPU)
        pbar = tqdm(total=len(symbols_to_process), desc="Sentiment", unit="sym")

        while True:
            if fetch_done.is_set() and prefetch_queue.empty():
                break

            try:
                result = prefetch_queue.get(timeout=1.0)
            except Empty:
                continue

            sym, cik, filing_texts, status = result
            pbar.update(1)

            if status != 'ok':
                if status in ('no_filings', 'no_mda'):
                    self.skipped += 1
                else:
                    self.logger.debug(f"Failed to fetch {sym}: {status}")
                    self.failed += 1
                pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped)
                continue

            try:
                pbar.set_description(f"FinBERT [{sym}]")
                sentiment_df = compute_fn(
                    cik=cik,
                    filing_texts=filing_texts,
                    model=model,
                    symbol=sym,
                    logger=self.logger
                )

                if len(sentiment_df) == 0:
                    self.skipped += 1
                    pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped)
                    continue

                # Upload to S3
                self._upload_sentiment(cik, sentiment_df)
                self.success += 1
                pbar.set_description("Sentiment")
                pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped)

            except Exception as e:
                self.logger.debug(f"Failed to process {sym}: {e}")
                self.failed += 1
                pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped)

        pbar.close()

    def _upload_sentiment(self, cik: str, sentiment_df) -> None:
        """Upload sentiment DataFrame to S3."""
        key = f"data/derived/features/sentiment/{cik}/sentiment.parquet"
        buffer = io.BytesIO()
        sentiment_df.write_parquet(buffer)
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue()
        )

    def _build_stats(
        self,
        start_time: float,
        model_time: float = 0,
        fetch_time: float = 0,
        processed: int = 0
    ) -> Dict[str, Any]:
        """Build stats dictionary."""
        total_time = time.time() - start_time
        avg_rate = processed / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Sentiment upload completed in {total_time:.1f}s: "
            f"{self.success} success, {self.failed} failed, "
            f"{self.skipped} skipped, {self.skipped_exists} already existed"
        )
        self.logger.info(
            f"Performance: Model load={model_time:.1f}s, "
            f"Processing={fetch_time:.1f}s, Avg rate={avg_rate:.2f} sym/sec"
        )

        return {
            'success': self.success,
            'failed': self.failed,
            'skipped': self.skipped,
            'skipped_exists': self.skipped_exists,
            'total_time': total_time,
            'model_time': model_time,
            'fetch_time': fetch_time,
            'avg_rate': avg_rate
        }
