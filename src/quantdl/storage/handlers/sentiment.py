"""
Sentiment upload handler.

Handles parallel SEC fetches + sequential GPU inference for sentiment analysis.
Uses S3 caching for MD&A texts to avoid re-fetching from SEC.
"""

from __future__ import annotations

import io
import json
import time
import logging
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any

from tqdm import tqdm

from quantdl.storage.utils import RateLimiter

if TYPE_CHECKING:
    from quantdl.storage.utils import CIKResolver
    from quantdl.universe.manager import UniverseManager


class SentimentCheckpoint:
    """
    Checkpoint tracker for sentiment processing.

    Tracks which CIKs have been fully processed to allow resume.
    Path: data/cache/sentiment_checkpoint.json
    """

    def __init__(self, s3_client, bucket: str, logger: logging.Logger):
        self.s3_client = s3_client
        self.bucket = bucket
        self.logger = logger
        self._checkpoint_key = "data/cache/sentiment_checkpoint.json"
        self._processed: set = set()
        self._load()

    def _load(self) -> None:
        """Load checkpoint from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=self._checkpoint_key
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            self._processed = set(data.get('processed_ciks', []))
            self.logger.info(f"Loaded checkpoint: {len(self._processed)} CIKs already processed")
        except:
            self._processed = set()
            self.logger.info("No checkpoint found, starting fresh")

    def _save(self) -> None:
        """Save checkpoint to S3."""
        data = {
            'processed_ciks': list(self._processed),
            'count': len(self._processed),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=self._checkpoint_key,
            Body=json.dumps(data).encode('utf-8')
        )

    def is_processed(self, cik: str) -> bool:
        """Check if CIK has been fully processed."""
        return cik in self._processed

    def mark_processed(self, cik: str) -> None:
        """Mark CIK as fully processed and save checkpoint."""
        self._processed.add(cik)
        # Save every 10 CIKs to reduce S3 calls
        if len(self._processed) % 10 == 0:
            self._save()

    def save(self) -> None:
        """Force save checkpoint."""
        self._save()

    def count(self) -> int:
        """Return number of processed CIKs."""
        return len(self._processed)


class MDACache:
    """
    S3 cache for extracted MD&A texts.

    Stores MD&A texts in S3 to avoid re-fetching from SEC.
    Path: data/cache/mda/{cik}/mda_cache.json
    """

    def __init__(self, s3_client, bucket: str, logger: logging.Logger):
        self.s3_client = s3_client
        self.bucket = bucket
        self.logger = logger
        self._local_cache: Dict[str, Dict] = {}  # cik -> filing_texts dict

    def _cache_key(self, cik: str) -> str:
        return f"data/cache/mda/{cik}/mda_cache.json"

    def get(self, cik: str) -> Optional[Dict]:
        """Get cached MD&A texts for a CIK."""
        if cik in self._local_cache:
            return self._local_cache[cik]

        key = self._cache_key(cik)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            self._local_cache[cik] = data
            return data
        except:
            return None

    def put(self, cik: str, filing_texts: List) -> None:
        """Cache MD&A texts for a CIK."""
        # Convert FilingText objects to dicts
        data = {
            'cik': cik,
            'filings': [ft.to_dict() for ft in filing_texts]
        }

        key = self._cache_key(cik)
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data).encode('utf-8')
        )
        self._local_cache[cik] = data

    def has(self, cik: str) -> bool:
        """Check if CIK has cached data."""
        if cik in self._local_cache:
            return True

        key = self._cache_key(cik)
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False


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
        cik_resolver: CIKResolver,
        universe_manager: UniverseManager,
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
        mda_cache = MDACache(self.s3_client, self.bucket, self.logger)
        checkpoint = SentimentCheckpoint(self.s3_client, self.bucket, self.logger)

        # Filter out already-checkpointed CIKs
        symbols_after_checkpoint = [
            (sym, cik) for sym, cik in symbols_to_process
            if not checkpoint.is_processed(cik)
        ]
        skipped_checkpoint = len(symbols_to_process) - len(symbols_after_checkpoint)
        if skipped_checkpoint > 0:
            self.logger.info(f"Skipping {skipped_checkpoint} CIKs from checkpoint (already processed)")

        try:
            self._process_symbols(
                symbols_to_process=symbols_after_checkpoint,
                collector=collector,
                model=model,
                compute_fn=compute_sentiment_for_cik,
                start_date=start_date,
                end_date=end_date,
                mda_cache=mda_cache,
                checkpoint=checkpoint
            )
        finally:
            checkpoint.save()  # Final checkpoint save
            model.unload()
            self.logger.info("Model unloaded, checkpoint saved")

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
        end_date: str,
        mda_cache: Optional[MDACache] = None,
        checkpoint: Optional[SentimentCheckpoint] = None
    ):
        """Process symbols with parallel SEC fetches + batched GPU inference."""
        from quantdl.derived.sentiment import chunk_text, _aggregate_sentiment_results
        from quantdl.collection.sentiment import FilingText

        self.logger.info(f"Processing {len(symbols_to_process)} symbols...")
        self.logger.info("Strategy: S3 cache + Parallel SEC fetches (10 req/sec) + Batched FinBERT (GPU)")
        self.logger.info("All filings will be processed (no limit). Progress is checkpointed.")

        # Queue for prefetched filing texts
        prefetch_queue: Queue = Queue(maxsize=100)  # Larger queue for batching
        fetch_done = threading.Event()

        # Track cache stats
        cache_hits = 0
        cache_misses = 0
        cache_lock = threading.Lock()

        def fetch_worker(sym: str, cik: str) -> Tuple[str, str, Any, str]:
            """Fetch filing texts for a symbol (cache first, then SEC)."""
            nonlocal cache_hits, cache_misses
            import time as _time

            try:
                # Check S3 cache first
                if mda_cache:
                    cached = mda_cache.get(cik)
                    if cached and cached.get('filings'):
                        # Convert cached dicts back to FilingText objects
                        filing_texts = [
                            FilingText(
                                cik=f['cik'],
                                accession_number=f['accession_number'],
                                filing_date=f['filing_date'],
                                filing_type=f['filing_type'],
                                section=f['section'],
                                text=f['text'],
                                fiscal_year=f.get('fiscal_year'),
                                fiscal_quarter=f.get('fiscal_quarter')
                            )
                            for f in cached['filings']
                        ]
                        with cache_lock:
                            cache_hits += 1
                        return (sym, cik, filing_texts, 'ok')

                # Cache miss - fetch from SEC
                t0 = _time.time()
                filings = collector.get_filings_metadata(
                    cik=cik,
                    start_date=start_date,
                    end_date=end_date
                )
                t1 = _time.time()
                if not filings:
                    return (sym, cik, None, 'no_filings')

                # Process all filings (no limit)
                filing_texts = collector.collect_filing_texts(
                    cik=cik,
                    filings=filings
                )
                t2 = _time.time()
                self.logger.debug(f"{sym}: metadata={t1-t0:.1f}s, texts={t2-t1:.1f}s ({len(filings)} filings)")
                if not filing_texts:
                    return (sym, cik, None, 'no_mda')

                # Cache for future runs
                if mda_cache:
                    try:
                        mda_cache.put(cik, filing_texts)
                    except Exception as e:
                        pass  # Don't fail on cache write errors

                with cache_lock:
                    cache_misses += 1

                return (sym, cik, filing_texts, 'ok')
            except Exception as e:
                return (sym, cik, None, f'error: {e}')

        def producer_thread():
            """Producer: fetch SEC filings in parallel."""
            # More workers since each fetch blocks on rate limiter
            with ThreadPoolExecutor(max_workers=30) as executor:
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

        # Batching parameters
        BATCH_CHUNK_TARGET = 512  # Target chunks per inference batch

        # Batch accumulator: stores (sym, cik, filing_texts, chunk_indices)
        batch_buffer: List[Tuple[str, str, List, List[Tuple[int, int]]]] = []
        all_chunks: List[str] = []  # Flattened chunks for batch inference

        pbar = tqdm(total=len(symbols_to_process), desc="FinBERT", unit="sym")
        symbols_in_batch = 0

        def process_batch():
            """Run batched inference and upload results."""
            nonlocal all_chunks, batch_buffer, symbols_in_batch

            if not all_chunks:
                return

            pbar.set_description(f"FinBERT [batch {len(batch_buffer)} syms, {len(all_chunks)} chunks]")

            # Run single batched inference
            try:
                all_results = model.predict(all_chunks)
            except Exception as e:
                self.logger.warning(f"Batch inference failed: {e}")
                self.failed += len(batch_buffer)
                all_chunks = []
                batch_buffer = []
                return

            # Map results back to symbols
            for sym, cik, filing_texts, chunk_map in batch_buffer:
                try:
                    # Extract results for this symbol's filings
                    from quantdl.derived.sentiment import compute_sentiment_long, FilingSentiment

                    sentiments = []
                    for filing_idx, (start_idx, end_idx) in enumerate(chunk_map):
                        filing_results = all_results[start_idx:end_idx]
                        if not filing_results:
                            continue

                        sentiment = _aggregate_sentiment_results(
                            filing_text=filing_texts[filing_idx],
                            results=filing_results,
                            model_name=model.name,
                            model_version=model.version
                        )
                        if sentiment:
                            sentiments.append(sentiment)

                    if not sentiments:
                        self.skipped += 1
                        continue

                    sentiment_df = compute_sentiment_long(
                        filing_sentiments=sentiments,
                        symbol=sym,
                        logger=self.logger
                    )

                    if len(sentiment_df) == 0:
                        self.skipped += 1
                        continue

                    self._upload_sentiment(cik, sentiment_df)
                    self.success += 1

                    # Mark as checkpointed
                    if checkpoint:
                        checkpoint.mark_processed(cik)

                except Exception as e:
                    self.logger.debug(f"Failed to process {sym}: {e}")
                    self.failed += 1

            pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped, cache=cache_hits, ckpt=checkpoint.count() if checkpoint else 0)

            # Clear batch
            all_chunks = []
            batch_buffer = []

        # Consumer: accumulate and batch process
        while True:
            if fetch_done.is_set() and prefetch_queue.empty():
                break

            try:
                result = prefetch_queue.get(timeout=0.5)
            except Empty:
                # If we have accumulated chunks, process them
                if all_chunks and len(all_chunks) >= BATCH_CHUNK_TARGET // 2:
                    process_batch()
                continue

            sym, cik, filing_texts, status = result
            pbar.update(1)

            if status != 'ok':
                if status in ('no_filings', 'no_mda'):
                    self.skipped += 1
                else:
                    self.logger.debug(f"Failed to fetch {sym}: {status}")
                    self.failed += 1
                pbar.set_postfix(ok=self.success, fail=self.failed, skip=self.skipped, cache=cache_hits)
                continue

            # Chunk all filings for this symbol and add to batch
            chunk_map = []  # Maps filing index -> (start_chunk_idx, end_chunk_idx)
            for filing_text in filing_texts:
                if not filing_text.text:
                    chunk_map.append((len(all_chunks), len(all_chunks)))
                    continue

                start_idx = len(all_chunks)
                chunks = chunk_text(filing_text.text, chunk_size=1500)
                all_chunks.extend(chunks)
                end_idx = len(all_chunks)
                chunk_map.append((start_idx, end_idx))

            batch_buffer.append((sym, cik, filing_texts, chunk_map))
            symbols_in_batch += 1

            # Process batch when we have enough chunks
            if len(all_chunks) >= BATCH_CHUNK_TARGET:
                process_batch()

        # Process remaining batch
        if all_chunks:
            process_batch()

        pbar.close()

        # Log cache stats
        if mda_cache:
            self.logger.info(
                f"Cache stats: {cache_hits} hits, {cache_misses} misses "
                f"({cache_hits / max(1, cache_hits + cache_misses) * 100:.1f}% hit rate)"
            )

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
