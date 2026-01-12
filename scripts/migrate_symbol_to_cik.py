"""
Migration script: Symbol-based to CIK-based fundamental data storage

Migrates fundamental data from symbol-based paths to CIK-based paths:
- data/raw/fundamental/{symbol}/ → data/raw/fundamental/{cik}/
- data/derived/features/fundamental/{symbol}/ttm.parquet → .../{cik}/ttm.parquet
- data/derived/features/fundamental/{symbol}/metrics.parquet → .../{cik}/metrics.parquet

Usage:
    python scripts/migrate_symbol_to_cik.py --dry-run  # Preview migration
    python scripts/migrate_symbol_to_cik.py             # Execute migration
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
from dotenv import load_dotenv

from quantdl.storage.s3_client import S3Client
from quantdl.utils.mapping import symbol_cik_mapping
from quantdl.utils.logger import setup_logger

load_dotenv()


class FundamentalMigration:
    """Migrates fundamental data from symbol-based to CIK-based storage"""

    def __init__(self, bucket_name: str, dry_run: bool = False):
        self.bucket_name = bucket_name
        self.dry_run = dry_run
        self.s3_client = S3Client().client

        # Setup logging
        log_dir = Path("data/logs/migration")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name='migration',
            log_dir=log_dir,
            level=logging.INFO
        )

        # Load CIK mapping from SEC (simpler, no WRDS needed)
        self.logger.info("Loading symbol-to-CIK mapping from SEC")
        self.symbol_to_cik_map = self._load_cik_mapping()
        self.logger.info(f"Loaded {len(self.symbol_to_cik_map)} symbol-CIK mappings")

        # Migration statistics
        self.stats = {
            'total_symbols': 0,
            'migrated': 0,
            'null_cik': 0,
            'failed': 0,
            'skipped': 0
        }
        self.null_cik_symbols = []
        self.failed_migrations = []

        self.logger.info("Migration instance initialized successfully")

    def _load_cik_mapping(self) -> Dict[str, str]:
        """Load symbol-CIK mapping from SEC with proper headers and timeout"""
        import requests
        import os

        url = "https://www.sec.gov/files/company_tickers.json"

        # SEC requires User-Agent with identifiable name/email
        # Use env var if available, otherwise use a default
        user_email = os.getenv('SEC_USER_AGENT_EMAIL', 'user@example.com')
        headers = {
            'User-Agent': f'US-Equity-Datalake Migration/1.0 ({user_email})'
        }

        try:
            self.logger.info(f"Fetching CIK mapping from {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            self.logger.info("Parsing CIK mapping JSON")
            symbol_cik = {}
            data = response.json()
            for key, value in data.items():
                symbol = value.get('ticker')
                cik_str = value.get('cik_str')
                if symbol and cik_str:
                    # Store with zero-padded CIK (10 digits)
                    symbol_cik[symbol] = str(cik_str).zfill(10)

            self.logger.info(f"Successfully loaded {len(symbol_cik)} symbol-CIK mappings")
            return symbol_cik

        except requests.Timeout:
            self.logger.error("Timeout fetching CIK mapping from SEC")
            raise
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch CIK mapping: {e}")
            raise

    def list_symbol_based_files(self, prefix: str) -> List[Tuple[str, str]]:
        """
        List all symbol-based files under a prefix

        Returns: List of (symbol, s3_key) tuples
        """
        self.logger.info(f"Listing files under {prefix}")
        symbol_files = []
        continuation_token = None

        while True:
            params = {
                'Bucket': self.bucket_name,
                'Prefix': prefix,
                'Delimiter': '/'
            }
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            response = self.s3_client.list_objects_v2(**params)

            # Extract symbol directories from common prefixes
            if 'CommonPrefixes' in response:
                for prefix_obj in response['CommonPrefixes']:
                    symbol_prefix = prefix_obj['Prefix']
                    # Extract symbol from prefix (e.g., 'data/raw/fundamental/AAPL/' → 'AAPL')
                    symbol = symbol_prefix.rstrip('/').split('/')[-1]

                    # List files under this symbol directory
                    file_response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=symbol_prefix
                    )

                    if 'Contents' in file_response:
                        for obj in file_response['Contents']:
                            key = obj['Key']
                            if key.endswith('.parquet'):
                                symbol_files.append((symbol, key))

            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break

        self.logger.info(f"Found {len(symbol_files)} files")
        return symbol_files

    def resolve_symbols_to_ciks(self, symbols: List[str]) -> Dict[str, str]:
        """
        Resolve symbols to CIKs using SEC mapping

        Returns: Dict[symbol -> cik] (only symbols with valid CIKs)
        """
        self.logger.info(f"Resolving {len(symbols)} symbols to CIKs")

        # Normalize symbols and look up in mapping
        valid_ciks = {}
        null_ciks = []

        for sym in symbols:
            # Try original symbol first
            cik = self.symbol_to_cik_map.get(sym)

            # Try uppercase if not found
            if cik is None:
                cik = self.symbol_to_cik_map.get(sym.upper())

            # Try without periods/hyphens (e.g., BRK.B -> BRKB)
            if cik is None:
                normalized = sym.replace('.', '').replace('-', '').upper()
                cik = self.symbol_to_cik_map.get(normalized)

            if cik is not None:
                valid_ciks[sym] = cik
            else:
                null_ciks.append(sym)

        self.logger.info(f"Resolved {len(valid_ciks)} symbols, {len(null_ciks)} have NULL CIKs")
        self.null_cik_symbols.extend(null_ciks)

        return valid_ciks

    def migrate_file(self, symbol: str, source_key: str, cik: str) -> bool:
        """
        Migrate a single file from symbol-based to CIK-based path

        Returns: True if successful, False otherwise
        """
        # Determine destination key
        if 'data/raw/fundamental/' in source_key:
            dest_key = source_key.replace(f'data/raw/fundamental/{symbol}/', f'data/raw/fundamental/{cik}/')
        elif 'data/derived/features/fundamental/' in source_key and '/ttm.parquet' in source_key:
            dest_key = source_key.replace(f'data/derived/features/fundamental/{symbol}/', f'data/derived/features/fundamental/{cik}/')
        elif 'data/derived/features/fundamental/' in source_key and '/metrics.parquet' in source_key:
            dest_key = source_key.replace(f'data/derived/features/fundamental/{symbol}/', f'data/derived/features/fundamental/{cik}/')
        else:
            self.logger.warning(f"Unknown file type: {source_key}")
            return False

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would copy {source_key} → {dest_key}")
            return True

        try:
            # Copy object with metadata
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}

            # Get original metadata
            head_response = self.s3_client.head_object(Bucket=self.bucket_name, Key=source_key)
            original_metadata = head_response.get('Metadata', {})

            # Update metadata with CIK
            new_metadata = original_metadata.copy()
            new_metadata['cik'] = cik
            new_metadata['migrated_from_symbol'] = symbol
            new_metadata['migration_timestamp'] = datetime.now().isoformat()

            # Copy object
            self.s3_client.copy_object(
                Bucket=self.bucket_name,
                CopySource=copy_source,
                Key=dest_key,
                Metadata=new_metadata,
                MetadataDirective='REPLACE'
            )

            self.logger.info(f"Migrated: {source_key} → {dest_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to migrate {source_key}: {e}")
            self.failed_migrations.append((symbol, source_key, str(e)))
            return False

    def migrate_data_type(self, prefix: str, data_type_name: str):
        """Migrate all files under a specific prefix"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Migrating {data_type_name}")
        self.logger.info(f"{'='*60}")

        # List all symbol-based files
        symbol_files = self.list_symbol_based_files(prefix)

        if not symbol_files:
            self.logger.info(f"No files found under {prefix}")
            return

        # Extract unique symbols
        symbols = list(set([sym for sym, _ in symbol_files]))
        self.stats['total_symbols'] += len(symbols)

        # Resolve symbols to CIKs
        cik_map = self.resolve_symbols_to_ciks(symbols)
        self.stats['null_cik'] += len(symbols) - len(cik_map)

        # Migrate each file
        for symbol, source_key in symbol_files:
            cik = cik_map.get(symbol)

            if cik is None:
                self.logger.warning(f"Skipping {symbol} (NULL CIK): {source_key}")
                self.stats['skipped'] += 1
                continue

            success = self.migrate_file(symbol, source_key, cik)
            if success:
                self.stats['migrated'] += 1
            else:
                self.stats['failed'] += 1

    def run_migration(self):
        """Execute the full migration"""
        start_time = datetime.now()

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting Fundamental Data Migration")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        self.logger.info(f"Bucket: {self.bucket_name}")
        self.logger.info(f"{'='*60}\n")

        # Migrate each data type
        self.migrate_data_type('data/raw/fundamental/', 'Raw Fundamental')
        self.migrate_data_type('data/derived/features/fundamental/', 'TTM & Derived Metrics')

        # Generate report
        self.generate_report(start_time)

    def generate_report(self, start_time: datetime):
        """Generate migration report"""
        duration = (datetime.now() - start_time).total_seconds()

        report = {
            'migration_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'dry_run': self.dry_run,
            'bucket': self.bucket_name,
            'statistics': self.stats,
            'null_cik_symbols': self.null_cik_symbols[:50],  # First 50
            'null_cik_total': len(self.null_cik_symbols),
            'failed_migrations': self.failed_migrations[:50],  # First 50
            'failed_total': len(self.failed_migrations)
        }

        # Save report to file
        report_path = Path(f"data/logs/migration/migration_report_{'dry_run_' if self.dry_run else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("MIGRATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total symbols: {self.stats['total_symbols']}")
        self.logger.info(f"Files migrated: {self.stats['migrated']}")
        self.logger.info(f"NULL CIK symbols: {self.stats['null_cik']} ({100*self.stats['null_cik']/max(self.stats['total_symbols'],1):.1f}%)")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Report saved: {report_path}")
        self.logger.info(f"{'='*60}\n")

        if self.stats['null_cik'] > 0:
            self.logger.warning(f"NULL CIK symbols (first 10): {self.null_cik_symbols[:10]}")

        if self.stats['failed'] > 0:
            self.logger.error(f"Failed migrations (first 5): {self.failed_migrations[:5]}")


def main():
    parser = argparse.ArgumentParser(description='Migrate fundamental data from symbol to CIK-based storage')
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without making changes')
    parser.add_argument('--bucket', type=str, default='us-equity-datalake', help='S3 bucket name')

    args = parser.parse_args()

    migrator = FundamentalMigration(bucket_name=args.bucket, dry_run=args.dry_run)
    migrator.run_migration()


if __name__ == '__main__':
    main()
