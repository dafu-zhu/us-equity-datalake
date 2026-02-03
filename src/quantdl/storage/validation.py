import datetime as dt
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from quantdl.storage.s3_client import S3Client
from typing import Optional
from quantdl.utils.logger import setup_logger

load_dotenv()


class Validator:
    def __init__(self, s3_client=None, bucket_name: Optional[str] = None):
        self.s3_client = s3_client or S3Client().client
        # Allow bucket_name to be passed as parameter or from environment variable
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'us-equity-datalake')
        self.log_dir = Path("data/logs/validation")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name='validation',
            log_dir=self.log_dir,
            level=logging.WARNING
        )

    def list_files_under_prefix(self, prefix: str) -> list[str]:
        """
        List all files (object keys) under a given prefix.

        :param prefix: Prefix/directory to list (e.g., 'data/raw/fundamental')
        :return: List of full object keys under the prefix
        """
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')

        if storage_backend == 'local':
            return self._list_local_files(prefix, local_path)
        else:
            return self._list_s3_files(prefix)

    def _list_local_files(self, prefix: str, local_path: str) -> list[str]:
        """List all files under a local directory prefix."""
        files = []
        local_dir = Path(local_path) / prefix
        if local_dir.exists():
            for file_path in local_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Convert to relative path (like S3 key)
                    rel_path = file_path.relative_to(Path(local_path))
                    files.append(str(rel_path).replace('\\', '/'))
        return files

    def _list_s3_files(self, prefix: str) -> list[str]:
        """List all files under an S3 prefix."""
        files = []
        continuation_token = None

        while True:
            # Build request params
            params = {
                'Bucket': self.bucket_name,
                'Prefix': prefix
            }
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            # List objects
            response = self.s3_client.list_objects_v2(**params)

            # Collect object keys
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])

            # Check for more pages
            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break

        return files

    def data_exists(
            self,
            symbol,
            data_type: str,
            year: Optional[int]=None,
            month: Optional[int]=None,
            day: Optional[str]=None,
            data_tier: str = "raw",
            cik: Optional[str] = None,
            security_id: Optional[int] = None
        ) -> bool:
        """
        For both daily and minute data, check if data exists

        :param symbol: Stock to inspect (used for ticks and as fallback for fundamental)
        :param data_type: "ticks", "fundamental", or "ttm"
        :param year: Daily data only, specify year
        :param month: Daily data only, specify month (1-12) for monthly partitions
        :param day: Minute data only, specify trade day. Format: YYYY-MM-DD
        :param data_tier: "raw" or "derived" (default: "raw")
        :param cik: CIK string for fundamental data (zero-padded to 10 digits). If provided, uses CIK-based paths for fundamental/ttm/derived data
        :param security_id: Security ID for ticks data (if None, falls back to symbol-based paths for backward compatibility)
        """
        if year and day:
            raise ValueError(f'Specify year OR day, not both')

        if data_tier not in {"raw", "derived"}:
            raise ValueError(f'Expected data_tier is raw or derived, get {data_tier} instead')

        base_prefix = f"data/{data_tier}"

        # Define Key based on year, month, day and data_type
        if data_type == 'ticks':
            identifier = str(security_id) if security_id is not None else symbol
            if year and month:
                # Monthly partition
                s3_key = f'{base_prefix}/{data_type}/daily/{identifier}/{year}/{month:02d}/{data_type}.parquet'
            elif year:
                # History file (for security_id) or yearly partition (legacy for symbol)
                if security_id is not None:
                    s3_key = f'{base_prefix}/{data_type}/daily/{identifier}/history.parquet'
                else:
                    s3_key = f'{base_prefix}/{data_type}/daily/{identifier}/{year}/{data_type}.parquet'
            elif day:
                date = dt.datetime.strptime(day, '%Y-%m-%d').date()
                year_str = date.strftime('%Y')
                month_str = date.strftime('%m')
                day_str = date.strftime('%d')
                s3_key = f'{base_prefix}/{data_type}/minute/{identifier}/{year_str}/{month_str}/{day_str}/{data_type}.parquet'
            else:
                raise ValueError(
                    "Must provide either year or day parameter "
                    f"(month optional). Got year={year}, month={month}, day={day}"
                )
        elif data_type == 'fundamental':
            identifier = cik if cik else symbol
            if data_tier == "derived":
                s3_key = f'data/derived/features/fundamental/{identifier}/metrics.parquet'
            else:
                s3_key = f'{base_prefix}/{data_type}/{identifier}/{data_type}.parquet'
        elif data_type == 'ttm':
            identifier = cik if cik else symbol
            s3_key = f'data/derived/features/fundamental/{identifier}/ttm.parquet'
        else:
            raise ValueError(f'Expected data_type is ticks, fundamental, or ttm, get {data_type} instead')
        
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')

        if storage_backend == 'local':
            local_file = Path(local_path) / s3_key
            return local_file.exists()
        else:
            try:
                self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                return True
            except ClientError as error:
                if error.response.get('Error', {}).get('Code') == '404':
                    return False
                else:
                    self.logger.error(f'Error checking {s3_key}: {error}')
                    return False
    
    def get_existing_minute_days(
            self,
            security_id: int,
            year: int,
            month: int,
            data_tier: str = "raw"
        ) -> set[str]:
        """
        List all existing minute tick days for a symbol/month.
        Returns set of day strings (DD format) that exist.

        :param security_id: Security ID for ticks data
        :param year: Year (YYYY)
        :param month: Month (1-12)
        :param data_tier: "raw" or "derived" (default: "raw")
        :return: Set of existing day strings (e.g., {'01', '02', '15'})
        """
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')
        prefix = f"data/{data_tier}/ticks/minute/{security_id}/{year}/{month:02d}/"
        existing_days = set()

        if storage_backend == 'local':
            local_dir = Path(local_path) / prefix
            if local_dir.exists():
                for day_dir in local_dir.iterdir():
                    if day_dir.is_dir():
                        existing_days.add(day_dir.name)
        else:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    Delimiter='/'
                )
                # CommonPrefixes contains subdirectories (day folders)
                for prefix_info in response.get('CommonPrefixes', []):
                    # prefix_info['Prefix'] = 'data/raw/ticks/minute/12345/2024/01/15/'
                    day = prefix_info['Prefix'].rstrip('/').split('/')[-1]
                    existing_days.add(day)
            except Exception as e:
                self.logger.error(f'Error listing {prefix}: {e}')

        return existing_days

    def top_3000_exists(self, year: int, month: int) -> bool:
        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')
        key = f"data/symbols/{year}/{month:02d}/top3000.txt"

        if storage_backend == 'local':
            local_file = Path(local_path) / key
            return local_file.exists()
        else:
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError as error:
                if error.response.get('Error', {}).get('Code') == '404':
                    return False
                self.logger.error(f"Error checking {key}: {error}")
                return False
        