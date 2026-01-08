import datetime as dt
from pathlib import Path
import logging
from botocore.exceptions import ClientError
from storage.s3_client import S3Client
from typing import Optional
from utils.logger import setup_logger


class Validator:
    def __init__(self, s3_client=None):
        self.s3_client = s3_client or S3Client().client
        self.bucket_name = "us-equity-datalake"
        self.log_dir = Path("data/logs/validation")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name='validation',
            log_dir=self.log_dir,
            level=logging.WARNING
        )

    def list_files_under_prefix(self, prefix: str) -> list[str]:
        """
        List all files (object keys) under a given S3 prefix

        :param prefix: S3 prefix/directory to list (e.g., 'data/raw/fundamental')
        :return: List of full S3 object keys under the prefix
        """
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

    def list_available_years(self, symbol: str, data_type: str) -> list:
        """
        For daily data, List all years we have for a symbol

        :param symbol: Stock to inspect
        :param data_type: "ticks", "fundamental", or "ttm"
        """
        if data_type == "ticks":
            prefix = f'data/raw/{data_type}/daily/{symbol}/'
        elif data_type == "fundamental":
            prefix = f'data/raw/{data_type}/{symbol}/'
        elif data_type == "ttm":
            prefix = f'data/derived/features/fundamental/{symbol}/'
        else:
            raise ValueError(
                f'Expected data_type is ticks, fundamental, or ttm, get {data_type} instead'
            )

        years = []
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

            # Extract years from object keys
            if 'Contents' in response:
                for obj in response['Contents']:
                    parts = obj['Key'].split('/')
                    if data_type == "fundamental":
                        continue
                    if len(parts) >= 6 and data_type in parts:
                        year = int(parts[-2])
                        if year not in years:
                            years.append(year)

            # Check for more pages
            if response.get('IsTruncated'):
                continuation_token = response['NextContinuationToken']
            else:
                break

        return sorted(years, reverse=True)

    def data_exists(
            self, 
            symbol, 
            data_type: str, 
            year: Optional[int]=None,  
            day: Optional[str]=None,
            data_tier: str = "raw"
        ) -> bool:
        """
        For both daily and minute data, check if data exists

        :param symbol: Stock to inspect
        :param data_type: "ticks", "fundamental", or "ttm"
        :param year: Daily data only, specify year
        :param month: Daily data only, use YYYY/MM to locate
        :param day: Minute data only, specify trade day. Format: YYYY-MM-DD
        :param data_tier: "raw" or "derived" (default: "raw")
        """
        if year and day:
            raise ValueError(f'Specify year OR day, not both')

        if data_tier not in {"raw", "derived"}:
            raise ValueError(f'Expected data_tier is raw or derived, get {data_tier} instead')

        base_prefix = f"data/{data_tier}"

        # Define Key based on year, day and data_type
        if data_type == 'ticks':
            if year:
                s3_key = f'{base_prefix}/{data_type}/daily/{symbol}/{year}/{data_type}.parquet'
            elif day:
                date = dt.datetime.strptime(day, '%Y-%m-%d').date()
                year_str = date.strftime('%Y')
                month_str = date.strftime('%m')
                day_str = date.strftime('%d')
                s3_key = f'{base_prefix}/{data_type}/minute/{symbol}/{year_str}/{month_str}/{day_str}/{data_type}.parquet'
            else:
                raise ValueError(f'Must provide either year or day parameter. Got year={year}, day={day}')
        elif data_type == 'fundamental':
            if data_tier == "derived":
                s3_key = f'data/derived/features/fundamental/{symbol}/metrics.parquet'
            else:
                s3_key = f'{base_prefix}/{data_type}/{symbol}/{data_type}.parquet'
        elif data_type == 'ttm':
            s3_key = f'data/derived/features/fundamental/{symbol}/ttm.parquet'
        else:
            raise ValueError(f'Expected data_type is ticks, fundamental, or ttm, get {data_type} instead')
        
        try:
            self.s3_client.head_object(
                Bucket = self.bucket_name,
                Key = s3_key
            )
            return True
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') == '404':
                return False
            else:
                self.logger.error(f'Error checking {s3_key}: {error}')
                return False
    
    def top_3000_exists(self, year: int, month: int) -> bool:
        s3_key = f"data/symbols/{year}/{month:02d}/top3000.txt"
        try:
            self.s3_client.head_object(Bucket="us-equity-datalake", Key=s3_key)
            return True
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') == '404':
                return False
            self.logger.error(f"Error checking {s3_key}: {error}")
            return False
        

if __name__ == '__main__':
    import time
    vld = Validator()
    start = time.perf_counter()
    years = vld.list_available_years('AAPL', data_type='ticks')
    print(time.perf_counter() - start)

    start = time.perf_counter()
    vld.data_exists('AAPL', 'ticks', 2024)
    print(time.perf_counter() - start)

    # Expected output
    # 0.229747585000041
    # 0.043064168999990216

    files = vld.list_files_under_prefix('data/raw/fundamental')
    print(len(files))
