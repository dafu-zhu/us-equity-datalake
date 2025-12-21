"""
AWS S3 Client Configuration for US Equity Data Lake

This module provides a configured S3 client with optimized settings for
handling large-scale data storage and retrieval operations.

AWS credentials are loaded from:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. AWS credentials file (~/.aws/credentials)
3. IAM role (if running on EC2)
"""
import os
from typing import Optional
import logging

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

logger = logging.getLogger(__name__)

class S3Config:
    """
    Create a singleton configuration for S3
    """

    _instance = None
    _boto_config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_config()
        return cls._instance
    
    def init_config(self):
        """Initialize AWS S3 config instance"""
        self._boto_config = Config(
            region_name="us-east-2",
            max_pool_connections=50,
            connect_timeout=60,
            read_timeout=60,

            # Retry configuration - standard mode with higher attempts for reliability
            retries={
                'mode': 'standard',  # Use AWS standard retry mode
                'total_max_attempts': 5  # Retry up to 5 times total
            },

            # S3-specific configuration
            s3={
                # Use virtual hosted-style addressing (bucket.s3.amazonaws.com)
                'addressing_style': 'virtual',
                'payload_signing_enabled': True,
                'us_east_1_regional_endpoint': 'regional'
            },

            # Enable TCP keepalive for long-running connections
            tcp_keepalive=True,

            # Checksum validation for data integrity
            request_checksum_calculation='when_supported',
            response_checksum_validation='when_supported',
        )
    
    @property
    def boto_config(self):
        return self._boto_config


class S3Client:
    """
    Configured S3 client for US equity data lake operations.

    This client is optimized for:
    - High-throughput data transfers
    - Parallel file operations
    - Reliable retry behavior
    - S3-specific optimizations for data lake workloads
    """

    def __init__(self):
        """
        Initialize S3 client with custom configuration.
        """
        cfg = S3Config()
        self.config = cfg.boto_config

        # Initialize S3 client and resource
        self.client = boto3.client(
            service_name = 's3',
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            config = self.config
        )
        self.resource = boto3.resource(
            service_name = 's3',
            aws_access_key_id = AWS_ACCESS_KEY_ID,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
            config = self.config
        )

    def get_client(self):
        """Get the configured boto3 S3 client."""
        return self.client

    def get_resource(self):
        """Get the configured boto3 S3 resource."""
        return self.resource

    def get_bucket(self, bucket_name: str):
        """
        Get a bucket object.

        :param bucket_name: Name of the S3 bucket
        :return: Boto3 bucket resource
        """
        return self.resource.Bucket(bucket_name)

    def upload_file(
        self,
        file_path: str,
        bucket_name: str,
        object_key: str,
        extra_args: Optional[dict] = None
    ):
        """
        Upload a file to S3 with progress tracking.

        :param file_path: Local file path
        :param bucket_name: S3 bucket name
        :param object_key: S3 object key (destination path)
        :param extra_args: Additional upload arguments (e.g., metadata, ACL)
        """
        try:
            self.client.upload_file(
                Filename=file_path,
                Bucket=bucket_name,
                Key=object_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{object_key}")
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {e}")
            raise

    def download_file(
        self,
        bucket_name: str,
        object_key: str,
        file_path: str
    ):
        """
        Download a file from S3.

        :param bucket_name: S3 bucket name
        :param object_key: S3 object key (source path)
        :param file_path: Local file path (destination)
        """
        try:
            self.client.download_file(
                Bucket=bucket_name,
                Key=object_key,
                Filename=file_path
            )
            logger.info(f"Downloaded s3://{bucket_name}/{object_key} to {file_path}")
        except Exception as e:
            logger.error(f"Error downloading {object_key}: {e}")
            raise

    def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        max_keys: int = 1000
    ):
        """
        List objects in S3 bucket with a given prefix.

        :param bucket_name: S3 bucket name
        :param prefix: Object key prefix to filter by
        :param max_keys: Maximum number of keys to return

        :return: List of object keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Error listing objects with prefix {prefix}: {e}")
            raise

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        """
        Check if an object exists in S3.

        :param bucket_name: S3 bucket name
        :param object_key: S3 object key

        :return: True if object exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Example usage
    # Create S3 client
    s3_client = S3Client()

    # Example: List buckets
    buckets = s3_client.get_client().list_buckets()
    print("Available S3 buckets:")
    for bucket in buckets['Buckets']:
        print(f"  - {bucket['Name']}")
