"""
AWS S3 Client Configuration for US Equity Data Lake

This module provides a configured S3 client with optimized settings for
handling large-scale data storage and retrieval operations.

AWS credentials are loaded from:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. AWS credentials file (~/.aws/credentials)
3. IAM role (if running on EC2)
"""

import boto3
from botocore.config import Config
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class S3Client:
    """
    Configured S3 client for US equity data lake operations.

    This client is optimized for:
    - High-throughput data transfers
    - Parallel file operations
    - Reliable retry behavior
    - S3-specific optimizations for data lake workloads
    """

    def __init__(
        self,
        region_name: str = "us-east-2",
        max_pool_connections: int = 50,
        connect_timeout: int = 60,
        read_timeout: int = 60,
    ):
        """
        Initialize S3 client with custom configuration.

        :param region_name: AWS region
        :param max_pool_connections: Maximum number of connections in the pool
        :param connect_timeout: Connection timeout in seconds
        :param read_timeout: Read timeout in seconds
        """

        # Create botocore config object with optimized settings
        self.config = Config(
            # Region configuration
            region_name=region_name,

            # Connection settings - increased for parallel operations
            max_pool_connections=max_pool_connections,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,

            # Retry configuration - standard mode with higher attempts for reliability
            retries={
                'mode': 'standard',  # Use AWS standard retry mode
                'total_max_attempts': 5  # Retry up to 5 times total
            },

            # S3-specific configuration
            s3={
                # Use virtual hosted-style addressing (bucket.s3.amazonaws.com)
                'addressing_style': 'virtual',

                # Enable payload signing for security
                'payload_signing_enabled': True,

                # Use regional endpoint for us-east-1 (better performance)
                'us_east_1_regional_endpoint': 'regional'
            },

            # Enable TCP keepalive for long-running connections
            tcp_keepalive=True,

            # Request compression settings (compress large payloads)
            request_min_compression_size_bytes=10240,  # 10KB minimum
            disable_request_compression=False,

            # Checksum validation for data integrity
            request_checksum_calculation='when_supported',
            response_checksum_validation='when_supported',
        )

        # Initialize S3 client and resource
        self.client = boto3.client('s3', config=self.config)
        self.resource = boto3.resource('s3', config=self.config)

        logger.info(f"S3 client initialized with region: {region_name}")

    def get_client(self):
        """Get the configured boto3 S3 client."""
        return self.client

    def get_resource(self):
        """Get the configured boto3 S3 resource."""
        return self.resource

    def get_bucket(self, bucket_name: str):
        """
        Get a bucket object.

        Args:
            bucket_name: Name of the S3 bucket

        Returns:
            Boto3 bucket resource
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

        Args:
            file_path: Local file path
            bucket_name: S3 bucket name
            object_key: S3 object key (destination path)
            extra_args: Additional upload arguments (e.g., metadata, ACL)
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

        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key (source path)
            file_path: Local file path (destination)
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

        Args:
            bucket_name: S3 bucket name
            prefix: Object key prefix to filter by
            max_keys: Maximum number of keys to return

        Returns:
            List of object keys
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

        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except Exception:
            return False


def create_s3_client(
    region_name: str = "us-east-1",
    **kwargs
) -> S3Client:
    """
    Factory function to create a configured S3 client.

    Args:
        region_name: AWS region
        **kwargs: Additional configuration parameters

    Returns:
        Configured S3Client instance
    """
    return S3Client(region_name=region_name, **kwargs)


if __name__ == "__main__":
    # Example usage
    # Create S3 client
    s3_client = create_s3_client()

    # Example: List buckets
    try:
        buckets = s3_client.get_client().list_buckets()
        print("Available S3 buckets:")
        for bucket in buckets['Buckets']:
            print(f"  - {bucket['Name']}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have AWS credentials configured:")
        print("1. Set environment variables:")
        print("   export AWS_ACCESS_KEY_ID='your-access-key'")
        print("   export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("\n2. Or configure AWS CLI:")
        print("   aws configure")
