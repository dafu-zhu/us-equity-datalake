import os
from typing import Dict, Any
from dotenv import load_dotenv
import boto3
from botocore.config import Config
from storage.config_loader import UploadConfig

load_dotenv()

class S3Client:
    """
    Load config with config_path
    Returns a S3 client instance by calling 'create' method
    """
    def __init__(self, config_path: str="configs/storage.yaml"):
        self.config = UploadConfig(config_path)
        self.boto_config = self._create_boto_config()

        # Load key and secrets from .env
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    def _create_boto_config(self) -> Config:
        """
        Create boto3 Config object from loaded configuration
        """
        client_cfg = self.config.client

        if not client_cfg:
            raise ValueError("Client configuration is empty or missing")

        # Build with required params
        config_kwargs: Dict[str, Any] = {
            'region_name': client_cfg.get('region_name', 'us-east-2'),
            'max_pool_connections': client_cfg.get('max_pool_connections', 50),
        }

        # Add optional params
        if 'connect_timeout' in client_cfg:
            config_kwargs['connect_timeout'] = client_cfg['connect_timeout']

        if 'read_timeout' in client_cfg:
            config_kwargs['read_timeout'] = client_cfg['read_timeout']

        if 'retries' in client_cfg:
            retries_cfg = client_cfg['retries']
            if isinstance(retries_cfg, dict):
                config_kwargs['retries'] = {
                    'mode': retries_cfg.get('mode', 'standard'),
                    'total_max_attempts': retries_cfg.get('total_max_attempts', 5)
                }

        if 's3' in client_cfg:
            s3_cfg = client_cfg['s3']
            if isinstance(s3_cfg, dict):
                config_kwargs['s3'] = {}

                if 'addressing_style' in s3_cfg:
                    config_kwargs['s3']['addressing_style'] = s3_cfg['addressing_style']

                if 'payload_signing_enabled' in s3_cfg:
                    config_kwargs['s3']['payload_signing_enabled'] = s3_cfg['payload_signing_enabled']

                if 'us_east_1_regional_endpoint' in s3_cfg:
                    config_kwargs['s3']['us_east_1_regional_endpoint'] = s3_cfg['us_east_1_regional_endpoint']

        if 'tcp_keepalive' in client_cfg:
            config_kwargs['tcp_keepalive'] = client_cfg['tcp_keepalive']

        if 'request_min_compression_size_bytes' in client_cfg:
            config_kwargs['request_min_compression_size_bytes'] = client_cfg['request_min_compression_size_bytes']

        if 'disable_request_compression' in client_cfg:
            config_kwargs['disable_request_compression'] = client_cfg['disable_request_compression']

        if 'request_checksum_calculation' in client_cfg:
            config_kwargs['request_checksum_calculation'] = client_cfg['request_checksum_calculation']

        if 'response_checksum_validation' in client_cfg:
            config_kwargs['response_checksum_validation'] = client_cfg['response_checksum_validation']

        return Config(**config_kwargs)
    
    @property
    def client(self):
        return boto3.client(
            's3', 
            config=self.boto_config,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key 
        )