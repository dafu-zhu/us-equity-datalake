from pathlib import Path
from typing import Optional
import yaml
import boto3
from botocore.config import Config


class S3Client:
    def __init__(self, config_path: Optional[str]=None):
        self.config = self._load_config(config_path)
        self.boto_config = self._create_boto_config()

    def _load_config(config_path: Optional[str]=None) -> dict:
        if not config_path:
            config_path = Path(__file__).parent / 'config.yaml'
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please create config.yaml in {config_path.parent}"
            )
        
        try:
            return yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in {config_path}: {error}")

    def _create_boto_config(self, cfg: Optional[dict]) -> Config:
        """
        Create boto3 Config object from loaded configuration
        """
        client_cfg = self.config.get('client', {})

        # Build with required params
        config_kwargs = {
            'region_name': client_cfg.get('region_name', 'us-east-2'),
            'max_pool_connections': client_cfg.get('max_pool_connections', 50),
        }

        # Add optional params
        if 'connect_timeout' in client_cfg:
            config_kwargs['connect_timeout'] = client_cfg['connect_timeout']

        if 'read_timeout' in client_cfg:
            config_kwargs['read_timeout'] = client_cfg['read_timeout']

        if 'retries' in client_cfg:
            config_kwargs['retries'] = {
                'mode': client_cfg['retries'].get('mode', 'standard'),
                'total_max_attempts': client_cfg['retries'].get('total_max_attempts', 5)
            }

        if 's3' in client_cfg:
            s3_cfg = client_cfg['s3']
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