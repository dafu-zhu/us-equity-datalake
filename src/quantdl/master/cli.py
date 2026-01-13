"""CLI for SecurityMaster operations"""
import argparse
from pathlib import Path
from dotenv import load_dotenv

from quantdl.master.security_master import SecurityMaster
from quantdl.storage.s3_client import S3Client
from quantdl.utils.logger import setup_logger

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="SecurityMaster operations")
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export SecurityMaster to S3'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild from WRDS (skip S3 cache)'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default='us-equity-datalake',
        help='S3 bucket name'
    )
    parser.add_argument(
        '--s3-key',
        type=str,
        default='data/master/security_master.parquet',
        help='S3 key for export'
    )

    args = parser.parse_args()

    logger = setup_logger(
        name="master.cli",
        log_dir=Path("data/logs/master"),
        console_output=True
    )

    if args.export:
        logger.info("Exporting SecurityMaster to S3...")

        s3_client = S3Client().client
        master = SecurityMaster(
            s3_client=s3_client,
            bucket_name=args.bucket,
            s3_key=args.s3_key,
            force_rebuild=args.force_rebuild
        )

        result = master.export_to_s3(s3_client, args.bucket, args.s3_key)
        logger.info(f"Export complete: {result}")
        master.close()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
