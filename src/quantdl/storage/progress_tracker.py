"""
Progress Tracker for Upload Operations
Tracks completed security_ids in S3 for resume capability
"""
import json
from datetime import datetime, timezone
from typing import Set, Optional
from botocore.exceptions import ClientError


class UploadProgressTracker:
    """
    Tracks completed security_ids for upload operations.

    Stores progress in S3 as JSON:
        data/upload_progress/{task_name}.json

    Schema:
        {
            'completed': [1001, 1002, ...],
            'last_updated': '2025-01-13T10:30:00Z',
            'stats': {'total': 5000, 'completed': 2500, 'failed': 10}
        }
    """

    def __init__(
        self,
        s3_client,
        bucket_name: str,
        task_name: str = 'daily_ticks_backfill',
        flush_interval: int = 100
    ):
        """
        Initialize progress tracker.

        :param s3_client: Boto3 S3 client
        :param bucket_name: S3 bucket name
        :param task_name: Unique name for this upload task
        :param flush_interval: Save to S3 every N completions (default: 100)
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.task_name = task_name
        self.s3_key = f'data/upload_progress/{task_name}.json'
        self.flush_interval = flush_interval

        self._completed: Set[int] = set()
        self._pending_count = 0
        self._stats = {'total': 0, 'completed': 0, 'failed': 0, 'skipped': 0}
        self._loaded = False

    def load(self) -> Set[int]:
        """
        Load completed security_ids from S3.

        :return: Set of completed security_ids
        """
        if self._loaded:
            return self._completed

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.s3_key
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            self._completed = set(data.get('completed', []))
            self._stats = data.get('stats', self._stats)
            self._loaded = True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'NoSuchKey':
                # No progress file yet - start fresh
                self._completed = set()
                self._loaded = True
            else:
                raise

        return self._completed

    def mark_completed(self, security_id: int):
        """
        Mark a security_id as completed.

        Automatically flushes to S3 every flush_interval completions.

        :param security_id: Security ID that was successfully processed
        """
        if not self._loaded:
            self.load()

        self._completed.add(security_id)
        self._stats['completed'] = len(self._completed)
        self._pending_count += 1

        if self._pending_count >= self.flush_interval:
            self.save()
            self._pending_count = 0

    def mark_failed(self, security_id: int):
        """
        Increment failed counter (does not mark as completed).

        :param security_id: Security ID that failed processing
        """
        self._stats['failed'] += 1

    def mark_skipped(self, security_id: int):
        """
        Increment skipped counter.

        :param security_id: Security ID that was skipped
        """
        self._stats['skipped'] += 1

    def set_total(self, total: int):
        """
        Set total number of security_ids to process.

        :param total: Total count
        """
        self._stats['total'] = total

    def is_completed(self, security_id: int) -> bool:
        """
        Check if a security_id has already been processed.

        :param security_id: Security ID to check
        :return: True if already completed
        """
        if not self._loaded:
            self.load()
        return security_id in self._completed

    def get_pending(self, all_security_ids: Set[int]) -> Set[int]:
        """
        Get security_ids that haven't been processed yet.

        :param all_security_ids: Full set of security_ids to process
        :return: Set of pending security_ids
        """
        if not self._loaded:
            self.load()
        return all_security_ids - self._completed

    def save(self):
        """
        Save progress to S3.
        """
        data = {
            'completed': sorted(list(self._completed)),
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'stats': self._stats
        }

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=self.s3_key,
            Body=json.dumps(data, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

    def reset(self):
        """
        Reset progress (delete S3 file and clear state).
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=self.s3_key
            )
        except ClientError:
            pass

        self._completed = set()
        self._stats = {'total': 0, 'completed': 0, 'failed': 0, 'skipped': 0}
        self._pending_count = 0
        self._loaded = True

    @property
    def stats(self) -> dict:
        """Get current stats."""
        return self._stats.copy()

    def __enter__(self):
        """Context manager entry - load state."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save state."""
        if self._pending_count > 0:
            self.save()
        return False
