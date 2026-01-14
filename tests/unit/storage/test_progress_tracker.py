"""
Unit tests for storage.progress_tracker.UploadProgressTracker
Tests progress tracking and resume capability
"""
import pytest
import json
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from quantdl.storage.progress_tracker import UploadProgressTracker


class TestUploadProgressTracker:
    """Test UploadProgressTracker class"""

    def test_init_defaults(self):
        """Test default initialization"""
        mock_s3 = Mock()
        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        assert tracker.bucket_name == 'test-bucket'
        assert tracker.task_name == 'daily_ticks_backfill'
        assert tracker.s3_key == 'data/upload_progress/daily_ticks_backfill.json'
        assert tracker.flush_interval == 100

    def test_load_fresh_start(self):
        """Test load when no progress file exists"""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket',
            task_name='test_task'
        )

        result = tracker.load()
        assert result == set()
        assert tracker._loaded is True

    def test_load_existing_progress(self):
        """Test load with existing progress file"""
        mock_s3 = Mock()
        progress_data = {
            'completed': [1001, 1002, 1003],
            'last_updated': '2025-01-13T10:00:00Z',
            'stats': {'total': 100, 'completed': 3, 'failed': 0, 'skipped': 0}
        }
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=json.dumps(progress_data).encode()))
        }

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        result = tracker.load()
        assert result == {1001, 1002, 1003}
        assert tracker._stats['completed'] == 3

    def test_mark_completed(self):
        """Test marking security_id as completed"""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket',
            flush_interval=3
        )

        tracker.mark_completed(1001)
        tracker.mark_completed(1002)
        assert 1001 in tracker._completed
        assert 1002 in tracker._completed
        # Should not have saved yet (flush_interval=3)
        mock_s3.put_object.assert_not_called()

        # Third completion triggers flush
        tracker.mark_completed(1003)
        mock_s3.put_object.assert_called_once()

    def test_is_completed(self):
        """Test checking if security_id is completed"""
        mock_s3 = Mock()
        progress_data = {
            'completed': [1001, 1002],
            'stats': {'total': 10, 'completed': 2, 'failed': 0, 'skipped': 0}
        }
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=json.dumps(progress_data).encode()))
        }

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        assert tracker.is_completed(1001) is True
        assert tracker.is_completed(1002) is True
        assert tracker.is_completed(1003) is False

    def test_get_pending(self):
        """Test getting pending security_ids"""
        mock_s3 = Mock()
        progress_data = {
            'completed': [1001, 1002],
            'stats': {}
        }
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=json.dumps(progress_data).encode()))
        }

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        all_sids = {1001, 1002, 1003, 1004, 1005}
        pending = tracker.get_pending(all_sids)

        assert pending == {1003, 1004, 1005}

    def test_save(self):
        """Test saving progress to S3"""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        tracker.load()
        tracker._completed = {1001, 1002}
        tracker._stats = {'total': 10, 'completed': 2, 'failed': 0, 'skipped': 0}

        tracker.save()

        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args
        assert call_args.kwargs['Bucket'] == 'test-bucket'
        assert call_args.kwargs['Key'] == 'data/upload_progress/daily_ticks_backfill.json'
        assert call_args.kwargs['ContentType'] == 'application/json'

        # Verify saved data
        saved_data = json.loads(call_args.kwargs['Body'].decode())
        assert set(saved_data['completed']) == {1001, 1002}
        assert 'last_updated' in saved_data
        assert saved_data['stats']['completed'] == 2

    def test_reset(self):
        """Test resetting progress"""
        mock_s3 = Mock()
        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        tracker._completed = {1001, 1002}
        tracker._stats = {'total': 10, 'completed': 2, 'failed': 0, 'skipped': 0}

        tracker.reset()

        assert tracker._completed == set()
        assert tracker._stats['completed'] == 0
        mock_s3.delete_object.assert_called_once()

    def test_context_manager(self):
        """Test context manager behavior"""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket',
            flush_interval=100
        )

        with tracker:
            tracker.mark_completed(1001)
            # Not flushed yet (below flush_interval)
            assert mock_s3.put_object.call_count == 0

        # Context exit should save pending changes
        mock_s3.put_object.assert_called_once()

    def test_mark_failed_and_skipped(self):
        """Test marking failed and skipped"""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        tracker.load()

        tracker.mark_failed(1001)
        tracker.mark_failed(1002)
        tracker.mark_skipped(1003)

        assert tracker._stats['failed'] == 2
        assert tracker._stats['skipped'] == 1
        # Failed/skipped should NOT be in completed set
        assert 1001 not in tracker._completed

    def test_set_total(self):
        """Test setting total count"""
        mock_s3 = Mock()
        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        tracker.set_total(5000)
        assert tracker._stats['total'] == 5000

    def test_stats_property(self):
        """Test stats property returns copy"""
        mock_s3 = Mock()
        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        tracker._stats = {'total': 100, 'completed': 50, 'failed': 5, 'skipped': 10}

        stats = tracker.stats
        assert stats == {'total': 100, 'completed': 50, 'failed': 5, 'skipped': 10}

        # Modifying returned stats shouldn't affect internal state
        stats['total'] = 999
        assert tracker._stats['total'] == 100

    def test_load_already_loaded_returns_early(self):
        """Test load returns early if already loaded (line 59)."""
        mock_s3 = Mock()
        progress_data = {
            'completed': [1001],
            'stats': {'total': 10, 'completed': 1, 'failed': 0, 'skipped': 0}
        }
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=json.dumps(progress_data).encode()))
        }

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        # First load
        result1 = tracker.load()
        assert result1 == {1001}

        # Second load should return cached result without calling S3 again
        result2 = tracker.load()
        assert result2 == {1001}

        # S3 get_object should only be called once
        assert mock_s3.get_object.call_count == 1

    def test_load_reraises_non_nosuchkey_error(self):
        """Test load re-raises non-NoSuchKey ClientError (line 76)."""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3.get_object.side_effect = ClientError(error_response, 'GetObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )

        with pytest.raises(ClientError) as exc_info:
            tracker.load()

        assert exc_info.value.response['Error']['Code'] == 'AccessDenied'

    def test_reset_handles_client_error_silently(self):
        """Test reset handles ClientError during delete (lines 171,172)."""
        mock_s3 = Mock()
        error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Not Found'}}
        mock_s3.delete_object.side_effect = ClientError(error_response, 'DeleteObject')

        tracker = UploadProgressTracker(
            s3_client=mock_s3,
            bucket_name='test-bucket'
        )
        tracker._completed = {1001, 1002}
        tracker._stats = {'total': 10, 'completed': 2, 'failed': 0, 'skipped': 0}

        # Should not raise, silently handles ClientError
        tracker.reset()

        # State should be reset
        assert tracker._completed == set()
        assert tracker._stats['completed'] == 0
        assert tracker._loaded is True
