"""
Unit tests for storage.rate_limiter module
Tests thread-safe rate limiting functionality
"""
import pytest
import time
import threading
from quantdl.storage.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test RateLimiter class"""

    def test_initialization(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter(max_rate=10.0)
        assert limiter.max_rate == 10.0
        assert limiter.min_interval == 0.1  # 1.0 / 10.0
        assert limiter.last_request_time == 0

    def test_single_request_no_delay(self):
        """Test that first request doesn't block"""
        limiter = RateLimiter(max_rate=10.0)
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start
        # First request should be immediate (< 10ms overhead)
        assert elapsed < 0.01

    def test_rate_limiting_enforced(self):
        """Test that rate limit is enforced"""
        limiter = RateLimiter(max_rate=10.0)  # 10 requests per second

        # Make first request
        start = time.time()
        limiter.acquire()

        # Make second request immediately
        limiter.acquire()
        elapsed = time.time() - start

        # Should have waited at least min_interval (0.1s)
        # Allow some tolerance for timing precision
        assert elapsed >= 0.09  # At least 90ms (allow 10ms tolerance)

    def test_multiple_sequential_requests(self):
        """Test rate limiting across multiple requests"""
        limiter = RateLimiter(max_rate=5.0)  # 5 requests per second

        num_requests = 5
        start = time.time()

        for _ in range(num_requests):
            limiter.acquire()

        elapsed = time.time() - start

        # Expected time: (num_requests - 1) * min_interval
        # First request is immediate, then 4 * 0.2s = 0.8s
        expected_min_time = (num_requests - 1) * limiter.min_interval
        assert elapsed >= expected_min_time * 0.9  # Allow 10% tolerance

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe"""
        limiter = RateLimiter(max_rate=10.0)
        num_threads = 5
        requests_per_thread = 3
        total_requests = num_threads * requests_per_thread

        results = []
        lock = threading.Lock()

        def worker():
            for _ in range(requests_per_thread):
                request_time = time.time()
                limiter.acquire()
                with lock:
                    results.append(request_time)

        # Start all threads
        threads = []
        start = time.time()
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        elapsed = time.time() - start

        # Verify correct number of requests
        assert len(results) == total_requests

        # Expected minimum time: (total_requests - 1) * min_interval
        expected_min_time = (total_requests - 1) * limiter.min_interval
        assert elapsed >= expected_min_time * 0.9  # Allow 10% tolerance

    def test_high_rate_limit(self):
        """Test with high rate limit (SEC EDGAR: 9.5 req/sec)"""
        limiter = RateLimiter(max_rate=9.5)

        # Make 10 requests
        num_requests = 10
        start = time.time()

        for _ in range(num_requests):
            limiter.acquire()

        elapsed = time.time() - start

        # Expected: ~0.95 seconds for 10 requests at 9.5 req/sec
        expected_min_time = (num_requests - 1) / 9.5
        assert elapsed >= expected_min_time * 0.9

    def test_low_rate_limit(self):
        """Test with low rate limit"""
        limiter = RateLimiter(max_rate=2.0)  # 2 requests per second

        num_requests = 3
        start = time.time()

        for _ in range(num_requests):
            limiter.acquire()

        elapsed = time.time() - start

        # Expected: ~1.0 seconds for 3 requests at 2 req/sec
        expected_min_time = (num_requests - 1) * 0.5
        assert elapsed >= expected_min_time * 0.9

    def test_reset_between_requests(self):
        """Test that limiter resets properly between requests"""
        limiter = RateLimiter(max_rate=5.0)

        # First request
        limiter.acquire()

        # Wait longer than min_interval
        time.sleep(0.3)  # min_interval is 0.2s

        # Second request should not block
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start

        # Should be immediate since we waited longer than min_interval
        assert elapsed < 0.01

    def test_concurrent_acquire(self):
        """Test concurrent acquires from multiple threads"""
        limiter = RateLimiter(max_rate=100.0)  # Fast rate for testing
        num_threads = 10
        acquire_count = []
        lock = threading.Lock()

        def worker():
            limiter.acquire()
            with lock:
                acquire_count.append(1)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should have acquired
        assert len(acquire_count) == num_threads
