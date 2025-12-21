# S3 Configuration Guide

## Overview

This guide explains the AWS S3 client configuration used in the US Equity Data Lake project. Understanding these configuration parameters is critical for optimizing performance, reliability, and cost-efficiency when working with millions of files and hundreds of gigabytes of data.

## Table of Contents

1. [Why Configuration Matters](#why-configuration-matters)
2. [Core Configuration Parameters](#core-configuration-parameters)
3. [Region Configuration](#region-configuration)
4. [Connection Settings](#connection-settings)
5. [Retry Configuration](#retry-configuration)
6. [S3-Specific Settings](#s3-specific-settings)
7. [Advanced Settings](#advanced-settings)
8. [Complete Configuration Example](#complete-configuration-example)

---

## Why Configuration Matters

The default boto3 S3 client works for basic operations, but our data lake has specific requirements:

- **Scale**: ~11.7 million files, ~292 GB of data
- **Concurrency**: Parallel uploads/downloads for faster backfills
- **Reliability**: Network issues shouldn't break long-running data collection
- **Performance**: Minimize latency for frequently accessed data
- **Cost**: Optimize request patterns to reduce S3 API costs

Without proper configuration:
- Connection pool exhaustion during parallel operations
- Failed uploads/downloads due to timeouts
- Wasted API requests from inadequate retry logic
- Suboptimal network routing and performance

---

## Core Configuration Parameters

### Region Name

```python
region_name="us-east-2"
```

**Purpose**: Specifies the AWS region where your S3 bucket is located.

**Why It's Necessary**:
- Ensures data sovereignty and compliance requirements
- Minimizes network latency by routing requests to the correct region
- Affects pricing (data transfer costs vary by region)
- Required for proper endpoint resolution

**What Happens If Removed**:
```python
# Without explicit region_name
client = boto3.client('s3')  # Falls back to default region
```

**Consequences**:
- Uses the default region from your AWS credentials/config file
- If no default is set, defaults to `us-east-1`
- **Cross-region requests**: If your bucket is in `us-east-2` but client uses `us-east-1`, requests still work BUT:
  - Higher latency (traffic routes across regions)
  - Potential data transfer charges between regions
  - Slower upload/download speeds

**Best Practice**: Always explicitly set the region to match your bucket location.

---

## Connection Settings

### Max Pool Connections

```python
max_pool_connections=50
```

**Purpose**: Maximum number of simultaneous HTTP connections in the connection pool.

**Why It's Necessary**:
- Default is only **10 connections**
- Our data lake requires parallel operations:
  - Uploading 100+ files simultaneously during backfills
  - Multi-threaded query API fetching data from multiple symbols
  - Concurrent daily updates for thousands of tickers

**What Happens If Removed** (defaults to 10):
```python
# With default (10 connections)
config = Config()  # max_pool_connections=10 (default)
```

**Scenario**: Parallel upload of 50 files
```python
# With max_pool_connections=10 (DEFAULT)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(s3_client.upload_file, file) for file in files]
    # Result: Only 10 uploads happen simultaneously
    # Remaining 40 threads WAIT for connections to become available
    # Total time: ~5x slower than with 50 connections
```

**With Our Configuration** (50 connections):
```python
# With max_pool_connections=50
# Result: All 50 uploads happen simultaneously
# Total time: ~5x faster for parallel operations
```

**Trade-offs**:
- More connections = more memory usage (minimal impact)
- More connections = better parallelism
- Sweet spot for our use case: 50-100 connections

**Best Practice**: Set to match or exceed your expected parallelism level.

---

### Connect Timeout

```python
connect_timeout=60
```

**Purpose**: Maximum time (in seconds) to wait when establishing a TCP connection to S3.

**Why It's Necessary**:
- Default is 60 seconds (we're being explicit)
- Network conditions vary (corporate firewalls, VPNs, international connections)
- Large file uploads start with connection establishment

**What Happens If Removed** (defaults to 60):
```python
# Default behavior (60 seconds)
config = Config()  # connect_timeout=60 (default)
```

**Scenario**: Unstable network connection
```python
# With connect_timeout=5 (too short)
# Network has temporary congestion taking 8 seconds to establish connection
# Result: Connection fails, upload doesn't even start
# Error: "ConnectTimeoutError: Connect timeout on endpoint URL"

# With connect_timeout=60 (our setting)
# Network takes 8 seconds to establish connection
# Result: Connection succeeds, upload proceeds normally
```

**Trade-offs**:
- Too low (5-10s): Fails on slow/congested networks
- Too high (300s+): Wastes time on truly dead connections
- 60s is a good balance for most scenarios

**When to Adjust**:
- Lower (15-30s): Very reliable, fast networks (AWS EC2 → S3 in same region)
- Higher (120s+): Unreliable networks, international transfers

---

### Read Timeout

```python
read_timeout=60
```

**Purpose**: Maximum time (in seconds) to wait for data chunks during an active transfer.

**Why It's Necessary**:
- Default is 60 seconds
- Critical for large file transfers
- Prevents hanging on stalled transfers

**What Happens If Removed** (defaults to 60):
```python
# Default behavior
config = Config()  # read_timeout=60 (default)
```

**Scenario**: Uploading 100MB Parquet file
```python
# File size: 100 MB
# Network speed: 10 Mbps (1.25 MB/s)
# Expected transfer time: 80 seconds

# With read_timeout=30 (too short)
# After 30 seconds of no data: ReadTimeoutError
# Result: Upload fails at ~37 MB, needs to restart
# Wastes time and bandwidth

# With read_timeout=60 (our setting)
# Still might timeout on very large files with slow connections
# Consider: read_timeout=120 for minute-tick Parquet files

# With read_timeout=300 (5 minutes)
# Handles even very large files on slow connections
# Trade-off: Slow to detect genuinely stalled transfers
```

**Best Practice**:
- Calculate: `file_size / minimum_expected_speed * 2` (2x safety margin)
- Our minute-tick files: 25 KB (instant even on slow connections)
- Our yearly daily-tick files: 16 KB (instant)
- Quarterly fundamental files: 25 KB (instant)
- **Recommendation**: 60s is sufficient for our file sizes

**When to Increase**:
- Large files (>100 MB)
- Slow network connections
- International data transfers

---

## Retry Configuration

```python
retries={
    'mode': 'standard',
    'total_max_attempts': 5
}
```

### Retry Mode

**Purpose**: Determines the retry strategy when requests fail.

**Why It's Necessary**:
- Network failures are inevitable over millions of API calls
- S3 occasionally returns throttling errors (503, 500)
- Automated data collection must be resilient

**Available Modes**:

#### 1. Legacy Mode
```python
retries={'mode': 'legacy'}
```
- Old boto3 retry behavior
- Limited retry rules
- **Not recommended** for new applications

#### 2. Standard Mode (Our Choice)
```python
retries={'mode': 'standard'}
```
- AWS best practices for retry behavior
- Exponential backoff with jitter
- Retries: 500, 502, 503, 504 errors
- Retries: Connection errors, timeouts
- Default: 3 max attempts (we override to 5)

**Example**: API request fails with 503 (Service Unavailable)
```python
# Attempt 1: Immediate request → 503 error
# Wait: ~1 second (exponential backoff)
# Attempt 2: Retry → 503 error
# Wait: ~2 seconds
# Attempt 3: Retry → Success!
# Total time: ~3 seconds (vs. failing immediately)
```

#### 3. Adaptive Mode
```python
retries={'mode': 'adaptive'}
```
- Standard mode + client-side rate limiting
- Dynamically adjusts request rate based on throttling
- **Trade-off**: More complex, may slow down legitimate requests
- **When to use**: Consistently hitting S3 rate limits

**What Happens If Removed**:
```python
# Default retry behavior
config = Config()  # Uses 'legacy' mode with default attempts
```

**Consequences**:
- Fewer intelligent retries
- More manual error handling required
- Long-running backfills may fail and need manual restart

---

### Total Max Attempts

```python
total_max_attempts=5
```

**Purpose**: Maximum number of attempts (initial + retries) for a single request.

**Why It's Necessary**:
- Default with `standard` mode is **3 attempts**
- Our use case: Automated daily updates (no human monitoring)
- Better to retry a few extra times than fail and require manual intervention

**What Happens If Removed** (defaults to 3):
```python
retries={'mode': 'standard'}  # total_max_attempts=3 (default)
```

**Scenario**: Network has brief 10-second outage
```python
# With total_max_attempts=3 (default)
# Attempt 1: Request → Connection error
# Wait: ~1 second
# Attempt 2: Request → Connection error
# Wait: ~2 seconds
# Attempt 3: Request → Connection error
# Result: FAILED after ~3 seconds
# Manual intervention required to restart upload

# With total_max_attempts=5 (our setting)
# Attempt 1: Request → Connection error
# Wait: ~1 second
# Attempt 2: Request → Connection error
# Wait: ~2 seconds
# Attempt 3: Request → Connection error
# Wait: ~4 seconds
# Attempt 4: Request → Connection error
# Wait: ~8 seconds (network recovers)
# Attempt 5: Request → Success!
# Result: Automatic recovery, no manual intervention
```

**Trade-offs**:
- More attempts = longer time before giving up
- More attempts = higher resilience to transient errors
- Too many attempts (20+) = wastes time on permanent failures

**Best Practice**:
- Interactive applications: 3 attempts (fail fast)
- Automated batch jobs: 5-10 attempts (resilience over speed)
- Our setting (5): Good balance for daily automated updates

---

## S3-Specific Settings

### Addressing Style

```python
s3={
    'addressing_style': 'virtual'
}
```

**Purpose**: Determines URL format for S3 requests.

**Why It's Necessary**:
- Affects how bucket names are resolved in URLs
- Impacts DNS resolution and request routing

**Available Styles**:

#### 1. Virtual Hosted-Style (Our Choice)
```python
addressing_style='virtual'
```

**URL Format**: `https://bucket-name.s3.region.amazonaws.com/object-key`

**Example**:
```
https://us-equity-datalake.s3.us-east-2.amazonaws.com/ticks/daily/AAPL/2024/ticks.json
```

**Advantages**:
- ✅ AWS recommended approach
- ✅ Better performance (CDN-friendly)
- ✅ Supports S3 Transfer Acceleration
- ✅ Works with bucket names containing dots

#### 2. Path-Style
```python
addressing_style='path'
```

**URL Format**: `https://s3.region.amazonaws.com/bucket-name/object-key`

**Example**:
```
https://s3.us-east-2.amazonaws.com/us-equity-datalake/ticks/daily/AAPL/2024/ticks.json
```

**Disadvantages**:
- ⚠️ Being deprecated by AWS
- ⚠️ Doesn't work with S3 Transfer Acceleration
- ⚠️ Some features unavailable

#### 3. Auto
```python
addressing_style='auto'  # Default
```

**Behavior**: Boto3 chooses based on bucket name
- Virtual style: If bucket name is DNS-compatible
- Path style: If bucket name has dots or special characters

**What Happens If Removed** (defaults to 'auto'):
```python
s3={}  # addressing_style='auto' (default)
```

**Consequences**:
- Usually fine, but less predictable
- May use path-style for some buckets (deprecated)
- Explicit `virtual` ensures consistent, modern behavior

**Best Practice**: Use `virtual` for all new applications.

---

### Payload Signing

```python
s3={
    'payload_signing_enabled': True
}
```

**Purpose**: Whether to compute SHA256 checksum of request body for SigV4 signing.

**Why It's Necessary**:
- Security: Verifies request integrity
- Prevents tampering with uploaded data
- Required for some S3 features

**What Happens If Removed** (defaults vary by operation):
```python
s3={
    'payload_signing_enabled': False  # Default for uploads >5MB
}
```

**Default Behavior**:
- `PutObject`: Unsigned payload (faster, less secure)
- `UploadPart`: Unsigned payload (streaming uploads)
- `GetObject`: N/A (downloads always verify server checksums)

**Trade-offs**:

**Enabled (Our Choice)**:
- ✅ Maximum security
- ✅ Ensures data integrity
- ⚠️ Slightly slower for large uploads (SHA256 computation)

**Disabled**:
- ✅ Faster uploads (no local checksum computation)
- ⚠️ Less security
- ⚠️ S3 Object Lock and other features may not work

**Performance Impact**:
```python
# File size: 100 MB
# Payload signing enabled: +0.5 seconds (SHA256 computation)
# Payload signing disabled: No overhead

# For our data lake:
# Largest files: ~25 KB (minute-tick daily Parquet)
# SHA256 overhead: <0.01 seconds
# Impact: Negligible
```

**Best Practice**:
- Enable for data integrity (our choice)
- Disable only if you need maximum upload speed and trust your network

---

### US East 1 Regional Endpoint

```python
s3={
    'us_east_1_regional_endpoint': 'regional'
}
```

**Purpose**: Controls S3 endpoint used when region is `us-east-1`.

**Why It's Necessary**:
- `us-east-1` has special historical behavior
- Default uses legacy global endpoint
- Regional endpoint offers better performance

**Available Options**:

#### 1. Regional (Our Choice)
```python
us_east_1_regional_endpoint='regional'
```

**Endpoint**: `https://bucket.s3.us-east-1.amazonaws.com`

**Advantages**:
- ✅ Stays within region (better latency)
- ✅ Consistent with other regions
- ✅ AWS best practice

#### 2. Legacy (Default)
```python
us_east_1_regional_endpoint='legacy'  # Default
```

**Endpoint**: `https://bucket.s3.amazonaws.com`

**Disadvantages**:
- ⚠️ May route through global edge locations
- ⚠️ Slightly higher latency
- ⚠️ Inconsistent with other regions

**What Happens If Removed** (defaults to 'legacy'):
```python
s3={}  # us_east_1_regional_endpoint='legacy'
```

**Impact**:
- Our bucket is in `us-east-2`, so this doesn't affect us
- If we later use `us-east-1`, would use slower legacy endpoint
- Best to be explicit for future-proofing

**Best Practice**: Always use `regional` for better performance.

---

## Advanced Settings

### TCP Keepalive

```python
tcp_keepalive=True
```

**Purpose**: Sends periodic packets on idle connections to detect dead connections.

**Why It's Necessary**:
- Long-running operations (large file uploads)
- Prevents silent connection failures
- Detects network issues earlier

**What Happens If Removed** (defaults to False):
```python
config = Config()  # tcp_keepalive=False (default)
```

**Scenario**: 10-minute upload over flaky corporate network
```python
# Without TCP keepalive (default)
# Upload starts: Connection established
# 5 minutes in: Network cable unplugged (client doesn't know)
# Client keeps sending data to dead connection
# 5 minutes later: Finally times out after read_timeout
# Result: Wasted 10 minutes before detecting failure

# With TCP keepalive (our setting)
# Upload starts: Connection established
# 5 minutes in: Network cable unplugged
# Next keepalive probe (30s): No response
# Client detects dead connection immediately
# Result: Retry starts within 30 seconds
```

**Trade-offs**:
- ✅ Detects connection issues faster
- ✅ Prevents wasted time on dead connections
- ⚠️ Minimal overhead (tiny periodic packets)

**Best Practice**: Enable for long-running operations and unreliable networks.

---

### Request Compression

```python
request_min_compression_size_bytes=10240,  # 10 KB
disable_request_compression=False
```

**Purpose**: Compresses request payloads before sending to S3.

**Why It's Necessary**:
- Reduces bandwidth usage
- Faster uploads on slow connections
- Lower data transfer costs

**Compression Threshold**: 10,240 bytes (10 KB)
- Requests <10 KB: Not compressed (overhead not worth it)
- Requests ≥10 KB: Compressed with gzip

**What Happens If Removed**:
```python
# Defaults
request_min_compression_size_bytes=0  # Compress everything
disable_request_compression=True  # Don't compress anything (default)
```

**Scenario**: Uploading JSON fundamental data (50 KB)
```python
# Without compression (default)
# File size: 50 KB (JSON is very compressible)
# Upload size: 50 KB
# Upload time on 1 Mbps: 0.4 seconds
# S3 API cost: Based on 50 KB

# With compression (our setting)
# File size: 50 KB
# Compressed size: ~10 KB (80% reduction typical for JSON)
# Upload size: 10 KB
# Upload time on 1 Mbps: 0.08 seconds (5x faster!)
# S3 API cost: Based on 10 KB (80% savings)
```

**Trade-offs**:
- ✅ Faster uploads (especially JSON/CSV)
- ✅ Lower bandwidth costs
- ⚠️ CPU overhead for compression (minimal)
- ⚠️ Not beneficial for already-compressed data (Parquet, images)

**Our Data Types**:
- JSON files (daily ticks, fundamentals): Highly compressible, benefit greatly
- Parquet files (minute ticks): Already compressed, minimal benefit
- 10 KB threshold: Good balance

**Best Practice**: Enable with appropriate threshold (10-100 KB).

---

### Checksum Validation

```python
request_checksum_calculation='when_supported',
response_checksum_validation='when_supported'
```

**Purpose**: Ensures data integrity during upload/download.

**Why It's Necessary**:
- Detects data corruption
- Verifies S3 received exactly what you sent
- Critical for financial data accuracy

**Request Checksum Calculation**:

**when_supported** (our choice):
- Calculates checksums for operations that support it
- S3 verifies uploaded data matches checksum
- Automatic data integrity verification

**when_required**:
- Only calculates checksums when absolutely required
- Faster but less data integrity checking

**What happens without it**:
```python
request_checksum_calculation='when_required'  # Minimal checking
```

**Scenario**: Network corruption during upload
```python
# Without checksum (when_required)
# Original data: {"price": 150.25}
# Network corrupts bit: {"price": 150.35}  # Wrong!
# S3 stores corrupted data
# Result: Financial data is WRONG, no error raised

# With checksum (when_supported - our setting)
# Original data: {"price": 150.25}
# Checksum: abc123def
# Network corrupts bit: {"price": 150.35}
# Checksum: xyz789ghi (doesn't match)
# S3 rejects upload: "Checksum mismatch"
# Client retries with correct data
# Result: Data integrity preserved
```

**Response Checksum Validation**: Similar for downloads

**Trade-offs**:
- ✅ Guarantees data integrity
- ⚠️ Tiny overhead (checksum calculation)
- For financial data: **Non-negotiable**

**Best Practice**: Always enable for critical data.

---

## Complete Configuration Example

Here's our full configuration with all parameters explained:

```python
from botocore.config import Config
import boto3

# Create optimized configuration
config = Config(
    # === REGION ===
    region_name="us-east-2",  # Match bucket location for best performance

    # === CONNECTION POOL ===
    max_pool_connections=50,  # Support 50 parallel uploads/downloads

    # === TIMEOUTS ===
    connect_timeout=60,  # 60s to establish connection (handles slow networks)
    read_timeout=60,      # 60s to receive data chunk (handles large files)

    # === RETRY BEHAVIOR ===
    retries={
        'mode': 'standard',         # AWS best-practice retry logic
        'total_max_attempts': 5     # Retry up to 5 times for resilience
    },

    # === S3-SPECIFIC ===
    s3={
        'addressing_style': 'virtual',  # Modern URL format (recommended)
        'payload_signing_enabled': True,  # Verify upload integrity
        'us_east_1_regional_endpoint': 'regional'  # Use regional endpoint
    },

    # === PERFORMANCE ===
    tcp_keepalive=True,  # Detect dead connections faster

    # === COMPRESSION ===
    request_min_compression_size_bytes=10240,  # Compress requests >10KB
    disable_request_compression=False,  # Enable compression

    # === DATA INTEGRITY ===
    request_checksum_calculation='when_supported',  # Verify uploads
    response_checksum_validation='when_supported',  # Verify downloads
)

# Create S3 client with configuration
s3_client = boto3.client('s3', config=config)
```

---

## Configuration Impact Summary

| Parameter | Default | Our Value | Impact if Removed |
|-----------|---------|-----------|-------------------|
| `region_name` | `us-east-1` | `us-east-2` | Cross-region latency, higher costs |
| `max_pool_connections` | 10 | 50 | 5x slower parallel operations |
| `connect_timeout` | 60 | 60 | Same (explicit for clarity) |
| `read_timeout` | 60 | 60 | Same (explicit for clarity) |
| `retries.mode` | `legacy` | `standard` | Less intelligent retry behavior |
| `retries.total_max_attempts` | 3 | 5 | More manual intervention needed |
| `addressing_style` | `auto` | `virtual` | May use deprecated path-style |
| `payload_signing_enabled` | Varies | `True` | Less data integrity verification |
| `us_east_1_regional_endpoint` | `legacy` | `regional` | Slower if using us-east-1 |
| `tcp_keepalive` | `False` | `True` | Slower dead connection detection |
| `request_min_compression_size_bytes` | 0 | 10240 | No compression (slower uploads) |
| `disable_request_compression` | `True` | `False` | No compression (higher bandwidth) |
| `request_checksum_calculation` | `when_required` | `when_supported` | Less data integrity checking |
| `response_checksum_validation` | `when_required` | `when_supported` | Less download verification |

---

## Tuning for Different Scenarios

### High-Speed Local Network (EC2 in same region as S3)
```python
config = Config(
    region_name="us-east-2",
    max_pool_connections=100,  # Even more parallelism
    connect_timeout=15,         # Fast network, fail fast
    read_timeout=30,            # Fast transfers
    retries={'mode': 'standard', 'total_max_attempts': 3}  # Fewer retries needed
)
```

### Slow/Unreliable Network (International, VPN)
```python
config = Config(
    region_name="us-east-2",
    max_pool_connections=20,    # Less parallelism (avoid overwhelming connection)
    connect_timeout=120,         # Give more time to connect
    read_timeout=300,            # Give more time for large transfers
    retries={'mode': 'adaptive', 'total_max_attempts': 10}  # More retries, adaptive throttling
)
```

### Maximum Performance (Trust Network, Need Speed)
```python
config = Config(
    region_name="us-east-2",
    max_pool_connections=100,
    connect_timeout=10,
    read_timeout=30,
    retries={'mode': 'standard', 'total_max_attempts': 2},
    s3={'payload_signing_enabled': False},  # Skip signing for speed
    tcp_keepalive=False,
    disable_request_compression=True  # Skip compression overhead
)
```

### Maximum Reliability (Financial Data, Can't Afford Errors)
```python
config = Config(
    region_name="us-east-2",
    max_pool_connections=50,
    connect_timeout=120,
    read_timeout=300,
    retries={'mode': 'adaptive', 'total_max_attempts': 10},
    s3={'payload_signing_enabled': True},  # Verify everything
    tcp_keepalive=True,
    request_checksum_calculation='when_supported',  # Always verify
    response_checksum_validation='when_supported'
)
```

---

## Monitoring and Troubleshooting

### How to Test Your Configuration

```python
import boto3
from botocore.config import Config
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create client
config = Config(region_name="us-east-2", max_pool_connections=50)
s3_client = boto3.client('s3', config=config)

# Test connection
try:
    response = s3_client.list_buckets()
    print(f"✓ Successfully connected to S3")
    print(f"✓ Available buckets: {len(response['Buckets'])}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Common Issues

**Issue**: `ConnectionClosedError` during parallel uploads
- **Cause**: `max_pool_connections` too low
- **Solution**: Increase to match parallelism level

**Issue**: `ReadTimeoutError` on large files
- **Cause**: `read_timeout` too low for file size / network speed
- **Solution**: Increase `read_timeout` or improve network speed

**Issue**: Uploads failing with 503 errors
- **Cause**: S3 throttling, need better retry logic
- **Solution**: Use `retries={'mode': 'adaptive'}` or reduce request rate

**Issue**: Slow uploads despite fast network
- **Cause**: Compression disabled or not enough parallelism
- **Solution**: Enable compression, increase `max_pool_connections`

---

## Next Steps

1. **Review your bucket location**: Ensure `region_name` matches your S3 bucket region
2. **Test the configuration**: Run the test script above
3. **Monitor performance**: Track upload/download speeds during backfills
4. **Tune as needed**: Adjust based on your network characteristics
5. **Document changes**: Update this guide if you modify configuration

## Further Reading

- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
- [Boto3 Configuration Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html)
- [Botocore Config Reference](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html)
- [AWS SDK Retry Behavior](https://aws.amazon.com/blogs/developer/tuning-the-aws-java-sdk-2-x-to-reduce-startup-time/)
