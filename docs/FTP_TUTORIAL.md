## FTP & ftplib Tutorial for Market Data Collection

What is FTP?

File Transfer Protocol (FTP) is a standard network protocol for transferring files between a client and server. NASDAQ and other financial data providers use FTP servers to distribute official reference data like:

- Ticker symbols and metadata
- Index constituents
- Corporate actions
- Trading calendars

Core ftplib Concepts

Python's ftplib module provides a simple interface to FTP servers. The workflow is:

1. Connect to FTP server
2. Login with credentials
3. Navigate directories
4. List or download files
5. Close connection

---
Essential Methods

1. Establishing Connection
```python
from ftplib import FTP

ftp = FTP('ftp.nasdaqtrader.com')
```
`FTP(host, user='', passwd='', timeout=...)`
- host: FTP server address (required)
- user, passwd: Login credentials (default: anonymous)
- timeout: Socket timeout in seconds (prevents hanging on slow connections)

Why needed: Initiates TCP connection to the FTP server on port 21.

---
2. Authentication
```python
ftp.login(user='anonymous', passwd='user@example.com')
```
`login(user, passwd)`
- user: Username (NASDAQ often uses 'anonymous')
- passwd: Password (email convention for anonymous FTP)

Why needed: FTP requires authentication even for public data. Anonymous FTP accepts any email as password.

---
3. Directory Navigation
```python
ftp.cwd('/SymbolDirectory')  # Change working directory
current_dir = ftp.pwd()      # Print working directory
```
`cwd(pathname)`
- pathname: Target directory path (absolute or relative)

Why needed: NASDAQ organizes data by category (e.g., /SymbolDirectory/, /Dividends/). You must navigate to the correct folder.

pwd() - Returns current directory as string. Useful for debugging or logging your position.

---
4. Listing Files
```python
files = []
ftp.dir(files.append)  # Stores each line of directory listing
# OR
ftp.retrlines('LIST', lambda line: print(line))
```
`dir(callback)`
- callback: Function called with each line of directory listing (includes permissions, size, date, filename)

Why needed: NASDAQ updates files daily with date-stamped names (nasdaqlisted_20241221.txt). You need to find the latest file programmatically.

Alternative: nlst() - Returns only filenames without metadata:
filenames = ftp.nlst()  # ['nasdaqlisted.txt', 'otherlisted.txt']

---
5. Downloading Files
```python
# Download as binary
with open('local_file.txt', 'wb') as f:
ftp.retrbinary('RETR nasdaqlisted.txt', f.write)

# Download as text (handles line endings)
lines = []
ftp.retrlines('RETR nasdaqlisted.txt', lines.append)
```
`retrbinary(cmd, callback, blocksize=8192)`
- cmd: FTP command (always 'RETR filename' for download)
- callback: Function to process each data block (typically file.write)
- blocksize: Bytes per chunk (8KB default balances memory/speed)

Why needed: Downloads file in binary mode. Critical for Parquet/Excel files where encoding matters.

`retrlines(cmd, callback)`
- Same as retrbinary but for text files
- Automatically handles CRLF line endings

Why needed: Simpler for CSV/TXT files, automatically decodes to strings.

---
6. Error Handling
```python
from ftplib import error_perm, error_temp

try:
ftp.retrbinary('RETR nonexistent.txt', open('out.txt', 'wb').write)
except error_perm as e:
print(f"Permanent error: {e}")  # File not found, access denied
except error_temp as e:
print(f"Temporary error: {e}")  # Server busy, retry later
```
Exception Types:
- error_perm: Permanent errors (550 File not found, 530 Login incorrect)
- error_temp: Temporary errors (421 Service not available, timeout)

Why needed: NASDAQ files may be temporarily unavailable during updates. Distinguish between "retry" vs "fail" scenarios.

---
7. Cleanup
```python
ftp.quit()  # Polite close with QUIT command
# OR
ftp.close()  # Immediate close (use if server unresponsive)

quit() - Sends QUIT command and waits for server acknowledgment.
close() - Immediately closes socket without server handshake.
```
Why needed: Releases server resources. Use quit() normally, close() in exception handlers.

---
Practical Example for NASDAQ Data
```python
from ftplib import FTP, error_perm
import re

def download_latest_ticker_list():
ftp = FTP('ftp.nasdaqtrader.com', timeout=30)
ftp.login()  # Anonymous login

try:
    ftp.cwd('/SymbolDirectory')

    # Find latest file (pattern: nasdaqlisted_YYYYMMDD.txt)
    files = ftp.nlst()
    latest = max(f for f in files if f.startswith('nasdaqlisted_'))

    # Download
    with open(f'data/reference/{latest}', 'wb') as f:
        ftp.retrbinary(f'RETR {latest}', f.write)

    print(f"Downloaded: {latest}")

except error_perm as e:
    print(f"FTP error: {e}")

finally:
    ftp.quit()
```
Key points:
1. Timeout: Prevents hanging on NASDAQ server delays
2. Anonymous login: No credentials needed for public data
3. File pattern matching: Handles date-stamped files
4. Exception handling: Graceful failure if file unavailable
5. finally block: Ensures connection cleanup

---
Common Pitfalls

1. Binary vs Text Mode
- Use retrbinary() for all files unless you specifically need automatic line-ending conversion
- Text mode can corrupt binary files (Parquet, Excel)

2. Working Directory
- FTP maintains state across commands - always verify pwd() after cwd()
- Relative paths depend on current directory

3. Connection Timeouts
- Set explicit timeout parameter for production code
- NASDAQ FTP can be slow during market hours

4. File Locks
- NASDAQ may update files during trading day
- Download outside market hours (after 6 PM ET) for consistency

---
Next Steps

For this data lake project, you'll use ftplib to:
1. Daily task: Download latest nasdaqlisted.txt and otherlisted.txt for ticker universe
2. Weekly task: Fetch index constituent files if NASDAQ provides them
3. Validation: Compare NASDAQ ticker list with SEC EDGAR filings to ensure completeness

The key is automating the "find latest file" logic since NASDAQ uses date-stamped filenames that change daily