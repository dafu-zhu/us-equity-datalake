"""
SEC EDGAR Stock Fetcher
===============================

Fetches all CURRENT actively traded US common stocks from Nasdaq Trader
"""
import io
import os
import re
import requests
import pandas as pd
from datetime import datetime
from ftplib import FTP
from pathlib import Path
import logging

from utils.logger import setup_logger


def is_common_stock(name: str) -> bool:
    """
    Determines if a security is a common stock based on its name.

    :return: True if the security is a common stock, False otherwise.
    """
    if pd.isna(name) or not isinstance(name, str):
        return False

    # Direct exclusions (simple substring matching)
    direct_exclusions = [
        "Preferred",
        "Preference",
        "Pfd Ser",
        "Series",
        "Subordinate",
        "Notes",
        "Limited Partner",
        "Beneficial Interest",
        "Cmn Shs of BI",
        "Closed End Fund",
        "Depositary Share",
        "Depositary Receipt",
        "Redeemable",
        "Perpetual",
        "Convertible"
    ]

    for keyword in direct_exclusions:
        if keyword in name:
            return False

    # Word boundary exclusions (must match whole words to avoid false positives)
    # e.g., "Unit" should match "Units" but not "Uniti"
    word_boundary_exclusions = [
        r'\bUnits?\b',      # Matches "Unit" or "Units"
        r'\bRights?\b',     # Matches "Right" or "Rights"
        r'\bWarrants?\b',   # Matches "Warrant" or "Warrants"
    ]

    for pattern in word_boundary_exclusions:
        if re.search(pattern, name):
            return False

    # Case-sensitive check
    for keyword in ["ADS", "ADR", "ETN"]:
        if keyword in name:
            return False

    # Check for percentage symbol
    if "%" in name:
        return False

    # Check for Closed End Fund patterns
    trust_fund_keywords = ["Trust", "Fund"]
    has_trust_fund = any(keyword in name for keyword in trust_fund_keywords)

    is_debt = False
    if has_trust_fund:
        debt_keywords = ["Income", "Municipal", "Bond", "Term", "Securities", "Premium", "Rate", "Yield"]
        is_debt = any(keyword in name for keyword in debt_keywords)

    reit_keywords = ["Realty", "Real Estate", "REIT"]
    is_reit = any(keyword in name for keyword in reit_keywords)

    if not is_reit and is_debt:
        return False

    return True


def fetch_all_stocks(with_filter=True, refresh=True, logger=None) -> pd.DataFrame:
    """
    Fetches or loads the current ticker list.

    :param with_filter: If True, filters out ETFs, test issues, and non-common stocks
    :param refresh: If True, fetches fresh data from Nasdaq FTP. If False, reads from existing stock_exchange.csv
    :param logger: Logger instance
    :return: DataFrame with Ticker and Name columns
    """
    ftp_host = "ftp.nasdaqtrader.com"
    ftp_dir = "SymbolDirectory"
    ftp_file = "nasdaqtraded.txt"

    if not logger:
        log_dir = Path("data/logs/symbols")
        logger = setup_logger("symbols", log_dir, logging.INFO, console_output=True)

    # If refresh=False, try to read from existing CSV
    if not refresh:
        csv_path = Path("data/symbols/stock_exchange.csv")
        if csv_path.exists():
            logger.info(f"Loading symbols from existing CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} symbols from cache")
            return df
        else:
            logger.warning(f"CSV file not found at {csv_path}, fetching fresh data instead")

    try:
        logger.info(f"Connecting to FTP: {ftp_host}...")
        
        # Establish FTP Connection
        ftp = FTP(ftp_host)
        ftp.login()
        ftp.cwd(ftp_dir)
        
        # Download file to memory (BytesIO)
        logger.info(f"Downloading the latest {ftp_file}...")
        byte_buffer = io.BytesIO()
        
        # Retrieve a file
        ftp.retrbinary(f"RETR {ftp_file}", byte_buffer.write)
        ftp.quit()
        
        # Reset buffer pointer to the beginning so Pandas can read it
        byte_buffer.seek(0)
        
        # Read CSV directly from URL (Separator is '|')
        # Funny enough: without specifying dtype, pandas recognize 'NaN' as null, which is in fact 'Nano Labs Ltd'
        df = pd.read_csv(byte_buffer, sep='|', dtype={'Symbol': str}, keep_default_na=False, na_values=[''])

        df = df.rename(columns={'Symbol': 'Ticker', 'Security Name': 'Name'})
        
        # Remove the file footer
        df = df[:-1]
        
        if with_filter:
            # FILTER: Exclude ETFs
            if 'ETF' in df.columns:
                df = df[df['ETF'] == 'N']
                
            # FILTER: Exclude Test Issues
            if 'Test Issue' in df.columns:
                df = df[df['Test Issue'] == 'N']

            # FILTER: Exclude non common stocks
            logger.info(f"Before common stock filter: {len(df)} securities")
            df = df[df['Name'].apply(is_common_stock)]
            df = df[~df['Ticker'].str.contains('$', regex=False)]
            logger.info(f"After common stock filter: {len(df)} securities")

        # Remove duplicates
        df = df.drop_duplicates(subset=['Ticker'], keep='first')
        df = df.sort_values('Ticker').reset_index(drop=True)
        
        logger.info(f"Final Universe: {len(df)} common stocks")
        
        # Store result
        dir_path = os.path.join('data', 'symbols')
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, 'stock_exchange.csv')
        
        # Save only relevant columns
        output_df = df[['Ticker', 'Name']]
        output_df.to_csv(file_path, index=False)
        
        return output_df

    except Exception as error:
        logger.error(f"Error fetching Nasdaq data: {error}")
        return pd.DataFrame()
    

if __name__ == "__main__":
    import time
    # Test the is_common_stock helper function
    test_cases = [
        ("FTAI Aviation Ltd. - 9.500% Fixed-Rate Reset Series D Cumulative Perpetual Redeemable Preferred Shares", False),
        ("Bank of America Corporation Non Cumulative Perpetual Conv Pfd Ser L", False),
        ("EPR Properties Series E Cumulative Conv Pfd Shs Ser E", False),
        ("Axiom Intelligence Acquisition Corp 1 - Right", False),
        ("Bitcoin Infrastructure Acquisition Corp Ltd. - Units", False),
        ("Digi Power X Inc. - Common Subordinate Voting Shares", False),
        ("Empire State Realty OP, L.P. Series ES Operating Partnership Units Representing Limited Partnership Interests", False),
        ("Eaton Vance Short Diversified Income Fund Eaton Vance Short Duration Diversified Income Fund Common Shares of Beneficial Interest", False),
        ("Fidus Investment Corporation - Closed End Fund", False),
        ("New Oriental Education & Technology Group, Inc. Sponsored ADR representing 10 Ordinary Share (Cayman Islands)", False),
        ("MicroSectors FANG  Index -3X Inverse Leveraged ETNs due January 8, 2038", False),
        ("Franklin BSP Realty Trust, Inc. 7.50% Series E Cumulative Redeemable Preferred Stock", False),
        ("Fortress Biotech, Inc. - 9.375% Series A Cumulative Redeemable Perpetual Preferred Stock", False),
        ("Shift4 Payments, Inc. 6.00% Series A Mandatory Convertible Preferred Stock", False),
        ("Structured Products Corp 8.205% CorTS 8.205% Corporate Backed Trust Securities (CorTS)", False),
        ("Federated Hermes Premier Municipal Income Fund", False),
        ("Credit Suisse High Yield Credit Fund Common Stock", False),
        ("BlackRock Municipal 2030 Target Term Trust", False),
        ("Saba Capital Income & Opportunities Fund SBI", False),
        ("BlackRock Investment Quality Municipal Trust Inc. (The)", False),
        ("Uniti Group Inc. - Common Stock", True),
        ("Universal Health Realty Income Trust Common Stock", True)
    ]

    print("Testing is_common_stock() function:")
    print("-" * 70)
    for name, expected in test_cases:
        result = is_common_stock(name)
        status = "✓" if result == expected else "✗"
        print(f"{status} {name:<50} -> {result}")
    print("-" * 70)
    print()

    # Fetch actual stock data
    start = time.perf_counter()
    result = fetch_all_stocks()
    print(result.tail())
    print(f"Execution time: {(time.perf_counter() - start):.2f} seconds")
