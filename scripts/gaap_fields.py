#!/usr/bin/env python3
"""
Generate common US-GAAP fields configuration file.

This script analyzes SEC EDGAR fundamental data for a diverse set of companies
and identifies fields that are available across all industries. The result is saved
to data/config/common-gaap-fields.txt for use by the fundamental data collection system.

The script:
1. Fetches company facts from SEC EDGAR API for diverse companies
2. Filters for fields with USD or shares units (quantitative metrics)
3. Calculates the intersection of fields available in ALL companies
4. Saves the common fields to a configuration file
"""

import sys
from pathlib import Path
import requests
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.mapping import symbol_cik_mapping

# SEC EDGAR API Configuration
# SEC requires a User-Agent header in the format: "AppName/Email"
headers = {'User-Agent': 'US-Equity-Datalake/research@example.com'}

# Diverse set of tickers across industries for comprehensive field analysis
# AAPL (Tech), JPM (Banking), PFE (Pharma), XOM (Energy), WMT (Retail), BA (Aerospace)
target_tickers = ["AAPL", "JPM", "PFE", "XOM", "WMT", "BA"]

industry_map = {
    "AAPL": "Tech",
    "JPM": "Banking",
    "PFE": "Pharma",
    "XOM": "Energy",
    "WMT": "Retail",
    "BA": "Aerospace"
}

# Get Ticker -> CIK Mapping
print("Fetching Ticker-CIK Mapping...")
ticker_to_cik = symbol_cik_mapping()

# Fetch Facts & Calculate Intersection
all_company_fields = []

print(f"\nProcessing {len(target_tickers)} companies...")
print("-" * 60)
print(f"{'Ticker':<10} {'Industry':<15} {'Total Fields'}")
print("-" * 60)

for ticker in target_tickers:
    cik = ticker_to_cik.get(ticker)

    if not cik:
        print(f"Skipping {ticker} (CIK not found)")
        continue

    cik_filled = str(cik).zfill(10)
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_filled}.json"

    try:
        resp = requests.get(facts_url, headers=headers, timeout=30)

        if resp.status_code == 200:
            facts_data = resp.json()

            # Only include fields that have 'USD' or 'shares' units (quantitative metrics)
            gaap_data = facts_data.get('facts', {}).get('us-gaap', {})
            company_fields = set()

            for field_name, field_data in gaap_data.items():
                units = field_data.get('units', {})
                if 'USD' in units or 'shares' in units:
                    company_fields.add(field_name)

            count = len(company_fields)
            all_company_fields.append(company_fields)
            print(f"{ticker:<10} {industry_map[ticker]:<15} {count}")
        else:
            print(f"{ticker:<10} Error: HTTP {resp.status_code}")

    except Exception as e:
        print(f"{ticker:<10} Error: {e}")

    # Respect SEC rate limit (10 req/sec max)
    time.sleep(0.2)

# Calculate Common Fields (Intersection)
if all_company_fields:
    common_fields = set.intersection(*all_company_fields)
else:
    common_fields = set()
    print("\nError: No company data was successfully fetched")
    sys.exit(1)

print("-" * 60)
print(f"Common US-GAAP Fields (in ALL companies): {len(common_fields)}")

# Save to Configuration File
output_dir = Path("data/config")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "common-gaap-fields.txt"
with open(output_file, 'w') as f:
    for field in sorted(common_fields):
        f.write(f"{field}\n")

print(f"\nSaved {len(common_fields)} common fields to: {output_file}")
print("\nSample common fields (first 20):")
for field in sorted(common_fields)[:20]:
    print(f"  - {field}")
