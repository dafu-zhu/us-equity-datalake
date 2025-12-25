import requests
import time

from utils.mapping import symbol_cik_mapping

# --- Configuration ---
# SEC requires a proper User-Agent (AppName/Email)
headers = {'User-Agent': 'ResearchApp/your.email@domain.com'}

# diverse_tickers = ["AAPL", "JPM", "PFE", "XOM", "WMT", "BA"]
# AAPL (Tech), JPM (Banking), PFE (Pharma), XOM (Energy), WMT (Retail), BA (Aerospace)
target_tickers = ["AAPL", "JPM", "PFE", "XOM", "WMT", "BA"]

# --- Step 1: Get the Ticker -> CIK Mapping ---
print("Fetching Master Ticker-CIK Index...")
ticker_mapping_url = "https://www.sec.gov/files/company_tickers.json"
response = requests.get(ticker_mapping_url, headers=headers)

ticker_to_cik = symbol_cik_mapping()

# --- Step 2: Fetch Facts & Calculate Intersection (Common Fields) ---
all_company_fields = []  # Store each company's field set

print(f"\nProcessing {len(target_tickers)} companies...")
print("-" * 50)
print(f"{'Ticker':<8} {'Industry Hint':<15} {'Total Fields'}")
print("-" * 50)

industry_map = {
    "AAPL": "Tech", "JPM": "Banking", "PFE": "Pharma",
    "XOM": "Energy", "WMT": "Retail", "BA": "Aerospace"
}

for ticker in target_tickers:
    # Get CIK from our mapping
    cik = ticker_to_cik.get(ticker)
    cik_filled = str(cik).zfill(10)

    if not cik:
        print(f"Skipping {ticker} (CIK not found)")
        continue

    # Fetch Company Facts
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_filled}.json"
    resp = requests.get(facts_url, headers=headers)

    if resp.status_code == 200:
        facts_data = resp.json()

        # safely access us-gaap keys
        company_fields = set(facts_data.get('facts', {}).get('us-gaap', {}).keys())
        count = len(company_fields)

        # Store this company's field set
        all_company_fields.append(company_fields)

        print(f"{ticker:<8} {industry_map[ticker]:<15} {count}")
    else:
        print(f"{ticker:<8} Error {resp.status_code}")

    # Respect SEC rate limit (10 req/sec max, but safer to sleep slightly)
    time.sleep(0.2)

# --- Step 3: Calculate Common Fields (Intersection) ---
if all_company_fields:
    common_fields = set.intersection(*all_company_fields)
else:
    common_fields = set()

print("-" * 50)
print(f"Common US-GAAP Fields (in ALL companies): {len(common_fields)}")

# --- Step 4: Save to File ---
from pathlib import Path

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