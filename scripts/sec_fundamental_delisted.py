import requests
import pandas as pd
import json

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Bed Bath & Beyond CIK (Delisted)
# CIK must be a string, padded with leading zeros to 10 digits
cik = "0000886158" 

# REQUIRED: SEC requires a User-Agent in the format: "Name email@domain.com"
headers = {
    'User-Agent': 'YourName your.email@example.com' 
}

# ---------------------------------------------------------
# 1. Fetch Company Facts Data
# ---------------------------------------------------------
url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

print(f"Fetching data for CIK: {cik}...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    
    # Verify we have the right company
    company_name = data['entityName']
    print(f"Successfully connected to: {company_name}")
    
    # ---------------------------------------------------------
    # 2. Extract Specific Fundamental (e.g., 'Assets')
    # ---------------------------------------------------------
    # The JSON structure is deep: facts -> us-gaap -> Concept -> units -> USD
    try:
        # 'Assets' is a standard US-GAAP tag
        assets_data = data['facts']['us-gaap']['Assets']['units']['USD']
        
        # Convert to DataFrame for easy viewing
        df = pd.DataFrame(assets_data)
        
        # Filter for 10-K (Annual) and 10-Q (Quarterly) only to clean up view
        df = df[df['form'].isin(['10-K', '10-Q'])].copy()
        
        # Sort by report period (end date of the data)
        df = df.sort_values('end')
        
        # Select relevant columns
        result = df[['end', 'val', 'form', 'fy', 'fp']]
        
        print("\n--- Recent Asset History (USD) ---")
        print(result.tail(10)) # Show the last 10 entries before delisting
        
    except KeyError:
        print("Could not find 'Assets' in the company data.")
        
else:
    print(f"Error: {response.status_code}")
    print("Ensure you updated the User-Agent header with a valid email.")