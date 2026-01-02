"""
Symbol Name Change Examples for CRSPDailyTicks

Demonstrates how CRSPDailyTicks handles ticker symbol changes (e.g., FB -> META)
Compares two modes: auto_resolve vs exact symbol matching
"""

from collection.crsp_ticks import CRSPDailyTicks

# Initialize the data collector
crsp = CRSPDailyTicks()

print("=" * 80)
print("SYMBOL NAME CHANGE HANDLING EXAMPLES")
print("=" * 80)
print("\nBackground: Facebook changed its ticker from 'FB' to 'META' on 2022-06-09")
print("The underlying company/security is the same, just the ticker symbol changed.\n")

# ============================================================================
# Example 1: AUTO_RESOLVE MODE (default, smart resolution)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 1: AUTO_RESOLVE MODE (auto_resolve=True)")
print("=" * 80)
print("Behavior: Like Alpaca's asof parameter - finds the underlying security entity")
print("even when symbol names have changed.\n")

print("-" * 80)
print("Case 1a: Query 'META' for dates BEFORE the rename (when it was 'FB')")
print("-" * 80)
print("Code: crsp.get_daily('META', '2021-12-31', auto_resolve=True)\n")

try:
    result_1a = crsp.get_daily('META', '2021-12-31', auto_resolve=True)
    if result_1a:  # Check if result is not empty
        print("Result: SUCCESS ✓")
        print(f"  - Date: {result_1a['timestamp']}")
        print(f"  - Close: ${result_1a['close']:.2f}")
        print(f"  - Volume: {result_1a['volume']:,}")
        print("\nExplanation:")
        print("  1. 'META' was not active on 2021-12-31 (exact match fails)")
        print("  2. SecurityMaster finds security that EVER used 'META' → security_id=X")
        print("  3. Checks if security_id=X was active on 2021-12-31 → YES (as 'FB')")
        print("  4. Fetches data using the underlying permno for that security")
        print("  5. Returns FB's price data from 2021-12-31")
    else:
        print("Result: NO DATA FOUND")
        print("  (No trading data available for this date)")
except ValueError as e:
    print(f"Error: {e}")

print("\n" + "-" * 80)
print("Case 1b: Query 'FB' for dates AFTER the rename (when it became 'META')")
print("-" * 80)
print("Code: crsp.get_daily('FB', '2023-12-29', auto_resolve=True)\n")

try:
    result_1b = crsp.get_daily('FB', '2023-12-29', auto_resolve=True)
    if result_1b:  # Check if result is not empty
        print("Result: SUCCESS ✓")
        print(f"  - Date: {result_1b['timestamp']}")
        print(f"  - Close: ${result_1b['close']:.2f}")
        print(f"  - Volume: {result_1b['volume']:,}")
        print("\nExplanation:")
        print("  1. 'FB' was not active on 2023-12-29 (exact match fails)")
        print("  2. SecurityMaster finds security that EVER used 'FB' → security_id=X")
        print("  3. Checks if security_id=X was active on 2023-12-29 → YES (as 'META')")
        print("  4. Fetches data using the underlying permno")
        print("  5. Returns META's price data from 2023-12-29")
    else:
        print("Result: NO DATA FOUND")
        print("  (No trading data available for this date)")
except ValueError as e:
    print(f"Error: {e}")

print("\n" + "-" * 80)
print("Case 1c: Query date range spanning the rename")
print("-" * 80)
print("Code: crsp.get_daily_range('META', '2022-06-01', '2022-06-15', auto_resolve=True)\n")

try:
    result_1c = crsp.get_daily_range('META', '2022-06-01', '2022-06-15', auto_resolve=True)
    if result_1c:  # Check if result list is not empty
        print(f"Result: SUCCESS ✓ - Retrieved {len(result_1c)} trading days")
        print("\nSample data:")
        for i, day in enumerate(result_1c[:3]):
            print(f"  {day['timestamp']}: Close=${day['close']:.2f}, Vol={day['volume']:,}")
        if len(result_1c) > 6:
            print("  ...")
        for i, day in enumerate(result_1c[-3:]):
            print(f"  {day['timestamp']}: Close=${day['close']:.2f}, Vol={day['volume']:,}")

        print("\nExplanation:")
        print("  1. Resolves 'META' using end_date (2022-06-15) → finds security_id=X")
        print("  2. Fetches ALL data for security_id=X in the date range")
        print("  3. Returns continuous data across the name change (FB→META)")
        print("  4. Data for 2022-06-01 to 2022-06-08 was labeled 'FB' in CRSP")
        print("  5. Data for 2022-06-09 onwards was labeled 'META' in CRSP")
        print("  6. Both are returned because they represent the same underlying security")
    else:
        print("Result: NO DATA FOUND")
        print("  (No trading data available for this date range)")
except ValueError as e:
    print(f"Error: {e}")


# ============================================================================
# Example 2: EXACT MATCH MODE (auto_resolve=False)
# ============================================================================
print("\n\n" + "=" * 80)
print("EXAMPLE 2: EXACT MATCH MODE (auto_resolve=False)")
print("=" * 80)
print("Behavior: Like Alpaca's asof='-' - only returns data for the EXACT symbol")
print("on that date, without looking up name changes.\n")

print("-" * 80)
print("Case 2a: Query 'META' for dates BEFORE the rename")
print("-" * 80)
print("Code: crsp.get_daily('META', '2021-12-31', auto_resolve=False)\n")

try:
    result_2a = crsp.get_daily('META', '2021-12-31', auto_resolve=False)
    if result_2a:
        print("Result: SUCCESS ✓")
        print(f"  - Close: ${result_2a['close']:.2f}")
    else:
        print("Result: NO DATA FOUND")
        print("  (Symbol matched but no trading data available)")
except ValueError as e:
    print(f"Result: FAIL ✗")
    print(f"  Error: {e}")
    print("\nExplanation:")
    print("  1. Looks for exact match: symbol='META' active on 2021-12-31")
    print("  2. No match found (META didn't exist yet)")
    print("  3. auto_resolve=False, so raises ValueError instead of resolving")
    print("  4. Returns nothing (strict symbol matching)")

print("\n" + "-" * 80)
print("Case 2b: Query 'FB' for dates AFTER the rename")
print("-" * 80)
print("Code: crsp.get_daily('FB', '2023-12-29', auto_resolve=False)\n")

try:
    result_2b = crsp.get_daily('FB', '2023-12-29', auto_resolve=False)
    if result_2b:
        print("Result: SUCCESS ✓")
        print(f"  - Close: ${result_2b['close']:.2f}")
    else:
        print("Result: NO DATA FOUND")
        print("  (Symbol matched but no trading data available)")
except ValueError as e:
    print(f"Result: FAIL ✗")
    print(f"  Error: {e}")
    print("\nExplanation:")
    print("  1. Looks for exact match: symbol='FB' active on 2023-12-29")
    print("  2. No match found (FB was renamed to META)")
    print("  3. auto_resolve=False, so raises ValueError")
    print("  4. Returns nothing (doesn't follow the rename)")

print("\n" + "-" * 80)
print("Case 2c: Query 'META' for dates AFTER the rename (should work)")
print("-" * 80)
print("Code: crsp.get_daily('META', '2023-12-29', auto_resolve=False)\n")

try:
    result_2c = crsp.get_daily('META', '2023-12-29', auto_resolve=False)
    if result_2c:
        print("Result: SUCCESS ✓")
        print(f"  - Date: {result_2c['timestamp']}")
        print(f"  - Close: ${result_2c['close']:.2f}")
        print(f"  - Volume: {result_2c['volume']:,}")
        print("\nExplanation:")
        print("  1. Looks for exact match: symbol='META' active on 2023-12-29")
        print("  2. Match found! META is the active symbol")
        print("  3. Returns data for META on that date")
    else:
        print("Result: NO DATA FOUND")
        print("  (Symbol matched but no trading data available)")
except ValueError as e:
    print(f"Error: {e}")


# ============================================================================
# KEY DIFFERENCES SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("KEY DIFFERENCES SUMMARY")
print("=" * 80)

print("""
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Scenario            │ auto_resolve=True        │ auto_resolve=False       │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Query 'META' in     │ ✓ Returns FB data        │ ✗ Error: symbol not      │
│ 2021 (before rename)│   (smart resolution)     │   found on that date     │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Query 'FB' in       │ ✓ Returns META data      │ ✗ Error: symbol not      │
│ 2023 (after rename) │   (smart resolution)     │   found on that date     │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Query 'META' in     │ ✓ Returns META data      │ ✓ Returns META data      │
│ 2023 (correct name) │   (direct match)         │   (direct match)         │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Date range spanning │ ✓ Returns continuous     │ ⚠ Returns partial data   │
│ rename (2022-06)    │   data (FB + META)       │   (only matching dates)  │
└─────────────────────┴──────────────────────────┴──────────────────────────┘

WHEN TO USE EACH MODE:

✓ auto_resolve=True (RECOMMENDED for most use cases):
  - Backtesting strategies that need continuous price history
  - Portfolio analysis across name changes
  - Computing returns over long periods
  - Any analysis where you care about the COMPANY, not the ticker symbol
  - Example: "Get me all data for Meta Platforms, regardless of ticker changes"

✓ auto_resolve=False (Use when you need strict symbol matching):
  - Regulatory compliance requiring exact symbol matching
  - Auditing historical records with specific ticker symbols
  - Reproducing results from legacy systems that don't handle renames
  - When ticker symbol itself has semantic meaning in your analysis
  - Example: "Get me ONLY data labeled as 'FB' at that time, nothing else"

COMPARISON TO ALPACA API:

Alpaca's 'asof' parameter:
  - asof='2023-12-29' → Similar to auto_resolve=True
    Resolves symbol to underlying entity as of that date

  - asof='-' → Similar to auto_resolve=False
    Skips symbol mapping, returns data based on symbol alone

CRSPDailyTicks advantage:
  - More sophisticated: auto_resolve uses temporal distance algorithm
  - Handles edge cases: multiple securities using same symbol at different times
  - Transparent: SecurityMaster shows full symbol history for any security_id
""")

# ============================================================================
# EDGE CASE: Symbol Reuse (Advanced)
# ============================================================================
print("\n" + "=" * 80)
print("EDGE CASE: When Symbol Gets Reused by Different Companies")
print("=" * 80)
print("""
Sometimes a ticker symbol gets reused after one company delists:
  - Company A uses 'XYZ' from 2010-2015 (then delists)
  - Company B uses 'XYZ' from 2018-2024 (different company!)

auto_resolve=True handles this intelligently:
  - Query 'XYZ' in 2012 → Finds Company A (was active that year)
  - Query 'XYZ' in 2020 → Finds Company B (was active that year)
  - Uses temporal distance algorithm to pick the right security

auto_resolve=False:
  - Query 'XYZ' in 2012 → Returns Company A data (exact match)
  - Query 'XYZ' in 2020 → Returns Company B data (exact match)
  - Both work, but you get different companies!
""")

# Close connection
crsp.close()

print("\n" + "=" * 80)
print("END OF EXAMPLES")
print("=" * 80)
