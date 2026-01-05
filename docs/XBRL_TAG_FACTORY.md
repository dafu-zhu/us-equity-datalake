# XBRL Tag Factory

## Overview

The XBRL Tag Factory is a data discovery and mapping tool that solves the fundamental challenge of extracting structured financial data from SEC EDGAR filings: **identifying which of the 10,000+ possible XBRL tags companies actually use and mapping them to your canonical data model**.

**File:** `scripts/xbrl_tag_factory.py`

**Purpose:**
1. Survey real-world XBRL tag usage across a stratified sample of company filings
2. Rank tags by frequency and importance
3. Generate intelligent mapping suggestions from discovered tags to your canonical concepts
4. Provide a foundation for building robust fundamental data extractors

## The Problem

SEC EDGAR filings use XBRL (eXtensible Business Reporting Language) to structure financial data. However:

- **Tag proliferation:** 10,000+ tags exist (US-GAAP standard taxonomy + thousands of company-specific extensions)
- **Synonym explosion:** "Revenue" alone has 50+ variants (`Revenues`, `SalesRevenueNet`, `RevenueFromContractWithCustomerExcludingAssessedTax`, etc.)
- **Company extensions:** Each company can define custom tags (e.g., `aapl:IPhoneRevenue`, `msft:AzureRevenue`)
- **No canonical mapping:** Your `fundamental.xlsx` defines ~50 concepts, but which XBRL tags map to each?

**Without this tool:** You'd either:
- Manually review thousands of filings (weeks of work)
- Hardcode a few tags and miss 70%+ of data
- Build a brittle extractor that breaks on edge cases

**With this tool:** You discover which tags actually matter and get AI-assisted mapping suggestions.

## Architecture

### Two-Phase Workflow

```
Phase 1: Harvest                    Phase 2: Suggest
┌─────────────────┐                ┌──────────────────┐
│ SEC EDGAR API   │                │  tag_usage.csv   │
│ (300 companies) │                │  tag_labels.csv  │
└────────┬────────┘                │  fundamental.xlsx│
         │                          └────────┬─────────┘
         ▼                                   │
┌─────────────────┐                         ▼
│ Download 600    │                ┌──────────────────┐
│ XBRL Instances  │                │ Fuzzy Matching + │
└────────┬────────┘                │ Keyword Rules    │
         │                          └────────┬─────────┘
         ▼                                   │
┌─────────────────┐                         ▼
│ Parse Tags      │                ┌──────────────────┐
│ (XML + iXBRL)   │                │ mapping_         │
└────────┬────────┘                │ suggestions.yaml │
         │                          └──────────────────┘
         ▼
┌─────────────────┐
│ Count & Label   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ tag_usage.csv   │
│ tag_labels.csv  │
└─────────────────┘
```

### Key Components

#### 1. SecClient (Rate-Limited HTTP Client)

```python
class SecClient:
    """
    Minimal SEC client with:
    - Required User-Agent header
    - Rate limiting (5 requests/sec)
    - Disk caching for all GET requests
    """
```

**Features:**
- Respects SEC's required User-Agent policy
- Automatic rate limiting (configurable, default 5 req/sec)
- Persistent disk cache (filings are immutable once published)
- Handles both `data.sec.gov` and `www.sec.gov/Archives` endpoints

#### 2. Company Sampling

```python
def stratified_sample_companies(sec, companies, n=300):
    """
    Stratify by:
    - SIC bucket (2-digit industry code)
    - Size tier (small/mid/large by Assets quantiles)

    Ensures diverse coverage across industries and company sizes.
    """
```

**Why stratification?**
- **Industry diversity:** Healthcare companies use different tags than tech companies
- **Size diversity:** Small caps have simpler filings than large caps
- **Avoids bias:** Random sampling might oversample dominant sectors

**Data sources:**
- `company_tickers.json`: Ticker-to-CIK mapping
- `submissions/CIK{cik}.json`: Filing history + SIC code
- `companyfacts/CIK{cik}.json`: Latest `us-gaap:Assets` for size proxy

#### 3. Filing Discovery & Download

**Challenge:** SEC filings come in two formats:

| Format | Period | File Type | Fact Encoding |
|--------|--------|-----------|---------------|
| Traditional XBRL | Pre-2019 | `.xml` | `<us-gaap:Assets>1000000</us-gaap:Assets>` |
| Inline XBRL | 2019+ | `.htm/.html` | `<ix:nonFraction name="us-gaap:Assets">1M</ix:nonFraction>` |

**Solution:** Multi-method instance file detection:

```python
def download_best_instance(sec, filing):
    # Method 1: FilingSummary.xml (most reliable)
    inst_name = _find_instance_via_filing_summary(sec, filing, idx)

    # Method 2-5: Heuristic fallbacks for edge cases
    candidates = find_instance_candidates(idx)
```

**FilingSummary.xml parsing** (5 fallback methods):

1. **Old format:** `<Instance>filename.xml</Instance>`
2. **New format (2019+):** First `.htm` in `<InputFiles>` without taxonomy suffix
3. **MyReports section:** First `<HtmlFileName>` matching ticker-date pattern
4. **Regex scan:** Find `ticker-YYYYMMDD.htm` pattern anywhere
5. **Legacy keyword:** Search for "INSTANCE" keyword near `<Href>` tags

**Why so many methods?** SEC filing structures evolved over 15+ years. Resilience requires handling all variants.

#### 4. XBRL Parsing (Dual Format Support)

**Traditional XBRL (.xml):**
```python
# Facts are XML element tags
for el in root.iter():
    qn = etree.QName(el.tag)
    tag = f"{prefix}:{local}"  # "us-gaap:Assets"
    tags_in_filing.add(tag)
```

**Inline XBRL (.htm/.html):**
```python
# Facts are in 'name' attributes embedded in HTML
for el in root.iter():
    name_attr = el.get('name')  # "us-gaap:Assets"
    if name_attr and ':' in name_attr:
        tags_in_filing.add(name_attr)
```

**Filtering logic:**
```python
# Ignore infrastructure namespaces
IGNORE_NS_URIS = {
    "http://www.xbrl.org/2003/instance",  # xbrli
    "http://www.xbrl.org/2003/linkbase",  # link
    "http://www.w3.org/1999/xhtml",       # html
}

# Ignore dimensional metadata
if local.endswith('.domain') or local.endswith('.typed'):
    continue  # xbrldi:dimension stuff

# Ignore axes/members (not financial facts)
if any(x in tag.lower() for x in ["axis", "member", "table"]):
    continue
```

**Label extraction:**
```python
def tag_to_label(tag="us-gaap:NetIncomeLoss"):
    # us-gaap:NetIncomeLoss → "Net Income Loss"
    local = tag.split(':')[1]
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', local)
    return spaced  # "Net Income Loss"
```

#### 5. Tag Counting & Aggregation

```python
# Count presence per filing (NOT per-fact occurrence)
for tag in tags_in_filing:
    tag_usage[tag] += 1  # How many filings contain this tag?

# Preserve best label seen across all filings
if info.label and tag not in tag_labels:
    tag_labels[tag] = info.label
```

**Why count presence vs. occurrence?**
- A tag appearing 50 times in 1 filing is less important than appearing once in 50 filings
- Presence indicates "how many companies report this concept"

**Outputs:**
- `tag_usage.csv`: Each tag's frequency across all filings
- `tag_labels.csv`: Human-readable labels for each tag

#### 6. Mapping Suggestion Engine

**Scoring algorithm:**
```python
def score_tag_for_concept(concept="revenue", tag="us-gaap:Revenues", label="Revenues"):
    # 1. Label similarity (75% weight)
    sim_lbl = fuzz.token_set_ratio("revenue", "Revenues")  # 100

    # 2. Tag similarity (25% weight)
    sim_tag = fuzz.token_set_ratio("revenue", "us-gaap:revenues")  # 85

    score = int(0.75 * sim_lbl + 0.25 * sim_tag)

    # 3. Keyword boosts (+10 per match)
    if all(w in lbl_l for w in ["revenue"]):
        score += 10

    # 4. Noise penalties (-30)
    if "axis" in tag or "member" in tag or "table" in tag:
        score -= 30

    return score
```

**Keyword rules** (high-precision heuristics):
```python
KEYWORD_RULES = {
    "revenue": [["revenue"], ["sales"]],
    "net_income": [["net", "income"], ["profit", "loss"]],
    "operating_income": [["operating", "income"]],
    "gross_profit": [["gross", "profit"]],
    "cost_of_revenue": [["cost", "revenue"], ["cost", "goods"]],
    "assets": [["assets"]],
    "cash_and_cash_equivalents": [["cash", "equivalents"]],
    # ... extend based on your fundamental.xlsx
}
```

**Output format:**
```yaml
revenue:
  gaap_candidates:
    - us-gaap:Revenues                              # 87% of filings
    - us-gaap:SalesRevenueNet                       # 45% of filings
    - us-gaap:RevenueFromContractWithCustomer       # 32% of filings
  extension_candidates:
    - aapl:ProductRevenue                           # Apple-specific
    - msft:ProductAndServiceRevenue                 # Microsoft-specific

net_income:
  gaap_candidates:
    - us-gaap:NetIncomeLoss                         # 95% of filings
    - us-gaap:ProfitLoss                            # 12% of filings (IFRS)
  extension_candidates:
    - brk:NetEarnings                               # Berkshire Hathaway
```

## Usage

### Prerequisites

**Python dependencies:**
```bash
pip install pandas requests pyyaml rapidfuzz lxml openpyxl arelle-release
```

**Required inputs:**
- `fundamental.xlsx`: Your canonical concept definitions (must have "fields" sheet with "Concept" column)
- SEC User-Agent: Email or company name (required by SEC)

### Command 1: Harvest

**Purpose:** Download filings, extract tags, generate usage statistics.

```bash
python scripts/xbrl_tag_factory.py harvest \
  --user_agent "YourName your@email.com" \
  --concepts_xlsx ./data/xbrl/fundamental.xlsx \
  --n_companies 300 \
  --start_year 2019 \
  --end_year 2025 \
  --forms 10-K 10-Q \
  --cache_dir ./data/xbrl/edgar_cache \
  --out_tag_usage ./data/xbrl/tag_usage.csv \
  --out_tag_labels ./data/xbrl/tag_labels.csv \
  --max_filings_total 600 \
  --seed 7
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--user_agent` | **Yes** | - | Email or identifier (SEC requirement) |
| `--concepts_xlsx` | **Yes** | - | Path to fundamental.xlsx |
| `--n_companies` | No | 300 | Number of companies to sample |
| `--start_year` | No | 2019 | Earliest filing year |
| `--end_year` | No | 2025 | Latest filing year |
| `--forms` | No | 10-K 10-Q | Filing types to process |
| `--cache_dir` | No | ./data/xbrl/edgar_cache | Download cache location |
| `--out_tag_usage` | No | ./data/xbrl/tag_usage.csv | Tag frequency output |
| `--out_tag_labels` | No | ./data/xbrl/tag_labels.csv | Tag labels output |
| `--max_filings_total` | No | 600 | Cap on total filings processed |
| `--seed` | No | 7 | Random seed for reproducibility |

**Expected runtime:** 2-6 hours (depending on network speed and `max_filings_total`)

**Output files:**

`tag_usage.csv`:
```csv
tag,usage_count,is_extension
us-gaap:Assets,580,False
us-gaap:Revenues,520,False
us-gaap:NetIncomeLoss,595,False
aapl:IPhoneRevenue,4,True
...
```

`tag_labels.csv`:
```csv
tag,label
us-gaap:Assets,Assets
us-gaap:Revenues,Revenues
us-gaap:NetIncomeLoss,Net Income (Loss)
aapl:IPhoneRevenue,iPhone Revenue
...
```

**Console output:**
```
[info] Loaded 48 canonical concepts from ./data/xbrl/fundamental.xlsx
[info] Loaded 8234 tickers from SEC list
[info] Sampled 300 companies (stratified by SIC bucket + Assets proxy)
[info] Will process up to 600 filings total
[debug] no_instance=12 parse_fail=3 zero_tags=0 processed=25
[debug] no_instance=28 parse_fail=7 zero_tags=1 processed=50
...
[info] Processing complete: no_instance=45 parse_fail=12 zero_tags=2 processed=541
[done] wrote ./data/xbrl/tag_usage.csv (8234 tags)
[done] wrote ./data/xbrl/tag_labels.csv
```

### Command 2: Suggest

**Purpose:** Generate intelligent mapping suggestions from discovered tags to your canonical concepts.

```bash
python scripts/xbrl_tag_factory.py suggest \
  --concepts_xlsx ./data/xbrl/fundamental.xlsx \
  --tag_usage ./data/xbrl/tag_usage.csv \
  --tag_labels ./data/xbrl/tag_labels.csv \
  --out_suggestions ./data/config/mapping_suggestions.yaml
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--concepts_xlsx` | **Yes** | - | Path to fundamental.xlsx |
| `--tag_usage` | **Yes** | - | Output from harvest command |
| `--tag_labels` | **Yes** | - | Output from harvest command |
| `--out_suggestions` | No | ./data/config/mapping_suggestions.yaml | Mapping output file |

**Expected runtime:** < 10 seconds

**Output file:**

`mapping_suggestions.yaml`:
```yaml
revenue:
  gaap_candidates:
  - us-gaap:Revenues
  - us-gaap:SalesRevenueNet
  - us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
  extension_candidates:
  - aapl:ProductRevenue
  - msft:ProductAndServiceRevenue

net_income:
  gaap_candidates:
  - us-gaap:NetIncomeLoss
  - us-gaap:ProfitLoss
  extension_candidates:
  - brk:NetEarnings

assets:
  gaap_candidates:
  - us-gaap:Assets
  - us-gaap:AssetsCurrent
  extension_candidates: []
...
```

### Workflow Integration

**Step 1: Initial discovery (one-time)**
```bash
# Harvest tags from 300 companies (2-6 hours)
python scripts/xbrl_tag_factory.py harvest \
  --user_agent "me@example.com" \
  --concepts_xlsx ./data/xbrl/fundamental.xlsx \
  --n_companies 300

# Generate suggestions (< 10 seconds)
python scripts/xbrl_tag_factory.py suggest \
  --concepts_xlsx ./data/xbrl/fundamental.xlsx \
  --tag_usage ./data/xbrl/tag_usage.csv \
  --tag_labels ./data/xbrl/tag_labels.csv
```

**Step 2: Human review**
- Open `mapping_suggestions.yaml`
- Review top candidates for each concept
- Validate against actual filings (spot-check 5-10 companies)
- Adjust/approve mappings

**Step 3: Create approved mappings**
```yaml
# data/config/approved_mappings.yaml (human-curated)
revenue:
  primary: us-gaap:Revenues
  fallbacks:
    - us-gaap:SalesRevenueNet
    - us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
  extension_rules:
    - pattern: ".*Revenue.*"
      exclude: ["RevenueBySegment", "RevenueAxis"]

net_income:
  primary: us-gaap:NetIncomeLoss
  fallbacks:
    - us-gaap:ProfitLoss
  derivation:  # Optional: calculate from other fields
    formula: "revenue - operating_expenses - taxes"
    required: [revenue, operating_expenses, taxes]
```

**Step 4: Use in fundamental collector**
```python
# src/collection/fundamental_collector.py
import yaml

with open("data/config/approved_mappings.yaml") as f:
    MAPPINGS = yaml.safe_load(f)

def extract_concept(facts, concept="revenue"):
    mapping = MAPPINGS[concept]

    # Try primary tag
    if mapping["primary"] in facts:
        return facts[mapping["primary"]]

    # Try fallbacks
    for tag in mapping.get("fallbacks", []):
        if tag in facts:
            return facts[tag]

    # Try extension pattern matching
    for rule in mapping.get("extension_rules", []):
        for tag, value in facts.items():
            if re.match(rule["pattern"], tag):
                if not any(exc in tag for exc in rule.get("exclude", [])):
                    return value

    # Try derivation
    if "derivation" in mapping:
        # ... implement formula calculation
        pass

    return None
```

## Data Quality & Validation

### Coverage Metrics

After running harvest, analyze coverage:

```python
import pandas as pd

usage = pd.read_csv("data/xbrl/tag_usage.csv")

# Tag frequency distribution
print(usage["usage_count"].describe())
# Output:
# count    8234.00
# mean       45.23
# std       112.45
# min         1.00
# 25%         2.00
# 50%         8.00
# 75%        35.00
# max       595.00

# Extension vs. GAAP breakdown
print(usage["is_extension"].value_counts())
# Output:
# False    6234  (75.7% are standard GAAP tags)
# True     2000  (24.3% are company extensions)

# Top 20 most common tags
print(usage.nlargest(20, "usage_count"))
```

### Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Survivorship bias** | Sample only includes active companies | Acceptable for forward-looking data lake |
| **Filing format evolution** | Pre-2009 filings use different structures | Scope limited to 2009+ (per CLAUDE.md) |
| **Small cap coverage** | ~25% of small caps lack recent filings | Use yfinance as fallback for pricing data |
| **Concept ambiguity** | "Revenue" might mean gross or net revenue | Human review required; add documentation |
| **Extension proliferation** | 2000+ company-specific tags discovered | Pattern matching + manual rules for common extensions |

### Validation Checks

**Post-harvest validation:**
```python
# Check 1: Minimum filings processed
assert processed >= 400, f"Only processed {processed} filings (target: 600)"

# Check 2: Core GAAP tags present
REQUIRED_TAGS = [
    "us-gaap:Assets",
    "us-gaap:Liabilities",
    "us-gaap:NetIncomeLoss",
    "us-gaap:Revenues",
]
for tag in REQUIRED_TAGS:
    assert tag in tag_usage, f"Missing critical tag: {tag}"

# Check 3: Label coverage
label_coverage = len(tag_labels) / len(tag_usage)
assert label_coverage >= 0.60, f"Only {label_coverage:.0%} tags have labels"
```

**Post-suggest validation:**
```python
# Check 4: All concepts mapped
for concept in concepts:
    assert concept in suggestions, f"No suggestions for: {concept}"
    assert len(suggestions[concept]["gaap_candidates"]) >= 1, \
        f"No GAAP candidates for: {concept}"
```

## Performance & Scalability

### Resource Requirements

**Harvest phase:**
- **Network:** 600 filings × ~500 KB avg = ~300 MB download
- **Disk cache:** ~1-2 GB (includes index.json, FilingSummary.xml, instance files)
- **Memory:** < 2 GB (streaming parser)
- **CPU:** Low (I/O bound, not compute bound)
- **Runtime:** 2-6 hours (dominated by network I/O + rate limiting)

**Suggest phase:**
- **Memory:** < 500 MB
- **CPU:** Moderate (fuzzy string matching on ~8000 tags × 50 concepts)
- **Runtime:** < 10 seconds

### Scaling Strategies

**To process more filings (e.g., 5000 filings):**

1. **Increase rate limit** (if you have SEC approval):
   ```python
   sec = SecClient(user_agent=..., max_rps=10.0)  # Default: 5.0
   ```

2. **Parallelize downloads** (requires more sophisticated rate limiting):
   ```python
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=5) as executor:
       futures = [executor.submit(download_best_instance, sec, f)
                  for f in filings]
   ```

3. **Incremental updates:**
   ```bash
   # Initial harvest (one-time)
   python scripts/xbrl_tag_factory.py harvest --max_filings_total 600

   # Later: add more recent filings
   python scripts/xbrl_tag_factory.py harvest \
     --start_year 2024 --end_year 2025 \
     --max_filings_total 200 \
     --out_tag_usage ./data/xbrl/tag_usage_2024.csv

   # Merge results
   pd.concat([
       pd.read_csv("data/xbrl/tag_usage.csv"),
       pd.read_csv("data/xbrl/tag_usage_2024.csv")
   ]).groupby("tag", as_index=False)["usage_count"].sum()
   ```

## Troubleshooting

### Common Issues

#### Issue: "no_instance" count is very high (> 30%)

**Symptoms:**
```
[info] Processing complete: no_instance=280 parse_fail=5 zero_tags=0 processed=315
```

**Causes:**
- Companies with non-XBRL filings (legacy ASCII format)
- Filings before 2009 (pre-XBRL mandate era)
- Data API failures

**Solutions:**
```python
# 1. Check if filing years are too early
--start_year 2019  # XBRL became reliable post-2009, inline XBRL post-2019

# 2. Enable verbose logging for first 10 filings
if processed < 10:
    print(f"[debug] CIK={f.cik} ACC={f.accession} instance={inst_path}")

# 3. Manually inspect a failed filing
# Visit: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K
```

#### Issue: "parse_fail" count is high (> 10%)

**Symptoms:**
```
[info] Processing complete: no_instance=45 parse_fail=78 zero_tags=0 processed=477
```

**Causes:**
- Malformed XML/HTML in instance files
- Non-standard inline XBRL extensions
- Encoding issues (non-UTF8 files)

**Solutions:**
```python
# 1. Add error logging to parse function
try:
    tags_in_filing, infos = parse_instance_tags_xml(inst_path)
except Exception as e:
    print(f"[error] Parse failed: {inst_path}")
    print(f"  Error: {type(e).__name__}: {e}")
    # Continue and analyze errors

# 2. Try Arelle parser as fallback (slower but more robust)
try:
    tags_in_filing, infos = parse_instance_tags_xml(inst_path)
except Exception:
    tags_in_filing, infos = parse_instance_tags_arelle(mm, inst_path)
```

#### Issue: All concept suggestions have empty candidates

**Symptoms:**
```yaml
revenue:
  gaap_candidates: []
  extension_candidates: []
```

**Causes:**
- `tag_usage.csv` is empty or malformed
- Concept names in `fundamental.xlsx` don't match any tags
- Fuzzy matching threshold too high

**Solutions:**
```python
# 1. Verify tag_usage.csv has data
usage = pd.read_csv("data/xbrl/tag_usage.csv")
assert len(usage) > 0, "tag_usage.csv is empty!"
print(f"Loaded {len(usage)} tags")

# 2. Lower score threshold
suggest_mappings(..., min_score=50)  # Default: 65

# 3. Check concept names
concepts = load_concepts_from_xlsx("data/xbrl/fundamental.xlsx")
print(concepts)  # Should be: ['revenue', 'net_income', ...]
```

#### Issue: Rate limit errors from SEC

**Symptoms:**
```
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**Solutions:**
```python
# 1. Reduce request rate
sec = SecClient(..., max_rps=3.0)  # Default: 5.0

# 2. Add exponential backoff
def get_bytes_with_retry(self, url, cache_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            return self.get_bytes(url, cache_key)
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
            else:
                raise
```

## Integration with Data Lake Pipeline

### Position in Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     US Equity Data Lake                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 0: Foundation                                         │
│    ├─ [ ] Ticker universe (SEC EDGAR + NASDAQ FTP)          │
│    ├─ [✓] XBRL Tag Factory (THIS TOOL) ◄─── You are here   │
│    └─ [ ] Schema definitions                                 │
│                                                               │
│  Phase 1: Reference Data                                     │
│    ├─ [ ] Corporate actions collector                        │
│    ├─ [ ] Index constituents collector                       │
│    └─ [ ] Data validation framework                          │
│                                                               │
│  Phase 2: Fundamental Data (Milestone 14-15) ◄─── Uses this │
│    ├─ [ ] SEC EDGAR fundamental collector                    │
│    │    └─── Reads: approved_mappings.yaml (from this tool) │
│    ├─ [ ] Quarterly data backfill                            │
│    └─ [ ] Data quality checks                                │
│                                                               │
│  Phase 3: Tick Data                                          │
│    ├─ [ ] Daily ticks (yfinance)                             │
│    └─ [ ] Minute ticks (Alpaca)                              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### File Dependencies

**Inputs (you provide):**
```
data/xbrl/fundamental.xlsx
  └─ Sheet: "fields"
      └─ Column: "Concept" (e.g., revenue, net_income, assets, ...)
```

**Outputs (this tool generates):**
```
data/xbrl/
  ├─ edgar_cache/                          # Cached downloads (gitignore)
  │   ├─ submissions_{cik}.json
  │   ├─ companyfacts_{cik}.json
  │   ├─ index_{cik}_{accession}.json
  │   └─ instance_{cik}_{accession}_*.{xml,htm}
  ├─ tag_usage.csv                         # Tag frequency stats
  ├─ tag_labels.csv                        # Human-readable labels
  └─ mapping_suggestions.yaml              # AI-generated mappings

data/config/
  └─ approved_mappings.yaml                # Human-reviewed (you create)
```

**Consumed by:**
```
src/collection/fundamental_collector.py   # Uses approved_mappings.yaml
src/validation/fundamental_validator.py   # References tag_labels.csv
```

### Next Steps After Running This Tool

1. **Review suggestions** (`mapping_suggestions.yaml`)
2. **Spot-check 10 companies** (verify mappings extract correct values)
3. **Create `approved_mappings.yaml`** with derivation rules
4. **Implement fundamental collector** (Milestone 14-15):
   ```python
   # src/collection/fundamental_collector.py
   def extract_fundamental_data(cik, filing_date, approved_mappings):
       # Download companyfacts JSON
       # Apply approved_mappings to extract canonical concepts
       # Validate against balance sheet equation
       # Return structured fundamental data
   ```
5. **Backfill fundamentals** (2009-2025, quarterly)
6. **Generate coverage report** (% of companies with each concept)

## References

### SEC EDGAR API Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| Company Tickers | Ticker-to-CIK mapping | `https://www.sec.gov/files/company_tickers.json` |
| Submissions | Filing history | `https://data.sec.gov/submissions/CIK0000320193.json` |
| Company Facts | Aggregated XBRL facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json` |
| Filing Index | Document list | `https://www.sec.gov/Archives/edgar/data/320193/0000320193-23-000077/index.json` |
| Instance File | XBRL document | `https://www.sec.gov/Archives/edgar/data/320193/0000320193-23-000077/aapl-20230930.htm` |

### XBRL Resources

- **US-GAAP Taxonomy:** https://xbrl.us/data-rule/dqc_0015-negative-values/
- **SEC EDGAR Filer Manual:** https://www.sec.gov/info/edgar/edgarfm-vol2-v66.pdf
- **Inline XBRL Spec:** https://www.xbrl.org/specification/inlinexbrl-1.1/rec-2013-11-18/inlinexbrl-1.1-rec-2013-11-18.html
- **Arelle Documentation:** https://arelle.org/arelle/

### Related Documentation

- `docs/SPECIFICATION.md`: Overall data lake specification
- `docs/DEVELOPMENT.md`: Development environment setup
- `docs/API.md`: Query API design (consumes approved_mappings.yaml)
- `CLAUDE.md`: Project objectives and roadmap (Milestone 14-15)

## Appendix: Example Output

### Sample tag_usage.csv (top 30 tags)

```csv
tag,usage_count,is_extension
us-gaap:Assets,595,False
us-gaap:NetIncomeLoss,594,False
us-gaap:Liabilities,593,False
us-gaap:StockholdersEquity,590,False
us-gaap:Revenues,520,False
us-gaap:CostOfRevenue,485,False
us-gaap:OperatingIncomeLoss,478,False
us-gaap:CashAndCashEquivalentsAtCarryingValue,472,False
us-gaap:AssetsCurrent,468,False
us-gaap:LiabilitiesCurrent,465,False
us-gaap:OperatingExpenses,455,False
us-gaap:GrossProfit,448,False
us-gaap:IncomeTaxExpenseBenefit,445,False
us-gaap:RetainedEarningsAccumulatedDeficit,442,False
us-gaap:CommonStockSharesOutstanding,438,False
us-gaap:EarningsPerShareBasic,435,False
us-gaap:EarningsPerShareDiluted,434,False
us-gaap:PropertyPlantAndEquipmentNet,428,False
us-gaap:IntangibleAssetsNetExcludingGoodwill,385,False
us-gaap:Goodwill,378,False
us-gaap:AccountsReceivableNetCurrent,375,False
us-gaap:InventoryNet,342,False
us-gaap:LongTermDebt,338,False
us-gaap:NetCashProvidedByUsedInOperatingActivities,335,False
us-gaap:NetCashProvidedByUsedInInvestingActivities,334,False
us-gaap:NetCashProvidedByUsedInFinancingActivities,333,False
us-gaap:DividendsCommonStock,298,False
us-gaap:ResearchAndDevelopmentExpense,287,False
aapl:ProductRevenue,4,True
msft:AzureRevenue,4,True
```

### Sample mapping_suggestions.yaml (excerpt)

```yaml
revenue:
  gaap_candidates:
  - us-gaap:Revenues
  - us-gaap:SalesRevenueNet
  - us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
  - us-gaap:SalesRevenueGoodsNet
  extension_candidates:
  - aapl:ProductRevenue
  - msft:ProductAndServiceRevenue

net_income:
  gaap_candidates:
  - us-gaap:NetIncomeLoss
  - us-gaap:ProfitLoss
  - us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic
  extension_candidates: []

assets:
  gaap_candidates:
  - us-gaap:Assets
  extension_candidates: []

liabilities:
  gaap_candidates:
  - us-gaap:Liabilities
  - us-gaap:LiabilitiesAndStockholdersEquity
  extension_candidates: []

stockholders_equity:
  gaap_candidates:
  - us-gaap:StockholdersEquity
  - us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
  extension_candidates: []

cash_and_cash_equivalents:
  gaap_candidates:
  - us-gaap:CashAndCashEquivalentsAtCarryingValue
  - us-gaap:Cash
  extension_candidates: []

gross_profit:
  gaap_candidates:
  - us-gaap:GrossProfit
  extension_candidates: []

operating_income:
  gaap_candidates:
  - us-gaap:OperatingIncomeLoss
  extension_candidates: []

earnings_per_share:
  gaap_candidates:
  - us-gaap:EarningsPerShareBasic
  - us-gaap:EarningsPerShareDiluted
  extension_candidates: []
```