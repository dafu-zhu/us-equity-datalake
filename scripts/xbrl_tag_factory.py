#!/usr/bin/env python3
"""
XBRL Tag Factory for SEC EDGAR
- Download filings for a stratified sample of companies
- Parse XBRL instance facts, count tag usage (presence per filing)
- Rank tags by frequency
- Suggest canonical mappings (GAAP-first, label similarity for extensions)
- Use derivations to avoid mapping everything

Usage examples:
  # Harvest tags
  python scripts/xbrl_tag_factory.py harvest \
    --user_agent "abc@example.com" \
    --concepts_xlsx ./data/xbrl/fundamental.xlsx \
    --n_companies 600 --start_year 2010 --end_year 2025 \
    --forms 10-K 10-Q \
    --cache_dir ./data/xbrl/edgar_cache \
    --out_tag_usage ./data/xbrl/tag_usage.csv \
    --out_tag_labels ./data/xbrl/tag_labels.csv

  # Suggest mappings (especially for extensions)
  python scripts/xbrl_tag_factory.py suggest \
    --concepts_xlsx data/xbrl/fundamental.xlsx \
    --tag_usage ./data/xbrl/tag_usage.csv \
    --tag_labels ./data/xbrl/tag_labels.csv \
    --out_suggestions ./data/config/mapping_suggestions.yaml

  # (Optional) Maintain an approved mapping file (human edits once)
  # Then your downstream extractor uses that mapping deterministically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
import yaml
from rapidfuzz import fuzz
from lxml import etree # type: ignore


SEC_DATA = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives"
SEC_TICKERS_JSON = "https://www.sec.gov/files/company_tickers.json"

STANDARD_PREFIXES = {
    "us-gaap", "dei", "srt", "ifrs-full",
    "invest", "rr", "country", "currency", "exch", "stpr",
}

# --------- Utilities ----------

def cik10(cik: int | str) -> str:
    s = str(cik).lstrip("0")
    return s.zfill(10)

def cik_nolead(cik: int | str) -> str:
    return str(int(str(cik)))

def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_concepts_from_xlsx(path: str, only_without_formula: bool = False) -> List[str]:
    df = pd.read_excel(path, sheet_name="fields")
    
    # Filter concepts based on Formula column if requested
    if only_without_formula and "Formula" in df.columns:
        # Only include concepts where Formula is empty/null
        df = df[df["Formula"].isna() | (df["Formula"].astype(str).str.strip() == "")]
    
    concepts = [c for c in df["Concept"].dropna().astype(str).tolist()]
    # de-dup preserving order
    seen = set()
    out = []
    for c in concepts:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

# --------- SEC Client with politeness ----------

def tag_to_label(tag: str) -> str:
    """
    Convert XBRL tag to human-readable label.
    
    Examples:
        us-gaap:Assets → "Assets"
        us-gaap:NetIncomeLoss → "Net Income Loss"
        dei:EntityRegistrantName → "Entity Registrant Name"
        aapl:RevenueByProduct → "Revenue By Product"
    """
    # Split prefix and local name
    if ':' in tag:
        prefix, local = tag.split(':', 1)
    else:
        local = tag
    
    # Remove .domain, .typed suffixes
    local = re.sub(r'\.(domain|typed)$', '', local)
    
    # Insert spaces before capital letters (camelCase to words)
    # NetIncomeLoss → Net Income Loss
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', local)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)  # Handle acronyms
    
    # Clean up multiple spaces
    spaced = re.sub(r'\s+', ' ', spaced).strip()
    
    return spaced

class SecClient:
    """
    Minimal SEC client with:
    - required User-Agent
    - rate limiting
    - caching to disk for GETs
    """
    def __init__(self, user_agent: str, cache_dir: str, max_rps: float = 5.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.cache_dir = ensure_dir(cache_dir)
        self.max_rps = max_rps
        self._last = 0.0

    def _sleep_rate_limit(self):
        # simple spacing limiter
        min_interval = 1.0 / max(self.max_rps, 0.1)
        now = time.time()
        dt = now - self._last
        if dt < min_interval:
            time.sleep(min_interval - dt)
        self._last = time.time()

    def get_json(self, url: str, cache_key: str) -> dict:
        path = self.cache_dir / (cache_key + ".json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        self._sleep_rate_limit()
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data

    def get_bytes(self, url: str, cache_key: str) -> bytes:
        path = self.cache_dir / cache_key
        if path.exists():
            return path.read_bytes()

        # SEC archives is different host; set host header dynamically
        self._sleep_rate_limit()
        headers = dict(self.session.headers)
        headers["Host"] = "www.sec.gov" if "sec.gov/Archives" in url else "data.sec.gov"
        r = self.session.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        b = r.content
        path.write_bytes(b)
        return b

# --------- Sampling companies (industry + size proxy) ----------

@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    title: str

def fetch_all_tickers(sec: SecClient) -> List[Company]:
    # SEC publishes a ticker->CIK mapping JSON
    data = sec.get_json(SEC_TICKERS_JSON, "sec_company_tickers")
    out: List[Company] = []
    for _, row in data.items():
        try:
            out.append(Company(cik=int(row["cik_str"]), ticker=row["ticker"], title=row["title"]))
        except Exception:
            continue
    return out

def fetch_submissions(sec: SecClient, cik: int) -> dict:
    url = f"{SEC_DATA}/submissions/CIK{cik10(cik)}.json"
    return sec.get_json(url, f"submissions_{cik10(cik)}")

def get_sic_bucket(submissions: dict) -> str:
    sic = submissions.get("sic")
    if sic is None:
        return "unknown"
    # bucket as first 2 digits (broad industry groups)
    s = str(sic)
    return s[:2].ljust(2, "0")

def extract_assets_size_proxy(submissions: dict) -> Optional[float]:
    """
    Free size proxy without external market cap:
    - use 'entityType' is not helpful
    - so we approximate size by filing count / large filers,
      but better: (optional) use companyfacts Assets later.
    Here we return None; we will compute size quantiles after pulling companyfacts.
    """
    return None

def fetch_companyfacts(sec: SecClient, cik: int) -> dict:
    url = f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik10(cik)}.json"
    return sec.get_json(url, f"companyfacts_{cik10(cik)}")

def latest_numeric_fact(companyfacts: dict, tag: str) -> Optional[float]:
    """
    Pull the latest numeric value for a given GAAP tag (e.g., us-gaap:Assets)
    from companyfacts JSON (not per filing, but good for size proxy).
    """
    try:
        facts = companyfacts["facts"]
        prefix, local = tag.split(":")
        units = facts[prefix][local]["units"]
        # choose any unit (USD etc), take latest by 'end' then 'fy'
        best = None
        for unit, arr in units.items():
            for x in arr:
                v = x.get("val")
                end = x.get("end") or ""
                fy = x.get("fy") or 0
                if v is None:
                    continue
                key = (end, fy)
                if best is None or key > best[0]:
                    best = (key, float(v))
        return None if best is None else best[1]
    except Exception:
        return None

def stratified_sample_companies(
    sec: SecClient,
    companies: List[Company],
    n: int,
    seed: int = 7,
    max_scan: int = 2500,
) -> List[Company]:
    """
    Stratify by:
      - SIC bucket (2-digit)
      - size tier using latest us-gaap:Assets from companyfacts (small/mid/large by quantiles)
    This keeps it fully free and SEC-native.

    We scan up to max_scan random companies to find enough with usable data.
    """
    rng = random.Random(seed)
    pool = companies[:]
    rng.shuffle(pool)
    pool = pool[:max_scan]

    rows = []
    for c in pool:
        try:
            subs = fetch_submissions(sec, c.cik)
            sic_bucket = get_sic_bucket(subs)

            cf = fetch_companyfacts(sec, c.cik)
            assets = latest_numeric_fact(cf, "us-gaap:Assets")
            if assets is None or not math.isfinite(assets):
                continue

            rows.append((c, sic_bucket, assets))
        except Exception:
            continue

    if len(rows) < n:
        # fall back to whatever we have
        return [r[0] for r in rows[:n]]

    # compute tiers
    assets_vals = sorted([r[2] for r in rows])
    q1 = assets_vals[int(0.33 * (len(assets_vals) - 1))]
    q2 = assets_vals[int(0.66 * (len(assets_vals) - 1))]

    def tier(a: float) -> str:
        if a <= q1:
            return "small"
        if a <= q2:
            return "mid"
        return "large"

    buckets: Dict[Tuple[str, str], List[Company]] = defaultdict(list)
    for c, sic, a in rows:
        buckets[(sic, tier(a))].append(c)

    # round-robin draw across buckets
    keys = list(buckets.keys())
    rng.shuffle(keys)

    sample: List[Company] = []
    idx = 0
    while len(sample) < n and keys:
        k = keys[idx % len(keys)]
        if buckets[k]:
            sample.append(buckets[k].pop())
        else:
            keys.remove(k)
            continue
        idx += 1

    return sample[:n]

# --------- Filings selection + download ----------

@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession: str
    form: str
    filing_date: str

def iter_filings_for_company(submissions: dict, forms: Set[str], start_year: int, end_year: int) -> List[FilingRef]:
    """
    Use submissions.filings.recent which is easy + free.
    Filter to requested forms + year range.
    Only include filings that likely have XBRL files (we'll confirm by checking index.json).
    """
    recent = submissions.get("filings", {}).get("recent", {})
    accs = recent.get("accessionNumber", []) or []
    form_list = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []

    out: List[FilingRef] = []
    for acc, form, d in zip(accs, form_list, dates):
        if form not in forms:
            continue
        year = int(str(d)[:4])
        if year < start_year or year > end_year:
            continue
        out.append(FilingRef(
            cik=int(submissions["cik"]),
            accession=acc,
            form=form,
            filing_date=d,
        ))
    return out

def filing_index_json_url(cik: int, accession: str) -> str:
    return f"{SEC_ARCHIVES}/edgar/data/{cik_nolead(cik)}/{accession_nodash(accession)}/index.json"

def filing_file_url(cik: int, accession: str, filename: str) -> str:
    return f"{SEC_ARCHIVES}/edgar/data/{cik_nolead(cik)}/{accession_nodash(accession)}/{filename}"

def find_instance_candidates(idx: dict) -> list[str]:
    items = idx.get("directory", {}).get("item", []) or []
    names = []
    for it in items:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        low = name.lower()
        # include iXBRL primary docs too
        if low.endswith((".xml", ".htm", ".html")):
            names.append(name)

    # Optional: sort to try likely instance-like names first
    def score(n: str) -> tuple[int, int]:
        low = n.lower()
        # Prefer explicit instance-ish names, then HTML, then XML
        s = 0
        if "instance" in low: s -= 50
        if low.endswith((".htm", ".html")): s -= 20
        if low.endswith(".xml"): s -= 10
        # Penalize known non-instance XML types
        if any(x in low for x in ("_cal", "_def", "_lab", "_pre", ".xsd")): s += 100
        return (s, len(low))

    names.sort(key=score)
    return names


def is_xbrl_instance(b: bytes) -> bool:
    """
    Detect if file is an XBRL instance document.
    
    Supports:
    - Traditional XBRL (.xml files)
    - Inline XBRL (.htm/.html files) ← Critical for modern filings!
    """
    # Scan first 500KB (instances can have large headers)
    head = b[:500_000].lower()
    
    # Traditional XBRL (XML format)
    if b"<xbrli:xbrl" in head or (b"<xbrl" in head and b"xmlns" in head):
        return True
    
    # Inline XBRL (HTML format) - most common since 2019
    if any(pattern in head for pattern in [
        b"xmlns:ix=",
        b"<ix:header",
        b"ix:nonfraction",
        b"ix:nonnumeric",
        b"http://www.xbrl.org/2013/inlinexbrl",
        b"http://www.xbrl.org/2008/inlinexbrl",
    ]):
        return True
    
    # Lenient check: HTML file with XBRL context references
    if b"us-gaap" in head and b"context" in head and b"<html" in head:
        return True
    
    return False


def _list_dir_names(index_json: dict) -> list[str]:
    items = index_json.get("directory", {}).get("item", []) or []
    names = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("name"), str):
            names.append(it["name"])
    return names

def _find_instance_via_filing_summary(
    sec,
    filing,  # FilingRef object
    index_json: dict,
) -> Optional[str]:
    """
    Extract instance filename from FilingSummary.xml.
    
    Handles both formats:
    - Old (pre-2019): <Instance>filename.xml</Instance>
    - New (2019+): First .htm file in <InputFiles> without _cal/_def/_lab/_pre suffix
    """
    # Check if FilingSummary.xml exists in the filing
    names = _list_dir_names(index_json)
    if "FilingSummary.xml" not in names:
        return None

    # Download FilingSummary.xml
    cache_key = f"filingsummary_{cik10(filing.cik)}_{accession_nodash(filing.accession)}.xml"
    url = filing_file_url(filing.cik, filing.accession, "FilingSummary.xml")
    
    try:
        b = sec.get_bytes(url, cache_key=cache_key)
    except Exception:
        return None

    txt = b.decode("utf-8", errors="ignore")

    # ========================================================================
    # METHOD 1: Old format - <Instance>something.xml</Instance>
    # ========================================================================
    m = re.search(r"<Instance>\s*([^<\s]+)\s*</Instance>", txt, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ========================================================================
    # METHOD 2: New format (2019+) - Parse <InputFiles> section
    # ========================================================================
    # The instance file is typically the first .htm file listed that doesn't
    # have a taxonomy suffix (_cal, _def, _lab, _pre, _schema)
    
    # Extract the <InputFiles> section
    input_files_match = re.search(
        r'<InputFiles>(.*?)</InputFiles>', 
        txt, 
        flags=re.IGNORECASE | re.DOTALL
    )
    
    if input_files_match:
        input_section = input_files_match.group(1)
        
        # Find all <File> tags
        file_tags = re.findall(r'<File>([^<]+)</File>', input_section, flags=re.IGNORECASE)
        
        # Filter for .htm files without taxonomy suffixes
        for filename in file_tags:
            filename = filename.strip()
            lower = filename.lower()
            
            # Must be .htm or .html
            if not (lower.endswith('.htm') or lower.endswith('.html')):
                continue
            
            # Must NOT be a taxonomy file
            if any(suffix in lower for suffix in ['_cal.', '_def.', '_lab.', '_pre.', '_schema.']):
                continue
            
            # Must NOT be an index or report reference file
            if any(word in lower for word in ['index', 'report', 'summary']):
                continue
            
            # This is likely the instance file!
            # Typically matches pattern: ticker-YYYYMMDD.htm
            return filename
    
    # ========================================================================
    # METHOD 3: Parse <MyReports> section as fallback
    # ========================================================================
    # Sometimes the instance is referenced in the first Report entry
    reports_match = re.search(
        r'<MyReports>(.*?)</MyReports>',
        txt,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    if reports_match:
        reports_section = reports_match.group(1)
        
        # Get the first <HtmlFileName>
        html_match = re.search(
            r'<HtmlFileName>([^<]+)</HtmlFileName>',
            reports_section,
            flags=re.IGNORECASE
        )
        
        if html_match:
            filename = html_match.group(1).strip()
            # Check if it looks like an instance (ticker-date pattern)
            if re.match(r'[a-z]{2,6}-\d{8}\.html?', filename, re.IGNORECASE):
                return filename

    # ========================================================================
    # METHOD 4: Scan entire document for ticker-date.htm pattern
    # ========================================================================
    # Last resort: find any file matching the ticker-date pattern
    # This is reliable because instance files follow a strict naming convention
    ticker_date_matches = re.findall(
        r'([a-z]{2,6}-\d{8}\.html?)',
        txt,
        flags=re.IGNORECASE
    )
    
    if ticker_date_matches:
        # Return the first match (usually the instance)
        return ticker_date_matches[0]

    # ========================================================================
    # METHOD 5: Very old format - search for INSTANCE keyword
    # ========================================================================
    m2 = re.search(
        r"(?:INSTANCE|Instance)[^<]{0,200}<Href>\s*([^<\s]+\.(?:xml|htm|html))\s*</Href>",
        txt,
        flags=re.IGNORECASE,
    )
    if m2:
        return m2.group(1).strip()

    return None

def download_best_instance(sec: SecClient, filing: FilingRef, verbose: bool = False) -> Optional[Path]:
    idx_cache_key = f"index_{cik10(filing.cik)}_{accession_nodash(filing.accession)}"
    
    try:
        idx = sec.get_json(
            filing_index_json_url(filing.cik, filing.accession),
            cache_key=idx_cache_key,
        )
    except Exception as e:
        if verbose:
            print(f"[warn] Failed to get index for {filing.cik}/{filing.accession}: {e}")
        return None

    base_key = f"instance_{cik10(filing.cik)}_{accession_nodash(filing.accession)}_"

    # 1) Try the reliable method first: FilingSummary.xml -> instance filename
    inst_name = _find_instance_via_filing_summary(sec, filing, idx)
    if inst_name:
        try:
            url = filing_file_url(filing.cik, filing.accession, inst_name)
            b = sec.get_bytes(url, cache_key=base_key + inst_name)
            if is_xbrl_instance(b):
                return sec.cache_dir / (base_key + inst_name)
        except Exception:
            pass  # fall through to heuristics

    # 2) Fallback to heuristics over directory listing
    candidates = find_instance_candidates(idx)
    if not candidates:
        return None

    for name in candidates[:50]:  # widen search a bit
        try:
            url = filing_file_url(filing.cik, filing.accession, name)
            b = sec.get_bytes(url, cache_key=base_key + name)
            if is_xbrl_instance(b):
                return sec.cache_dir / (base_key + name)
        except Exception:
            continue

    return None


# --------- XBRL parsing + tag extraction ----------

@dataclass
class TagInfo:
    tag: str               # e.g. us-gaap:Revenues or aapl:NetSales
    prefix: str            # us-gaap / aapl etc
    local: str             # Revenues / NetSales etc
    is_extension: bool
    label: Optional[str]   # best human label if available


def parse_instance_tags_xml(instance_path: Path) -> Tuple[Set[str], Dict[str, TagInfo]]:
    """
    CORRECTED: Extract tags from both traditional XBRL and inline XBRL.
    
    For inline XBRL (.htm/.html):
      Facts are in 'name' attributes: <ix:nonFraction name="us-gaap:Assets" ...>
      
    For traditional XBRL (.xml):
      Facts are element tags: <us-gaap:Assets ...>
    """
    tree = etree.parse(str(instance_path))
    root = tree.getroot()

    tags_in_filing: Set[str] = set()
    taginfo: Dict[str, TagInfo] = {}

    # Build namespace-uri -> prefix mapping
    uri_to_prefix = {}
    for pfx, uri in (root.nsmap or {}).items():
        if pfx and uri:
            uri_to_prefix[uri] = pfx

    # Infrastructure to ignore (for traditional XBRL element tags)
    IGNORE_LOCAL = {
        "context", "unit", "schemaRef", "linkbaseRef", "footnoteLink",
        "header", "hidden", "references", "resources", "tuple",
    }
    
    IGNORE_NS_URIS = {
        "http://www.xbrl.org/2003/instance",
        "http://www.xbrl.org/2003/linkbase",
        "http://www.w3.org/1999/xlink",
        "http://www.w3.org/XML/1998/namespace",
        "http://www.w3.org/1999/xhtml",
    }
    
    # Prefixes that indicate infrastructure (not facts)
    IGNORE_PREFIXES = {
        "xbrldi", "xbrli", "link", "xlink", "html", "body", "head"
    }

    # ========================================================================
    # STEP 1: Extract facts from 'name' attributes (inline XBRL)
    # ========================================================================
    for el in root.iter():
        name_attr = el.get('name')
        
        if name_attr and ':' in name_attr:
            # This is an inline XBRL fact!
            # Format: "prefix:localName" e.g., "us-gaap:Assets"
            try:
                prefix, local = name_attr.split(':', 1)
                
                # Skip dimensional metadata
                if prefix.lower() in IGNORE_PREFIXES:
                    continue
                
                # Skip .domain and .typed suffixes (dimensional)
                if local.endswith('.domain') or local.endswith('.typed'):
                    continue
                
                tag = name_attr  # Already in "prefix:local" format
                tags_in_filing.add(tag)
                
                if tag not in taginfo:
                    taginfo[tag] = TagInfo(
                        tag=tag,
                        prefix=prefix,
                        local=local,
                        is_extension=(prefix not in STANDARD_PREFIXES),
                        label=None,
                    )
            except ValueError:
                # Malformed name attribute, skip
                continue

    # ========================================================================
    # STEP 2: Extract facts from element tags (traditional XBRL)
    # ========================================================================
    # This handles .xml files where facts are actual element tags
    for el in root.iter():
        qn = etree.QName(el.tag)
        local = qn.localname
        uri = qn.namespace

        # Skip root
        if el is root:
            continue

        # Skip infrastructure elements
        if local in IGNORE_LOCAL:
            continue
            
        if uri in IGNORE_NS_URIS:
            continue

        # Get prefix
        prefix = el.prefix or uri_to_prefix.get(uri, "")
        
        # Skip infrastructure prefixes
        if prefix.lower() in IGNORE_PREFIXES:
            continue
        
        # Skip if no prefix
        if not prefix:
            continue

        # Skip dimensional suffixes
        if local.endswith('.domain') or local.endswith('.typed'):
            continue
        
        # Skip inline XBRL wrapper elements (these were already processed via 'name' attr)
        if prefix == 'ix':
            continue

        # This is a traditional XBRL fact element
        tag = f"{prefix}:{local}"
        tags_in_filing.add(tag)

        if tag not in taginfo:
            taginfo[tag] = TagInfo(
                tag=tag,
                prefix=prefix,
                local=local,
                is_extension=(prefix not in STANDARD_PREFIXES),
                label=None,
            )

    return tags_in_filing, taginfo


# --------- Filtering: what NOT to map ----------

def should_ignore_for_mapping(tag: str, info: Optional[TagInfo]) -> bool:
    """
    Skip things not to map:
    - segment/disaggregation axes/members, dimensional stuff
    - footnote-y / breakdown disclosures
    Heuristics (not perfect, but good enough to cut noise):
    """
    t = tag.lower()

    # dimensions & members often contain Axis/Member/Domain/Table/LineItems
    if any(x in t for x in ["axis", "member", "domain", "table", "lineitems", "lineitems"]):
        return True

    # statement presentation / taxonomy structural items
    if info and info.prefix in {"srt"}:
        return True

    # DEI is mostly identifiers; you typically don't map into fundamentals (except maybe shares)
    if info and info.prefix == "dei":
        return True

    # explicit “segment” / “geograph” / “byproduct” etc
    if any(x in t for x in ["segment", "geograph", "disaggregation", "breakdown", "concentration"]):
        return True

    return False

# --------- Suggest mappings (GAAP-first; label similarity for extensions) ----------

def concept_display_name(concept: str) -> str:
    # revenue -> "Revenue", net_income -> "Net income"
    return concept.replace("_", " ").strip()

def keyword_rules_for_concept(concept: str) -> List[List[str]]:
    """
    Very lightweight rules that give high-precision suggestions.
    These are NOT exhaustive; the idea is to reduce human time.
    """
    rules = {
        "revenue": [["revenue"], ["sales"]],
        "net_income": [["net", "income"], ["profit", "loss"]],
        "operating_income": [["operating", "income"]],
        "gross_profit": [["gross", "profit"]],
        "cost_of_revenue": [["cost", "revenue"], ["cost", "goods"]],
        "assets": [["assets"]],
        "liabilities": [["liabilities"]],
        "cash_and_cash_equivalents": [["cash", "equivalents"]],
        "operating_cash_flow": [["net", "cash", "operating"]],
        "capital_expenditures": [["capital", "expenditures"], ["purchase", "property", "equipment"]],
        # extend from your fundamental.xlsx concepts over time
    }
    return rules.get(concept, [])

def score_tag_for_concept(concept: str, tag: str, label: str | None) -> int:
    """
    Score with:
      - label similarity (primary)
      - tag text similarity (secondary)
      - keyword boosts (high precision)
    """
    c_name = concept_display_name(concept).lower()
    tag_l = tag.lower()
    lbl_l = (label or "").lower()

    sim_lbl = fuzz.token_set_ratio(c_name, lbl_l) if lbl_l else 0
    sim_tag = fuzz.token_set_ratio(c_name, tag_l)

    score = int(0.75 * sim_lbl + 0.25 * sim_tag)

    # keyword boosts
    for must_words in keyword_rules_for_concept(concept):
        if all(w in lbl_l for w in must_words) or all(w in tag_l for w in must_words):
            score += 10

    # penalty for “axis/member/table…” noise
    if any(x in tag_l for x in ["axis", "member", "domain", "table", "lineitems"]):
        score -= 30

    return score

def suggest_mappings(
    concepts: List[str],
    tag_usage: pd.DataFrame,     # columns: tag, usage_count, is_extension
    tag_labels: pd.DataFrame,    # columns: tag, label
    top_k_tags: int = 400,
    per_concept: int = 8,
    min_score: int = 65,
) -> dict:
    """
    Produce YAML suggestions:
      concept:
        gaap_candidates: [...]
        extension_candidates: [...]
    """
    labels = dict(zip(tag_labels["tag"].astype(str), tag_labels["label"].astype(str)))

    tag_usage = tag_usage.sort_values("usage_count", ascending=False).head(top_k_tags).copy()

    suggestions = {}
    for concept in concepts:
        scored = []
        for _, row in tag_usage.iterrows():
            tag = str(row["tag"])
            if should_ignore_for_mapping(tag, None):
                continue
            label = labels.get(tag)
            score = score_tag_for_concept(concept, tag, label)
            scored.append((score, tag, bool(row.get("is_extension", False)), label))

        scored.sort(reverse=True, key=lambda x: x[0])
        gaap = [t for s, t, ext, _ in scored if (not ext and s >= min_score)][:per_concept]
        extc = [t for s, t, ext, _ in scored if (ext and s >= min_score)][:per_concept]

        suggestions[concept] = {
            "gaap_candidates": gaap,
            "extension_candidates": extc,
        }
    return suggestions

# --------- Harvest pipeline ----------

def harvest(
    user_agent: str,
    concepts_xlsx: str,
    n_companies: int,
    start_year: int,
    end_year: int,
    forms: List[str],
    cache_dir: str,
    out_tag_usage: str,
    out_tag_labels: str,
    max_filings_total: int = 600,
    seed: int = 7,
):
    sec = SecClient(user_agent=user_agent, cache_dir=cache_dir, max_rps=5.0)

    concepts = load_concepts_from_xlsx(concepts_xlsx)
    print(f"[info] Loaded {len(concepts)} canonical concepts from {concepts_xlsx}")

    all_companies = fetch_all_tickers(sec)
    print(f"[info] Loaded {len(all_companies)} tickers from SEC list")

    sample = stratified_sample_companies(sec, all_companies, n=n_companies, seed=seed)
    print(f"[info] Sampled {len(sample)} companies (stratified by SIC bucket + Assets proxy)")

    forms_set = set(forms)

    # Build candidate filings list (cap total for speed)
    filings: List[FilingRef] = []
    for c in sample:
        try:
            subs = fetch_submissions(sec, c.cik)
            fs = iter_filings_for_company(subs, forms_set, start_year, end_year)
            # keep newest first
            fs = sorted(fs, key=lambda x: x.filing_date, reverse=True)
            filings.extend(fs[: max(1, (end_year - start_year + 1))])  # rough per-company cap
        except Exception:
            continue

    filings = sorted(filings, key=lambda x: x.filing_date, reverse=True)[:max_filings_total]
    print(f"[info] Will process up to {len(filings)} filings total")

    tag_usage = Counter()
    tag_labels: Dict[str, str] = {}
    tag_is_extension: Dict[str, bool] = {}

    processed = 0
    n_no_instance = 0
    n_zero_tags = 0
    n_parse_fail = 0

    for f in filings:
        try:
            inst_path = download_best_instance(sec, f, verbose=(processed < 5))
            if inst_path is None:
                n_no_instance += 1
                continue

            try:
                tags_in_filing, infos = parse_instance_tags_xml(inst_path)
            except Exception:
                n_parse_fail += 1
                continue

            processed += 1

            if not tags_in_filing:
                n_zero_tags += 1
                continue

            # Count presence per filing (NOT per-fact)
            for tag in tags_in_filing:
                tag_usage[tag] += 1

            # Keep best label we saw
            for tag, info in infos.items():
                if info.label and (tag not in tag_labels):
                    tag_labels[tag] = info.label
                tag_is_extension[tag] = info.is_extension

            if processed % 25 == 0:
                print(f"[debug] no_instance={n_no_instance} parse_fail={n_parse_fail} zero_tags={n_zero_tags} processed={processed}")
        except Exception as e:
            if processed < 3:
                print(
                    f"[warn] filing failed "
                    f"cik={f.cik} acc={f.accession} "
                    f"{type(e).__name__}: {e}"
                )
            continue

    # Final summary
    print(f"[info] Processing complete: no_instance={n_no_instance} parse_fail={n_parse_fail} zero_tags={n_zero_tags} processed={processed}")

    # Write tag_usage.csv
    usage_rows = []
    for tag, cnt in tag_usage.most_common():
        usage_rows.append({
            "tag": tag,
            "usage_count": cnt,
            "is_extension": bool(tag_is_extension.get(tag, False)),
        })
    pd.DataFrame(usage_rows).to_csv(out_tag_usage, index=False)
    print(f"[done] wrote {out_tag_usage} ({len(usage_rows)} tags)")

    # Write tag_labels.csv
    label_rows = []
    for tag, _ in tag_usage.most_common():
        # Use extracted label if available, otherwise generate from tag name
        label = tag_labels.get(tag, "") or tag_to_label(tag)
        label_rows.append({"tag": tag, "label": label})
    pd.DataFrame(label_rows).to_csv(out_tag_labels, index=False)
    print(f"[done] wrote {out_tag_labels}")

def run_suggest(concepts_xlsx: str, tag_usage_path: str, tag_labels_path: str, out_yaml: str):
    # Only load concepts where Formula column is empty
    concepts = load_concepts_from_xlsx(concepts_xlsx, only_without_formula=True)
    tag_usage = pd.read_csv(tag_usage_path)
    tag_labels = pd.read_csv(tag_labels_path)

    sug = suggest_mappings(concepts, tag_usage, tag_labels)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(sug, f, sort_keys=False, allow_unicode=True)
    print(f"[done] wrote {out_yaml}")

# --------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_h = sub.add_parser("harvest")
    ap_h.add_argument("--user_agent", required=True,
                      help="REQUIRED by SEC. Example: 'YourName your@email.com'")
    ap_h.add_argument("--concepts_xlsx", required=True)
    ap_h.add_argument("--n_companies", type=int, default=300)
    ap_h.add_argument("--start_year", type=int, default=2019)
    ap_h.add_argument("--end_year", type=int, default=2025)
    ap_h.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    ap_h.add_argument("--cache_dir", default="./data/xbrl/edgar_cache")
    ap_h.add_argument("--out_tag_usage", default="./data/xbrl/tag_usage.csv")
    ap_h.add_argument("--out_tag_labels", default="./data/xbrl/tag_labels.csv")
    ap_h.add_argument("--max_filings_total", type=int, default=600)
    ap_h.add_argument("--seed", type=int, default=7)

    ap_s = sub.add_parser("suggest")
    ap_s.add_argument("--concepts_xlsx", required=True)
    ap_s.add_argument("--tag_usage", required=True)
    ap_s.add_argument("--tag_labels", required=True)
    ap_s.add_argument("--out_suggestions", default="./data/config/mapping_suggestions.yaml")

    args = ap.parse_args()

    if args.cmd == "harvest":
        harvest(
            user_agent=args.user_agent,
            concepts_xlsx=args.concepts_xlsx,
            n_companies=args.n_companies,
            start_year=args.start_year,
            end_year=args.end_year,
            forms=args.forms,
            cache_dir=args.cache_dir,
            out_tag_usage=args.out_tag_usage,
            out_tag_labels=args.out_tag_labels,
            max_filings_total=args.max_filings_total,
            seed=args.seed,
        )
    elif args.cmd == "suggest":
        run_suggest(
            concepts_xlsx=args.concepts_xlsx,
            tag_usage_path=args.tag_usage,
            tag_labels_path=args.tag_labels,
            out_yaml=args.out_suggestions,
        )

if __name__ == "__main__":
    main()