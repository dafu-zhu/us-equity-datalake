# Sentiment Analysis Models

This module provides NLP-based sentiment analysis for SEC EDGAR filings using FinBERT and Loughran-McDonald financial dictionaries.

## Overview

The sentiment pipeline extracts Management's Discussion and Analysis (MD&A) sections from 10-K and 10-Q filings, then computes sentiment metrics using both deep learning (FinBERT) and dictionary-based (Loughran-McDonald) approaches.

## Data Collection Methodology

### Source
- **SEC EDGAR**: Full-submission text files from 10-K (annual) and 10-Q (quarterly) filings
- **Section**: Item 7 (10-K) / Item 2 (10-Q) - Management's Discussion and Analysis

### Extraction Process
1. Fetch `full-submission.txt` from SEC EDGAR Archives
2. Decode HTML entities (&#8217; → ', &#160; → space, etc.)
3. Extract MD&A section using regex pattern matching on Item headers
4. Clean HTML tags and normalize whitespace
5. Validate minimum length (500+ characters)

### Text Chunking
- FinBERT has 512 token limit (~1500 characters)
- Text split into overlapping chunks (1500 chars, 200 char overlap)
- Chunk boundaries prefer sentence endings for context continuity

## Sentiment Metrics

### 1. FinBERT Deep Learning Metrics

**Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

FinBERT is a BERT model fine-tuned on financial communication text for sentiment classification.

| Metric | Description | Range |
|--------|-------------|-------|
| `sentiment_score` | Weighted aggregate sentiment (positive contributes +score, negative -score, normalized by chunk count) | -1 to +1 |
| `positive_ratio` | Fraction of chunks classified as positive | 0 to 1 |
| `negative_ratio` | Fraction of chunks classified as negative | 0 to 1 |
| `neutral_ratio` | Fraction of chunks classified as neutral | 0 to 1 |
| `avg_positive_confidence` | Mean confidence score for positive classifications | 0 to 1 |
| `avg_negative_confidence` | Mean confidence score for negative classifications | 0 to 1 |
| `chunk_count` | Number of text chunks processed | Integer |
| `text_length` | Total characters in MD&A section | Integer |

**Reference**:
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)

### 2. Sentiment Distribution Metrics

These metrics capture the volatility and distribution of sentiment across the document, which research shows predicts future stock volatility.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `sentiment_std` | Standard deviation of chunk sentiments | Higher = more mixed/volatile tone |
| `sentiment_skew` | Skewness of sentiment distribution (Pearson's) | Negative = more negative outliers |
| `sentiment_range` | Max - min sentiment across chunks | Higher = wider sentiment variation |
| `extreme_negative_ratio` | Fraction of chunks with sentiment < -0.5 | Higher = presence of strongly negative sections |
| `confidence_std` | Standard deviation of confidence scores | Higher = more uncertainty in classifications |

**Reference**:
- Garcia, D. (2013). Sentiment during Recessions. *Journal of Finance*, 68(3), 1267-1300. [DOI](https://doi.org/10.1111/jofi.12027)

### 3. Loughran-McDonald Dictionary Metrics

Dictionary-based word counts using the Loughran-McDonald Master Dictionary, specifically designed for financial text (unlike general sentiment dictionaries which misclassify financial terms).

| Metric | Description | Research Finding |
|--------|-------------|------------------|
| `word_count` | Total words in MD&A | Document length baseline |
| `uncertainty_ratio` | Fraction of uncertainty words (e.g., "may", "could", "risk", "volatile") | Associated with higher stock return volatility |
| `litigious_ratio` | Fraction of litigious words (e.g., "lawsuit", "plaintiff", "court") | Associated with higher litigation risk |
| `constraining_ratio` | Fraction of constraining words (e.g., "required", "obligated", "must") | Indicates operational/regulatory constraints |
| `weak_modal_ratio` | Fraction of weak modal words (e.g., "may", "might", "could") | Higher usage associated with earnings restatements |
| `strong_modal_ratio` | Fraction of strong modal words (e.g., "will", "must", "always") | Indicates management certainty |

**Reference**:
- Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65. [DOI](https://doi.org/10.1111/j.1540-6261.2010.01625.x)
- Master Dictionary: [https://sraf.nd.edu/loughranmcdonald-master-dictionary/](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

### 4. Readability Metrics

Text complexity metrics that research shows predict earnings and stock returns.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `avg_sentence_length` | Average words per sentence | Higher = more complex sentences |
| `fog_index` | Gunning Fog Index: 0.4 × (avg_words_per_sentence + percent_complex_words) | Higher = harder to read (12+ = college level) |

**Reference**:
- Li, F. (2008). Annual Report Readability, Current Earnings, and Earnings Persistence. *Journal of Accounting and Economics*, 45(2-3), 221-247. [DOI](https://doi.org/10.1016/j.jacceco.2008.02.003)
- Loughran, T., & McDonald, B. (2014). Measuring Readability in Financial Disclosures. *Journal of Finance*, 69(4), 1643-1671. [DOI](https://doi.org/10.1111/jofi.12162)

## Storage Schema

Output stored as long-format Parquet at `data/derived/features/sentiment/{cik}/sentiment.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| cik | str | SEC CIK (10-digit zero-padded) |
| as_of_date | str | Filing date (YYYY-MM-DD) |
| filing_type | str | "10-K" or "10-Q" |
| fiscal_year | int | Fiscal year (nullable) |
| fiscal_quarter | int | Quarter 1-4, null for 10-K |
| metric | str | Metric name from above |
| value | float | Metric value |
| model_name | str | "finbert" |
| model_version | str | Version for reproducibility |

## Usage

```python
from quantdl.models.finbert import FinBERTModel
from quantdl.collection.sentiment import SentimentCollector
from quantdl.derived.sentiment import compute_sentiment_for_cik

# Load model (lazy loading, auto-detects CUDA)
model = FinBERTModel()

# Collect filing texts
collector = SentimentCollector()
filings = collector.get_filings_metadata(cik="0000320193", start_date="2020-01-01")
filing_texts = collector.collect_filing_texts(cik="0000320193", filings=filings)

# Compute sentiment
df = compute_sentiment_for_cik(
    cik="0000320193",
    filing_texts=filing_texts,
    model=model,
    symbol="AAPL"
)
```

## CLI Usage

```bash
# Run sentiment analysis
uv run quantdl-storage --run-sentiment --start-year 2020

# Full backfill
uv run quantdl-storage --run-sentiment --start-year 2009 --end-year 2025
```

## Hardware Requirements

- **GPU (recommended)**: CUDA-compatible GPU with 1+ GB VRAM
- **CPU**: Supported but significantly slower (~10x)
- **Memory**: ~500 MB for model weights

## Rate Limiting

- SEC EDGAR: 10 requests/second (configured at 9.5 req/sec)
- Processing: Sequential per-symbol (GPU inference is bottleneck)

## References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)

2. Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.

3. Li, F. (2008). Annual Report Readability, Current Earnings, and Earnings Persistence. *Journal of Accounting and Economics*, 45(2-3), 221-247.

4. Garcia, D. (2013). Sentiment during Recessions. *Journal of Finance*, 68(3), 1267-1300.

5. Loughran, T., & McDonald, B. (2014). Measuring Readability in Financial Disclosures. *Journal of Finance*, 69(4), 1643-1671.

6. Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy Prices. *Journal of Finance*, 75(3), 1371-1415. (Methodology for textual change analysis)
