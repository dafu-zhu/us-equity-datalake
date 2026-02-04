"""
Sentiment computation from SEC filing texts.

Computes sentiment metrics from MD&A sections using FinBERT.
Includes Loughran-McDonald word ratios and readability metrics.
Outputs long-format DataFrame compatible with storage pipeline.

References:
- Loughran & McDonald (2011): Financial sentiment dictionaries
- Li (2008): Annual report readability and earnings
"""

import logging
import re
import statistics
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import polars as pl

from quantdl.models.base import SentimentModel, SentimentResult
from quantdl.collection.sentiment import FilingText
from quantdl.derived.word_lists import compute_word_ratios


@dataclass
class FilingSentiment:
    """Aggregated sentiment for a single filing."""

    cik: str
    filing_date: str
    filing_type: str
    fiscal_year: Optional[int]
    fiscal_quarter: Optional[int]

    # FinBERT sentiment metrics
    sentiment_score: float  # Aggregated -1 to +1
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    avg_positive_confidence: float
    avg_negative_confidence: float
    chunk_count: int
    text_length: int

    # Distribution metrics (sentiment volatility)
    sentiment_std: float  # Std dev of chunk sentiments
    sentiment_skew: float  # Skewness of sentiment distribution
    sentiment_range: float  # Max - min sentiment
    extreme_negative_ratio: float  # Chunks with sentiment < -0.5
    confidence_std: float  # Std dev of confidence scores

    # Loughran-McDonald word ratios
    word_count: int
    uncertainty_ratio: float
    litigious_ratio: float
    constraining_ratio: float
    weak_modal_ratio: float
    strong_modal_ratio: float

    # Readability metrics
    avg_sentence_length: float
    fog_index: float

    # Model info
    model_name: str
    model_version: str


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks for model inference.

    FinBERT has 512 token limit. Using ~1500 chars (~375 tokens) per chunk
    with overlap ensures context continuity.

    :param text: Full text to chunk
    :param chunk_size: Target characters per chunk
    :param overlap: Overlap between chunks
    :return: List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in last 20% of chunk
            search_start = int(end - chunk_size * 0.2)
            search_text = text[search_start:end]

            # Find last sentence boundary
            for sep in ['. ', '.\n', '! ', '? ']:
                last_sep = search_text.rfind(sep)
                if last_sep != -1:
                    end = search_start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start with overlap
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def count_sentences(text: str) -> int:
    """Count sentences in text using simple heuristics."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    return len([s for s in sentences if s.strip()])


def count_complex_words(text: str) -> int:
    """
    Count complex words (3+ syllables) for Fog Index.

    Uses simple syllable counting heuristic.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    complex_count = 0

    for word in words:
        # Simple syllable count: count vowel groups
        syllables = len(re.findall(r'[aeiouy]+', word))
        # Adjust for silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        # Adjust for -ed, -es endings
        if word.endswith(('ed', 'es')) and syllables > 1:
            syllables -= 1
        if syllables >= 3:
            complex_count += 1

    return complex_count


def compute_fog_index(text: str) -> float:
    """
    Compute Gunning Fog Index for readability.

    Fog Index = 0.4 * (avg_words_per_sentence + percent_complex_words)
    Higher values indicate more difficult text.

    Reference: Li (2008) Annual report readability and earnings.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    word_count = len(words)
    sentence_count = count_sentences(text)

    if word_count == 0 or sentence_count == 0:
        return 0.0

    avg_words_per_sentence = word_count / sentence_count
    complex_word_count = count_complex_words(text)
    percent_complex = (complex_word_count / word_count) * 100

    fog = 0.4 * (avg_words_per_sentence + percent_complex)
    return round(fog, 2)


def _aggregate_sentiment_results(
    filing_text: FilingText,
    results: List[SentimentResult],
    model_name: str,
    model_version: str
) -> Optional[FilingSentiment]:
    """
    Aggregate sentiment results for a single filing.

    Internal helper that computes metrics from pre-computed model results.

    :param filing_text: FilingText with extracted MD&A
    :param results: Pre-computed SentimentResult list for this filing's chunks
    :param model_name: Name of the model used
    :param model_version: Version of the model used
    :return: FilingSentiment with aggregated metrics or None if no results
    """
    if not results:
        return None

    text = filing_text.text

    # Aggregate results
    positive_scores = []
    negative_scores = []
    all_confidences = []
    chunk_sentiments = []  # For distribution metrics
    neutral_count = 0
    positive_count = 0
    negative_count = 0

    for result in results:
        all_confidences.append(result.score)
        if result.label == "positive":
            positive_count += 1
            positive_scores.append(result.score)
            chunk_sentiments.append(result.score)  # Positive = +score
        elif result.label == "negative":
            negative_count += 1
            negative_scores.append(result.score)
            chunk_sentiments.append(-result.score)  # Negative = -score
        else:
            neutral_count += 1
            chunk_sentiments.append(0.0)  # Neutral = 0

    total = len(results)

    # Compute sentiment score: weighted by confidence
    sentiment_score = sum(chunk_sentiments) / total if total > 0 else 0.0

    # Compute distribution metrics
    if len(chunk_sentiments) > 1:
        sentiment_std = round(statistics.stdev(chunk_sentiments), 4)
        sentiment_range = round(max(chunk_sentiments) - min(chunk_sentiments), 4)
        # Skewness: (mean - median) / std (simplified Pearson's)
        mean_sent = statistics.mean(chunk_sentiments)
        median_sent = statistics.median(chunk_sentiments)
        sentiment_skew = round(
            (mean_sent - median_sent) / sentiment_std, 4
        ) if sentiment_std > 0 else 0.0
    else:
        sentiment_std = 0.0
        sentiment_range = 0.0
        sentiment_skew = 0.0

    # Extreme negative ratio: chunks with sentiment < -0.5
    extreme_neg_count = sum(1 for s in chunk_sentiments if s < -0.5)
    extreme_negative_ratio = round(extreme_neg_count / total, 4) if total > 0 else 0.0

    # Confidence distribution
    confidence_std = round(
        statistics.stdev(all_confidences), 4
    ) if len(all_confidences) > 1 else 0.0

    # Compute Loughran-McDonald word ratios
    word_ratios = compute_word_ratios(text)

    # Compute readability metrics
    sentence_count = count_sentences(text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    word_count = len(words)
    avg_sentence_length = round(
        word_count / sentence_count, 2
    ) if sentence_count > 0 else 0.0
    fog_index = compute_fog_index(text)

    return FilingSentiment(
        cik=filing_text.cik,
        filing_date=filing_text.filing_date,
        filing_type=filing_text.filing_type,
        fiscal_year=filing_text.fiscal_year,
        fiscal_quarter=filing_text.fiscal_quarter,
        # FinBERT metrics
        sentiment_score=round(sentiment_score, 4),
        positive_ratio=round(positive_count / total, 4) if total > 0 else 0.0,
        negative_ratio=round(negative_count / total, 4) if total > 0 else 0.0,
        neutral_ratio=round(neutral_count / total, 4) if total > 0 else 0.0,
        avg_positive_confidence=round(
            sum(positive_scores) / len(positive_scores), 4
        ) if positive_scores else 0.0,
        avg_negative_confidence=round(
            sum(negative_scores) / len(negative_scores), 4
        ) if negative_scores else 0.0,
        chunk_count=total,
        text_length=len(text),
        # Distribution metrics
        sentiment_std=sentiment_std,
        sentiment_skew=sentiment_skew,
        sentiment_range=sentiment_range,
        extreme_negative_ratio=extreme_negative_ratio,
        confidence_std=confidence_std,
        # Loughran-McDonald ratios
        word_count=word_ratios['word_count'],
        uncertainty_ratio=round(word_ratios['uncertainty_ratio'], 4),
        litigious_ratio=round(word_ratios['litigious_ratio'], 4),
        constraining_ratio=round(word_ratios['constraining_ratio'], 4),
        weak_modal_ratio=round(word_ratios['weak_modal_ratio'], 4),
        strong_modal_ratio=round(word_ratios['strong_modal_ratio'], 4),
        # Readability metrics
        avg_sentence_length=avg_sentence_length,
        fog_index=fog_index,
        # Model info
        model_name=model_name,
        model_version=model_version
    )


def compute_filing_sentiment(
    filing_text: FilingText,
    model: SentimentModel,
    chunk_size: int = 1500,
    logger: Optional[logging.Logger] = None
) -> Optional[FilingSentiment]:
    """
    Compute sentiment metrics for a single filing.

    :param filing_text: FilingText with extracted MD&A
    :param model: Loaded sentiment model
    :param chunk_size: Characters per chunk for inference
    :param logger: Optional logger
    :return: FilingSentiment with aggregated metrics or None on error
    """
    if not filing_text.text:
        return None

    # Chunk the text
    chunks = chunk_text(filing_text.text, chunk_size=chunk_size)
    if not chunks:
        return None

    # Run inference
    try:
        results = model.predict(chunks)
    except Exception as e:
        if logger:
            logger.warning(
                f"Inference failed for {filing_text.cik} "
                f"{filing_text.filing_date}: {e}"
            )
        return None

    return _aggregate_sentiment_results(
        filing_text=filing_text,
        results=results,
        model_name=model.name,
        model_version=model.version
    )


def compute_sentiment_long(
    filing_sentiments: List[FilingSentiment],
    symbol: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> pl.DataFrame:
    """
    Convert filing sentiments to long-format DataFrame for storage.

    Output schema matches derived metrics pattern:
    [cik, as_of_date, filing_type, fiscal_year, fiscal_quarter,
     metric, value, model_name, model_version]

    :param filing_sentiments: List of FilingSentiment objects
    :param symbol: Optional symbol for logging
    :param logger: Optional logger
    :return: Long-format DataFrame
    """
    if not filing_sentiments:
        if logger:
            logger.debug(f"{symbol}: No filing sentiments to convert")
        return pl.DataFrame()

    # Define metrics to extract
    metrics = [
        # FinBERT sentiment metrics
        "sentiment_score",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "avg_positive_confidence",
        "avg_negative_confidence",
        "chunk_count",
        "text_length",
        # Distribution metrics
        "sentiment_std",
        "sentiment_skew",
        "sentiment_range",
        "extreme_negative_ratio",
        "confidence_std",
        # Loughran-McDonald word ratios
        "word_count",
        "uncertainty_ratio",
        "litigious_ratio",
        "constraining_ratio",
        "weak_modal_ratio",
        "strong_modal_ratio",
        # Readability metrics
        "avg_sentence_length",
        "fog_index"
    ]

    records = []

    for fs in filing_sentiments:
        for metric in metrics:
            value = getattr(fs, metric, None)
            if value is not None:
                records.append({
                    "cik": fs.cik,
                    "as_of_date": fs.filing_date,
                    "filing_type": fs.filing_type,
                    "fiscal_year": fs.fiscal_year,
                    "fiscal_quarter": fs.fiscal_quarter,
                    "metric": metric,
                    "value": float(value),
                    "model_name": fs.model_name,
                    "model_version": fs.model_version
                })

    if not records:
        return pl.DataFrame()

    df = pl.DataFrame(records)

    if logger:
        logger.debug(
            f"{symbol}: Sentiment computation complete: "
            f"{len(filing_sentiments)} filings, {len(df)} metric rows"
        )

    return df


def compute_sentiment_for_cik(
    cik: str,
    filing_texts: List[FilingText],
    model: SentimentModel,
    symbol: Optional[str] = None,
    chunk_size: int = 1500,
    logger: Optional[logging.Logger] = None
) -> pl.DataFrame:
    """
    Compute sentiment for all filings for a CIK.

    Batches all chunks from all filings into a single model.predict() call
    for efficient GPU utilization, then maps results back to filings.

    :param cik: Company CIK
    :param filing_texts: List of extracted filing texts
    :param model: Loaded sentiment model
    :param symbol: Optional symbol for logging
    :param chunk_size: Characters per chunk for inference
    :param logger: Optional logger
    :return: Long-format DataFrame with all sentiment metrics
    """
    log_prefix = f"{symbol or cik}: "

    if not filing_texts:
        if logger:
            logger.debug(f"{log_prefix}No filing texts provided")
        return pl.DataFrame()

    # Step 1: Collect all chunks from all filings with indices
    all_chunks: List[str] = []
    chunk_filing_indices: List[int] = []  # Maps chunk index -> filing index
    filing_chunk_counts: List[int] = []  # Number of chunks per filing

    for filing_idx, filing_text in enumerate(filing_texts):
        if not filing_text.text:
            filing_chunk_counts.append(0)
            continue

        chunks = chunk_text(filing_text.text, chunk_size=chunk_size)
        filing_chunk_counts.append(len(chunks))

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_filing_indices.append(filing_idx)

    if not all_chunks:
        if logger:
            logger.debug(f"{log_prefix}No text chunks to process")
        return pl.DataFrame()

    # Step 2: Run single batched inference on all chunks
    try:
        all_results = model.predict(all_chunks)
    except Exception as e:
        if logger:
            logger.warning(f"{log_prefix}Inference failed: {e}")
        return pl.DataFrame()

    if not all_results:
        if logger:
            logger.debug(f"{log_prefix}No results from model")
        return pl.DataFrame()

    # Step 3: Map results back to filings
    # Group results by filing index
    filing_results: Dict[int, List[SentimentResult]] = {}
    for chunk_idx, result in enumerate(all_results):
        filing_idx = chunk_filing_indices[chunk_idx]
        if filing_idx not in filing_results:
            filing_results[filing_idx] = []
        filing_results[filing_idx].append(result)

    # Step 4: Aggregate sentiment for each filing
    sentiments = []
    for filing_idx, filing_text in enumerate(filing_texts):
        results = filing_results.get(filing_idx, [])
        if not results:
            continue

        sentiment = _aggregate_sentiment_results(
            filing_text=filing_text,
            results=results,
            model_name=model.name,
            model_version=model.version
        )
        if sentiment:
            sentiments.append(sentiment)

    if not sentiments:
        if logger:
            logger.debug(f"{log_prefix}No sentiments computed")
        return pl.DataFrame()

    if logger:
        logger.debug(
            f"{log_prefix}Computed sentiment for "
            f"{len(sentiments)}/{len(filing_texts)} filings "
            f"({len(all_chunks)} chunks total)"
        )

    # Convert to long format
    return compute_sentiment_long(
        filing_sentiments=sentiments,
        symbol=symbol,
        logger=logger
    )
