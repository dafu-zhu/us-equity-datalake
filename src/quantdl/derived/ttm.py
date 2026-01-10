"""
Trailing Twelve Months (TTM) Computation

Computes TTM values from long-format fundamental data in memory.
"""

import datetime as dt
import logging
from collections import defaultdict
from typing import Optional, Iterable

import polars as pl


def compute_ttm_long(
    raw_df: pl.DataFrame,
    logger: Optional[logging.Logger] = None,
    symbol: Optional[str] = None,
    duration_concepts: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    """
    Compute TTM values from long-format fundamental data.

    Expected columns:
    [symbol, as_of_date, accn, form, concept, value, start, end, frame]
    """
    if len(raw_df) == 0:
        if logger:
            logger.debug(f"Empty input DataFrame for {symbol}, skipping TTM computation")
        return pl.DataFrame()

    if duration_concepts is None:
        try:
            from quantdl.collection.fundamental import DURATION_CONCEPTS

            duration_concepts = DURATION_CONCEPTS
        except Exception:
            duration_concepts = set()

    required_cols = {
        "symbol",
        "as_of_date",
        "concept",
        "value",
        "frame",
    }
    missing_cols = required_cols - set(raw_df.columns)
    if missing_cols:
        if logger:
            logger.debug(f"Missing columns for TTM: {sorted(missing_cols)}")
        return pl.DataFrame()

    df = raw_df
    if duration_concepts:
        df = df.filter(pl.col("concept").is_in(list(duration_concepts)))

    if len(df) == 0:
        return pl.DataFrame()

    select_cols = [
        "symbol",
        "as_of_date",
        "concept",
        "value",
        "accn",
        "form",
        "start",
        "end",
        "frame",
    ]
    groups = defaultdict(list)
    for row in df.select(select_cols).to_dicts():
        if row.get("value") is None:
            continue
        if not row.get("as_of_date"):
            continue
        if not row.get("frame"):
            continue
        try:
            as_of_date = dt.datetime.strptime(row["as_of_date"], "%Y-%m-%d").date()
            end_date = (
                dt.datetime.strptime(row["end"], "%Y-%m-%d").date()
                if row.get("end")
                else None
            )
            start_date = (
                dt.datetime.strptime(row["start"], "%Y-%m-%d").date()
                if row.get("start")
                else None
            )
        except ValueError:
            continue

        groups[(row["symbol"], row["concept"])].append(
            {
                "symbol": row["symbol"],
                "concept": row["concept"],
                "as_of_date": as_of_date,
                "accn": row.get("accn"),
                "form": row.get("form"),
                "value": float(row["value"]),
                "start_date": start_date,
                "end_date": end_date,
                "frame": row.get("frame"),
            }
        )

    output_rows = []
    for _, rows in groups.items():
        rows.sort(key=lambda r: r["as_of_date"])
        for idx in range(3, len(rows)):
            window = rows[idx - 3:idx + 1]
            if any(w.get("value") is None for w in window):
                continue
            ttm_value = sum(w["value"] for w in window)
            ttm_start = (
                window[0]["start_date"].isoformat()
                if window[0].get("start_date")
                else None
            )
            ttm_end = (
                rows[idx]["end_date"].isoformat()
                if rows[idx].get("end_date")
                else None
            )

            output_rows.append(
                {
                    "symbol": rows[idx]["symbol"],
                    "as_of_date": rows[idx]["as_of_date"].isoformat(),
                    "accn": rows[idx].get("accn"),
                    "form": rows[idx].get("form"),
                    "concept": rows[idx]["concept"],
                    "value": ttm_value,
                    "start": ttm_start,
                    "end": ttm_end,
                    "frame": rows[idx].get("frame"),
                }
            )

    if not output_rows:
        return pl.DataFrame()

    return pl.DataFrame(output_rows)
