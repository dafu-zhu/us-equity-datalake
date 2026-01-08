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
    [symbol, as_of_date, accn, form, concept, value, start, end, fp]
    """
    if len(raw_df) == 0:
        if logger:
            logger.debug(f"Empty input DataFrame for {symbol}, skipping TTM computation")
        return pl.DataFrame()

    if duration_concepts is None:
        try:
            from collection.fundamental import DURATION_CONCEPTS

            duration_concepts = DURATION_CONCEPTS
        except Exception:
            duration_concepts = set()

    required_cols = {
        "symbol",
        "as_of_date",
        "accn",
        "form",
        "concept",
        "value",
        "start",
        "end",
        "fp",
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

    groups = defaultdict(list)
    for row in df.select(list(required_cols)).to_dicts():
        if row.get("value") is None:
            continue
        if not row.get("as_of_date") or not row.get("end") or not row.get("start"):
            continue
        try:
            as_of_date = dt.datetime.strptime(row["as_of_date"], "%Y-%m-%d").date()
            end_date = dt.datetime.strptime(row["end"], "%Y-%m-%d").date()
            start_date = dt.datetime.strptime(row["start"], "%Y-%m-%d").date()
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
            }
        )

    output_rows = []
    for _, rows in groups.items():
        rows.sort(key=lambda r: (r["end_date"], r["as_of_date"]))
        end_order: list[dt.date] = []
        end_values: dict[dt.date, dict] = {}

        for row in rows:
            end_date = row["end_date"]
            if end_date not in end_values:
                end_order.append(end_date)

            end_values[end_date] = {
                "value": row["value"],
                "start_date": row["start_date"],
            }

            if len(end_order) < 4:
                continue

            last_four = end_order[-4:]
            if any(end not in end_values for end in last_four):
                continue

            ttm_value = sum(end_values[end]["value"] for end in last_four)
            ttm_start = end_values[last_four[0]]["start_date"]
            if ttm_start is None:
                continue

            output_rows.append(
                {
                    "symbol": row["symbol"],
                    "as_of_date": row["as_of_date"].isoformat(),
                    "accn": row.get("accn"),
                    "form": row.get("form"),
                    "concept": row["concept"],
                    "value": ttm_value,
                    "start": ttm_start.isoformat(),
                    "end": last_four[-1].isoformat(),
                    "fp": "TTM",
                }
            )

    if not output_rows:
        return pl.DataFrame()

    return pl.DataFrame(output_rows)
