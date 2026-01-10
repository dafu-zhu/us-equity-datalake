"""
Derived Fundamental Metrics Computation

Computes derived fundamental metrics from long-format TTM data.

This module provides in-memory computation for the storage pipeline.
Derived data is stored separately from raw data:
- Input TTM: data/derived/features/fundamental/{symbol}/ttm.parquet (long)
- Output metrics: data/derived/features/fundamental/{symbol}/metrics.parquet (long)
"""

import polars as pl
import logging
from typing import Optional


def compute_derived(
    raw_df: pl.DataFrame,
    logger: Optional[logging.Logger] = None,
    symbol: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute derived fundamental metrics from long-format TTM data.

    Takes a TTM long DataFrame and computes 24 derived metrics.
    Returns long-format derived metrics.

    All formulas based on data/xbrl/fundamental.xlsx (Priority = 3 rows).

    :param raw_df: TTM long DataFrame with columns
                   [symbol, as_of_date, concept, value]
    :param logger: Optional logger for debug messages
    :param symbol: Optional symbol for logging
    :return: Long-format DataFrame with columns [symbol, as_of_date, metric, value]
             Returns empty DataFrame if input is empty

    Derived concepts (24 total):
    - Profitability: grs_pft, grs_mgn, op_mgn, net_mgn, ebitda
    - Cash Flow: fcf, fcf_mgn, capex_ratio
    - Balance Sheet: ttl_dbt, net_dbt, wc
    - Returns: roa, roe, roic, nopat, etr, avg_ast, avg_eqt, inv_cap
    - Growth: rev_grw, ast_grw, inv_rt
    - Accruals: acc, wc_acc

    Example:
        >>> raw_df = ttm_df_long
        >>> derived_df = compute_derived(raw_df, logger, symbol='RKLB')
        >>> # Store separately:
        >>> # - raw_df → data/derived/features/fundamental/RKLB/ttm.parquet
        >>> # - derived_df → data/derived/features/fundamental/RKLB/metrics.parquet
    """
    # Return empty DataFrame if input is empty
    if len(raw_df) == 0:
        if logger:
            logger.debug(f"Empty input DataFrame for {symbol}, skipping derived computation")
        return pl.DataFrame()

    log_prefix = f"{symbol}: " if symbol else ""
    try:
        required_cols = {
            "symbol",
            "as_of_date",
            "concept",
            "value",
        }
        missing_cols = required_cols - set(raw_df.columns)
        if missing_cols:
            if logger:
                logger.debug(f"{log_prefix}Missing columns for derived: {sorted(missing_cols)}")
            return pl.DataFrame()

        df = raw_df.clone()
        if "as_of_date" in df.columns:
            df = df.sort("as_of_date")

        wide_df = df.pivot(
            values="value",
            index=["symbol", "as_of_date"],
            on="concept",
            aggregate_function="first",
        )

        required_inputs = [
            "rev", "cor", "op_inc", "net_inc", "dna",
            "std", "ltd", "cce", "ca", "cl",
            "cfo", "capex", "ta", "te",
            "inc_tax_exp", "ibt",
        ]
        missing_inputs = [col for col in required_inputs if col not in wide_df.columns]
        if missing_inputs:
            wide_df = wide_df.with_columns([pl.lit(None).alias(col) for col in missing_inputs])

        # Helper functions for safe arithmetic
        def safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
            return pl.when(denominator.is_not_null() & (denominator != 0)).then(
                numerator / denominator
            ).otherwise(None)

        def safe_subtract(a: pl.Expr, b: pl.Expr) -> pl.Expr:
            return pl.when(a.is_not_null() & b.is_not_null()).then(a - b).otherwise(None)

        def safe_add(a: pl.Expr, b: pl.Expr) -> pl.Expr:
            return pl.when(a.is_not_null() & b.is_not_null()).then(a + b).otherwise(None)

        def safe_multiply(a: pl.Expr, b: pl.Expr) -> pl.Expr:
            return pl.when(a.is_not_null() & b.is_not_null()).then(a * b).otherwise(None)

        # 1. Profitability Metrics
        if logger:
            logger.debug(f"{log_prefix}Computing profitability metrics")

        df = wide_df.sort(["symbol", "as_of_date"]).with_columns([
            # Gross Profit = revenue - cost of revenue
            safe_subtract(pl.col("rev"), pl.col("cor")).alias("grs_pft"),
        ]).with_columns([
            # Gross Margin = gross profit / revenue
            safe_divide(pl.col("grs_pft"), pl.col("rev")).alias("grs_mgn"),
            # Operating Margin = operating income / revenue
            safe_divide(pl.col("op_inc"), pl.col("rev")).alias("op_mgn"),
            # Net Margin = net income / revenue
            safe_divide(pl.col("net_inc"), pl.col("rev")).alias("net_mgn"),
            # EBITDA = operating income + depreciation and amortization
            safe_add(pl.col("op_inc"), pl.col("dna")).alias("ebitda"),
        ])

        # 2. Balance Sheet Constructs
        if logger:
            logger.debug(f"{log_prefix}Computing balance sheet constructs")

        df = df.with_columns([
            # Total Debt = short term debt + long term debt
            safe_add(pl.col("std"), pl.col("ltd")).alias("ttl_dbt"),
        ]).with_columns([
            # Net Debt = total debt - cash
            safe_subtract(pl.col("ttl_dbt"), pl.col("cce")).alias("net_dbt"),
            # Working Capital = current assets - current liabilities
            safe_subtract(pl.col("ca"), pl.col("cl")).alias("wc"),
        ])

        # 3. Cash Flow Metrics
        if logger:
            logger.debug(f"{log_prefix}Computing cash flow metrics")

        df = df.with_columns([
            # Free Cash Flow = CFO - CapEx
            safe_subtract(pl.col("cfo"), pl.col("capex")).alias("fcf"),
        ]).with_columns([
            # FCF Margin = free cash flow / revenue
            safe_divide(pl.col("fcf"), pl.col("rev")).alias("fcf_mgn"),
            # CapEx Ratio = capex / total assets
            safe_divide(pl.col("capex"), pl.col("ta")).alias("capex_ratio"),
        ])

        # 4. Return Metrics
        if logger:
            logger.debug(f"{log_prefix}Computing return metrics")

        df = df.with_columns([
            # Average Assets = (total assets(t) + total assets(t-1Y)) / 2
            ((pl.col("ta") + pl.col("ta").shift(4).over("symbol")) / 2).alias("avg_ast"),
            # Average Equity = (total equity(t) + total equity(t-1Y)) / 2
            ((pl.col("te") + pl.col("te").shift(4).over("symbol")) / 2).alias("avg_eqt"),
            # Effective Tax Rate = income tax expense / income before tax
            safe_divide(pl.col("inc_tax_exp"), pl.col("ibt")).alias("etr"),
        ]).with_columns([
            # ROA = net income / avg assets
            safe_divide(pl.col("net_inc"), pl.col("avg_ast")).alias("roa"),
            # ROE = net income / avg equity
            safe_divide(pl.col("net_inc"), pl.col("avg_eqt")).alias("roe"),
            # NOPAT = operating income × (1 − effective tax rate)
            safe_multiply(pl.col("op_inc"), (1 - pl.col("etr"))).alias("nopat"),
        ]).with_columns([
            # Invested Capital = total equity + total debt - cash
            safe_subtract(
                safe_add(pl.col("te"), pl.col("ttl_dbt")),
                pl.col("cce")
            ).alias("inv_cap"),
        ]).with_columns([
            # ROIC = NOPAT / invested capital
            safe_divide(pl.col("nopat"), pl.col("inv_cap")).alias("roic"),
        ])

        # 5. Growth Metrics
        if logger:
            logger.debug(f"{log_prefix}Computing growth metrics")

        df = df.with_columns([
            # Revenue Growth = revenue(t) - revenue(t-1)
            (pl.col("rev") - pl.col("rev").shift(1).over("symbol")).alias("rev_grw"),
            # Asset Growth = total assets(t) - total assets(t-1)
            (pl.col("ta") - pl.col("ta").shift(1).over("symbol")).alias("ast_grw"),
            # Investment Rate = capex / total assets
            safe_divide(pl.col("capex"), pl.col("ta")).alias("inv_rt"),
        ])

        # 6. Accruals
        if logger:
            logger.debug(f"{log_prefix}Computing accruals")

        df = df.with_columns([
            # Accruals = net income - CFO
            safe_subtract(pl.col("net_inc"), pl.col("cfo")).alias("acc"),
            # WC Accruals = Delta(working capital) - depreciation and amortization
            safe_subtract(
                (pl.col("wc") - pl.col("wc").shift(1).over("symbol")),
                pl.col("dna")
            ).alias("wc_acc"),
        ])

        # Extract ONLY derived columns (keys + 24 derived metrics)
        key_columns = [
            'symbol', 'as_of_date'
        ]
        derived_columns = [
            # Profitability (5)
            'grs_pft', 'grs_mgn', 'op_mgn', 'net_mgn', 'ebitda',
            # Balance Sheet (3)
            'ttl_dbt', 'net_dbt', 'wc',
            # Cash Flow (3)
            'fcf', 'fcf_mgn', 'capex_ratio',
            # Returns (8)
            'avg_ast', 'avg_eqt', 'etr', 'roa', 'roe', 'nopat', 'inv_cap', 'roic',
            # Growth (3)
            'rev_grw', 'ast_grw', 'inv_rt',
            # Accruals (2)
            'acc', 'wc_acc'
        ]

        select_cols = [col for col in key_columns if col in df.columns] + derived_columns
        derived_df = df.select(select_cols)

        long_df = derived_df.melt(
            id_vars=key_columns,
            value_vars=derived_columns,
            variable_name="metric",
            value_name="value",
        ).drop_nulls(subset=["value"])

        if logger:
            logger.debug(
                f"{log_prefix}Derived computation complete: {long_df.shape[0]} rows, "
                f"{long_df.shape[1]} columns (long format)"
            )

        return long_df

    except Exception as e:
        if logger:
            logger.error(f"{log_prefix}Failed to compute derived fundamentals: {e}", exc_info=True)
        return pl.DataFrame()  # Return empty DataFrame on error
