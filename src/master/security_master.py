import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
import time
import requests
import datetime as dt
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from utils.logger import setup_logger

load_dotenv()

class SecurityMaster:
    def __init__(self, db: Optional[wrds.Connection] = None):
        if db is None:
            username = os.getenv('WRDS_USERNAME')
            password = os.getenv('WRDS_PASSWORD')
            self.db = wrds.Connection(
                wrds_username=username,
                wrds_password=password
            )
        else:
            self.db = db

        self.logger = setup_logger(
            name="security_master",
            log_dir=Path("data/logs/master"),
            level=logging.INFO
        )
        self.cik_cusip = self.cik_cusip_mapping()
        self.master_tb = self.master_table()
    
    def cik_cusip_mapping(self) -> pl.DataFrame:
        """
        All historical mappings, updated until Dec 31, 2024

        Schema: (permno, symbol, company, cik, cusip, start_date, end_date)
        """
        query = """
        SELECT DISTINCT
            a.kypermno, 
            a.ticker, 
            a.tsymbol,
            a.comnam, 
            a.ncusip, 
            b.cik,
            a.namedt, 
            a.nameenddt
        FROM 
            crsp.s6z_nam AS a
        LEFT JOIN
            wrdssec_common.wciklink_cusip AS b
            ON SUBSTR(a.ncusip, 1, 8) = SUBSTR(b.cusip, 1, 8)
            AND (b.cik IS NULL OR a.namedt <= b.cikdate2)
            AND (b.cik IS NULL OR a.nameenddt >= b.cikdate1)
        WHERE
            a.shrcd IN (10, 11)
        ORDER BY 
            a.kypermno, a.namedt
        """

        # Execute and load into a DataFrame
        map_df = self.db.raw_sql(query)
        map_df['namedt'] = pd.to_datetime(map_df['namedt'])
        map_df['nameenddt'] = pd.to_datetime(map_df['nameenddt'])

        # Forward-fill CIK for records with NULL CIK (due to stale CIK mapping data)
        map_df = map_df.sort_values(['kypermno', 'ncusip', 'namedt'])
        map_df['cik'] = map_df.groupby(['kypermno', 'ncusip'])['cik'].ffill()

        result = (
            map_df.groupby(
                ['kypermno', 'cik', 'ticker', 'tsymbol', 'comnam', 'ncusip'],
                dropna=False
            ).agg({
                'namedt': 'min',
                'nameenddt': 'max'
            })
            .reset_index()
            .sort_values(['kypermno', 'namedt'])
            .dropna(subset=['tsymbol'])
        )

        pl_map = pl.DataFrame(result).with_columns(
            pl.col('kypermno').cast(pl.Int32).alias('permno'),
            pl.col('tsymbol').alias('symbol'),
            pl.col('comnam').alias('company'),
            pl.col('ncusip').alias('cusip'),
            pl.col('namedt').cast(pl.Date).alias('start_date'),
            pl.col('nameenddt').cast(pl.Date).alias('end_date')
        ).select(['permno', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date'])

        return pl_map
    
    def security_map(self) -> pl.DataFrame:
        """
        Maps security_id with unique permno, replacing permno

        Schema: (security_id, permno)
        """
        unique_permnos = self.cik_cusip.select('permno').unique(maintain_order=True)
        result = pl.DataFrame({
            'security_id': range(1000, 1000 + len(unique_permnos)),
            'permno': unique_permnos
        })

        return result
    
    def master_table(self) -> pl.DataFrame:
        """
        Create comprehensive table with security_id as master key, includes historically used symbols, cik, cusip

        Schema: (security_id, symbol, company, cik, cusip, start_date, end_date)
        """
        security_map = self.security_map()
        
        full_history = self.cik_cusip.join(security_map, on='permno', how='left')
        result = full_history[['security_id', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date']]

        return result
    
    def auto_resolve(self, symbol: str, day: str) -> int:
        """
        Smart resolve unmatched symbol and query day.
        Route to the security that is active on 'day', and have most recently used / use 'symbol' in the future
        """
        self.logger.info(f"auto_resolve triggered for symbol='{symbol}' on date='{day}'")

        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()

        # Find all securities that ever used this symbol
        candidates = self.master_tb.filter(
            pl.col('symbol').eq(symbol)
        ).select('security_id').unique()

        if candidates.is_empty():
            self.logger.warning(f"auto_resolve failed: symbol '{symbol}' never existed in security master")
            raise ValueError(f"Symbol '{symbol}' never existed in security master")

        # For each candidate, check if it was active on target date (under ANY symbol)
        active_securities = []
        for candidate_sid in candidates['security_id']:
            was_active = self.master_tb.filter(
                pl.col('security_id').eq(candidate_sid),
                pl.col('start_date').le(date_check),
                pl.col('end_date').ge(date_check)
            )
            if not was_active.is_empty():
                # Find when this security used the queried symbol
                symbol_usage = self.master_tb.filter(
                    pl.col('security_id').eq(candidate_sid),
                    pl.col('symbol').eq(symbol)
                ).select(['start_date', 'end_date']).head(1)

                active_securities.append({
                    'sid': candidate_sid,
                    'symbol_start': symbol_usage['start_date'][0],
                    'symbol_end': symbol_usage['end_date'][0]
                })

        # Resolve ambiguity
        if len(active_securities) == 0:
            self.logger.warning(
                    f"auto_resolve failed: symbol '{symbol}' exists but associated security "
                    f"was not active on {day}"
                )
            raise ValueError(
                f"Symbol '{symbol}' exists but the associated security was not active on {day}"
            )
        elif len(active_securities) == 1:
            sid = active_securities[0]['sid']
        else:
            # Multiple securities used this symbol and were active on target date
            # Pick the one that used this symbol closest to the query date
            def distance_to_date(sec):
                """Calculate temporal distance from query date to when symbol was used"""
                if date_check < sec['symbol_start']:
                    return (sec['symbol_start'] - date_check).days
                elif date_check > sec['symbol_end']:
                    return (date_check - sec['symbol_end']).days
                else:
                    return 0

            # Pick security with minimum distance
            best_match = min(active_securities, key=distance_to_date)
            sid = best_match['sid']

            self.logger.info(
                f"auto_resolve: Multiple candidates found, selected security_id={sid} "
                f"(minimum temporal distance)"
            )
            
        return sid

    def get_security_id(self, symbol: str, day: str, auto_resolve: bool=True) -> int:
        """
        Finds the Internal ID for a specific Symbol at a specific point in time.

        :param auto_resolve: If the security has name change, resolve symbol to the nearest security
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()
            
        match = self.master_tb.filter(
            pl.col('symbol').eq(symbol),
            pl.col('start_date').le(date_check),
            pl.col('end_date').ge(date_check)
        )
        
        if match.is_empty():
            if not auto_resolve:
                raise ValueError(f"Symbol {symbol} not found in day {day}")
            else:
                return self.auto_resolve(symbol, day)
        
        result = match.head(1).select('security_id').item()

        return result
    
    def get_symbol_history(self, sid: int) -> List[Tuple[str, str, str]]:
        """
        Full list of symbol usage history for a given security_id

        Example: [('META', '2022-06-09', '2024-12-31'), ('FB', '2012-05-18', '2022-06-08')]
        """
        mtb = self.master_table()
        sid_df = mtb.filter(
            pl.col('security_id').eq(sid)
        ).group_by('symbol').agg(
            pl.col('start_date').min(),
            pl.col('end_date').max()
        )

        hist = sid_df.select(['symbol', 'start_date', 'end_date']).rows()
        isoformat_hist = [(sym, start.isoformat(), end.isoformat()) for sym, start, end in hist]

        return isoformat_hist
    
    def sid_to_permno(self, sid: int) -> int:
        security_map = self.security_map()
        permno = (
            security_map.filter(
                pl.col('security_id').eq(sid)
            )
            .select('permno')
            .head(1)
            .item()
        )
        return permno
    
    def sid_to_symbol(self, sid: int, day: str):
        pass

    def sid_to_cik(self, sid: int, day: str):
        pass

    def sid_to_cusip(self, sid: int, day: str):
        pass

    def close(self):
        """Close WRDS connection"""
        self.db.close()
    

if __name__ == "__main__":
    sm = SecurityMaster()

    df = sm.cik_cusip_mapping()

    df = df.filter(pl.col('symbol').eq('RCM'))
    print(df)

    # Scenario A: You ask for "AH" in 2012
    print(f"Who was AH in 2012? ID: {sm.get_security_id('AH', '2012-06-01')}")

    # Scenario B: You ask for "RCM" in 2012 (The Trap)
    print(f"Who was RCM in 2012? ID: {sm.get_security_id('RCM', '2012-06-01')}")
    # Output: ID: None (Correct! RCM didn't exist then)

    # Scenario C: You ask for "RCM" in 2020
    print(f"Who was RCM in 2020? ID: {sm.get_security_id('RCM', '2020-01-01')}")

    print(f"Who was MSFT in 2024? ID: {sm.get_security_id('MSFT', '2024-01-01')}")

    print(f"Who was BRKB in 2022? ID: {sm.get_security_id('BRKB', '2022-01-01')}")

    sm.close()