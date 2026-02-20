#!/usr/bin/env python3
"""
Fetch macroeconomic data relevant to BTC.
Used by the Macro Analyst sub-agent.
Usage: python fetch_macro_data.py [date]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from tradingagents.dataflows.macro_data import (
    get_dxy_data,
    get_treasury_yields,
    get_sp500_data,
    get_fred_data,
    get_economic_calendar_summary,
)

def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    
    date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_60d = (date_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    start_1y = (date_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"=== MACRO DATA — {trade_date} ===\n")
    
    # 1. DXY
    print("## 1. US Dollar Index (DXY) — 60 days")
    try:
        print(get_dxy_data(start_60d, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Treasury Yields
    print("\n## 2. Treasury Yields")
    try:
        print(get_treasury_yields(start_60d, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. S&P 500
    print("\n## 3. S&P 500 — 60 days")
    try:
        print(get_sp500_data(start_60d, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Fed Funds Rate
    print("\n## 4. Federal Funds Rate — 1 year")
    try:
        print(get_fred_data("FEDFUNDS", start_1y, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. CPI
    print("\n## 5. CPI (Inflation) — 1 year")
    try:
        print(get_fred_data("CPIAUCSL", start_1y, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 6. M2 Money Supply
    print("\n## 6. M2 Money Supply — 1 year")
    try:
        print(get_fred_data("M2SL", start_1y, trade_date))
    except Exception as e:
        print(f"Error: {e}")
    
    # 7. Economic Calendar
    print("\n## 7. Economic Calendar")
    print(get_economic_calendar_summary())

if __name__ == "__main__":
    main()
