#!/usr/bin/env python3
"""
Fetch market/technical data for BTC/USDT.
Used by the Market Analyst sub-agent.
Usage: python fetch_market_data.py [date] [symbol]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from tradingagents.dataflows.ccxt_crypto import (
    get_crypto_ohlcv,
    get_crypto_indicators,
    get_crypto_funding_rate,
    get_crypto_open_interest,
    get_crypto_orderbook,
)

def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
    
    date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_30d = (date_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    start_90d = (date_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    
    print(f"=== MARKET DATA FOR {symbol} — {trade_date} ===\n")
    
    # 1. OHLCV (last 30 days, daily)
    print("## 1. OHLCV Data (30 days, daily)")
    try:
        print(get_crypto_ohlcv(symbol, start_30d, trade_date, timeframe="1d"))
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Key technical indicators
    indicators = ["rsi", "macd", "macdh", "td_sequential", "fib_channel", "atr", "close_50_sma", "boll_ub", "boll_lb"]
    for ind in indicators:
        print(f"\n## 2. Indicator: {ind}")
        try:
            print(get_crypto_indicators(symbol, ind, trade_date, look_back_days=30))
        except Exception as e:
            print(f"Error: {e}")
    
    # 3. Funding rate
    print("\n## 3. Funding Rate")
    try:
        print(get_crypto_funding_rate(f"{symbol}:USDT"))
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Open interest
    print("\n## 4. Open Interest")
    try:
        print(get_crypto_open_interest(f"{symbol}:USDT"))
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Order book
    print("\n## 5. Order Book (top 10)")
    try:
        print(get_crypto_orderbook(symbol, depth=10))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
