#!/usr/bin/env python3
"""
Fetch sentiment data for BTC/USDT.
Used by the Sentiment Analyst sub-agent.
Usage: python fetch_sentiment_data.py [date] [symbol]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from tradingagents.dataflows.ccxt_crypto import (
    get_crypto_funding_rate,
    get_crypto_open_interest,
    get_fear_greed_index,
    get_crypto_liquidations_summary,
)

def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
    
    print(f"=== SENTIMENT DATA FOR {symbol} — {trade_date} ===\n")
    
    # 1. Fear & Greed Index
    print("## 1. Crypto Fear & Greed Index")
    try:
        print(get_fear_greed_index())
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Funding Rate
    print("\n## 2. Funding Rate")
    try:
        print(get_crypto_funding_rate(f"{symbol}:USDT"))
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Open Interest
    print("\n## 3. Open Interest")
    try:
        print(get_crypto_open_interest(f"{symbol}:USDT"))
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Liquidation Info
    print("\n## 4. Liquidation Risk Assessment")
    try:
        print(get_crypto_liquidations_summary(symbol))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
