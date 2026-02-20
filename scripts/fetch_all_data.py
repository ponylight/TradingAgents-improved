#!/usr/bin/env python3
"""
Fetch all data for BTC/USDT analysis in one go.
Saves results to a JSON file for sub-agents to consume.
Usage: python fetch_all_data.py [date] [symbol] [output_dir]
"""

import sys
import os
import json
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
    get_fear_greed_index,
    get_crypto_liquidations_summary,
)
from tradingagents.dataflows.macro_data import (
    get_dxy_data,
    get_treasury_yields,
    get_sp500_data,
    get_fred_data,
    get_economic_calendar_summary,
)


def fetch_safe(fn, *args, **kwargs):
    """Execute a fetch function safely, returning error string on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return f"ERROR: {e}"


def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    
    date_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_30d = (date_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    start_60d = (date_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    start_1y = (date_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    
    perp_symbol = f"{symbol}:USDT"
    
    print(f"Fetching data for {symbol} on {trade_date}...")
    
    data = {
        "symbol": symbol,
        "trade_date": trade_date,
        "fetched_at": datetime.now().isoformat(),
    }
    
    # === MARKET DATA ===
    print("  [1/5] OHLCV...")
    data["ohlcv_30d"] = fetch_safe(get_crypto_ohlcv, symbol, start_30d, trade_date, "1d")
    
    print("  [2/5] Technical indicators...")
    data["indicators"] = {}
    for ind in ["rsi", "macd", "macdh", "td_sequential", "fib_channel", "atr", "close_50_sma", "close_200_sma", "boll_ub", "boll_lb", "mfi"]:
        data["indicators"][ind] = fetch_safe(get_crypto_indicators, symbol, ind, trade_date, 30)
    
    print("  [3/5] Derivatives data...")
    data["funding_rate"] = fetch_safe(get_crypto_funding_rate, perp_symbol)
    data["open_interest"] = fetch_safe(get_crypto_open_interest, perp_symbol)
    data["orderbook"] = fetch_safe(get_crypto_orderbook, symbol, 10)
    
    # === SENTIMENT DATA ===
    print("  [4/5] Sentiment...")
    data["fear_greed"] = fetch_safe(get_fear_greed_index)
    data["liquidation_info"] = fetch_safe(get_crypto_liquidations_summary, symbol)
    
    # === MACRO DATA ===
    print("  [5/6] Macro...")
    data["dxy"] = fetch_safe(get_dxy_data, start_60d, trade_date)
    data["treasury_yields"] = fetch_safe(get_treasury_yields, start_60d, trade_date)
    data["sp500"] = fetch_safe(get_sp500_data, start_60d, trade_date)
    data["fed_funds"] = fetch_safe(get_fred_data, "FEDFUNDS", start_1y, trade_date)
    data["cpi"] = fetch_safe(get_fred_data, "CPIAUCSL", start_1y, trade_date)
    data["m2_supply"] = fetch_safe(get_fred_data, "M2SL", start_1y, trade_date)
    data["economic_calendar"] = get_economic_calendar_summary()
    
    # === CROSS-EXCHANGE ===
    print("  [6/6] Binance comparison...")
    data["binance_comparison"] = fetch_binance_comparison()
    
    # Save
    output_file = os.path.join(output_dir, f"btc_data_{trade_date}.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ All data saved to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")

def fetch_binance_comparison():
    """Fetch Binance BTC data for cross-exchange comparison."""
    import ccxt
    try:
        binance = ccxt.binance({'enableRateLimit': True})
        ticker = binance.fetch_ticker('BTC/USDT')
        orderbook = binance.fetch_order_book('BTC/USDT', limit=10)
        
        total_bid = sum(p * a for p, a in orderbook['bids'][:10])
        total_ask = sum(p * a for p, a in orderbook['asks'][:10])
        imbalance = total_bid / total_ask if total_ask > 0 else 0
        
        return (
            f"# Binance BTC/USDT Cross-Exchange Comparison\n"
            f"Price: ${ticker['last']:,.2f}\n"
            f"24h Volume: {ticker.get('baseVolume', 'N/A')} BTC\n"
            f"24h Change: {ticker.get('percentage', 'N/A')}%\n"
            f"Bid/Ask Imbalance: {imbalance:.4f} ({'Bullish' if imbalance > 1 else 'Bearish'})\n"
            f"Spread: ${orderbook['asks'][0][0] - orderbook['bids'][0][0]:,.2f}\n"
        )
    except Exception as e:
        return f"ERROR fetching Binance data: {e}"


if __name__ == "__main__":
    main()
