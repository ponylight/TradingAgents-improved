"""Fetch 4 years of BTC/USDT 4h candles from Binance via ccxt. Cache locally."""

import os
import time
import ccxt
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "btc_usdt_4h.csv")

SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
FOUR_YEARS_MS = 4 * 365.25 * 24 * 60 * 60 * 1000  # ~4 years in ms


def fetch_ohlcv(use_cache: bool = True) -> pd.DataFrame:
    """Fetch 4yr of 4h BTC/USDT candles. Returns DataFrame with OHLCV columns."""
    if use_cache and os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
        return df

    exchange = ccxt.binance({"enableRateLimit": True})
    now = exchange.milliseconds()
    since = int(now - FOUR_YEARS_MS)

    all_candles = []
    print(f"Fetching {SYMBOL} {TIMEFRAME} candles from Binance...")

    while since < now:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1  # next ms after last candle
            print(f"  Fetched {len(all_candles)} candles so far (up to {pd.Timestamp(candles[-1][0], unit='ms')})")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"Saved {len(df)} candles to {CACHE_FILE}")
    return df


if __name__ == "__main__":
    df = fetch_ohlcv(use_cache=False)
    print(f"\nData range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Total candles: {len(df)}")
    print(df.tail())
