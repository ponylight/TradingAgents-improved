"""
Crypto data provider using ccxt for BTC/USDT trading on Bybit.
Replaces yfinance for cryptocurrency market data.
"""

import ccxt
import pandas as pd
import os
import json
from typing import Annotated
from datetime import datetime, timedelta
from .config import get_config
from .utils import is_cache_stale
from .exceptions import DataFetchError


def _get_exchange(exchange_id: str = "bybit", api_key: str = None, secret: str = None, testnet: bool = False):
    """Create and return a ccxt exchange instance."""
    exchange_class = getattr(ccxt, exchange_id)
    config = {
        'enableRateLimit': True,
    }
    if api_key:
        config['apiKey'] = api_key
    if secret:
        config['secret'] = secret
    if testnet:
        config['sandbox'] = True
    
    exchange = exchange_class(config)
    return exchange


def _fetch_ohlcv_all(exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int, limit: int = 1000):
    """Fetch all OHLCV data between since and end, handling pagination."""
    all_candles = []
    current_since = since_ms
    
    while current_since < end_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        # Move to after the last candle
        current_since = candles[-1][0] + 1
        if len(candles) < limit:
            break
    
    return all_candles


def get_crypto_ohlcv(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    timeframe: str = "1d",
    exchange_id: str = "bybit",
) -> str:
    """
    Fetch OHLCV crypto data via ccxt.
    Equivalent to get_YFin_data_online but for crypto.
    """
    config = get_config()
    cache_dir = config.get("data_cache_dir", "data")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file
    safe_symbol = symbol.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{safe_symbol}-{exchange_id}-{timeframe}-{start_date}-{end_date}.csv")
    cache_ttl = config.get("cache_ttl_hours", 24)
    
    if os.path.exists(cache_file) and not is_cache_stale(cache_file, cache_ttl):
        data = pd.read_csv(cache_file)
    else:
        try:
            exchange = _get_exchange(exchange_id)
            
            since_ms = exchange.parse8601(f"{start_date}T00:00:00Z")
            end_ms = exchange.parse8601(f"{end_date}T23:59:59Z")
            
            candles = _fetch_ohlcv_all(exchange, symbol, timeframe, since_ms, end_ms)
            
            if not candles:
                return f"No data found for '{symbol}' between {start_date} and {end_date}"
            
            data = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
            
            # Filter to requested date range
            data_dt = pd.to_datetime(data["Date"])
            data = data[(data_dt >= start_date) & (data_dt <= f"{end_date} 23:59:59")]
            
            # Round values
            for col in ["Open", "High", "Low", "Close"]:
                data[col] = data[col].round(2)
            data["Volume"] = data["Volume"].round(4)
            
            data.to_csv(cache_file, index=False)
            
        except Exception as e:
            raise DataFetchError(f"Error fetching crypto data for {symbol}: {e}") from e
    
    csv_string = data.to_csv(index=False)
    header = f"# Crypto OHLCV data for {symbol} from {start_date} to {end_date} ({timeframe})\n"
    header += f"# Exchange: {exchange_id}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return header + csv_string


def get_crypto_indicators(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
    timeframe: str = "1d",
    exchange_id: str = "bybit",
) -> str:
    """
    Calculate technical indicators for crypto using stockstats.
    Equivalent to get_stock_stats_indicators_window but for crypto.
    """
    from stockstats import wrap
    from .y_finance import _calculate_td_sequential, _calculate_fibonacci_channel
    
    config = get_config()
    cache_dir = config.get("data_cache_dir", "data")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Indicator descriptions (same as y_finance.py)
    best_ind_params = {
        "close_50_sma": "50 SMA: Medium-term trend indicator.",
        "close_200_sma": "200 SMA: Long-term trend benchmark.",
        "close_10_ema": "10 EMA: Responsive short-term average.",
        "macd": "MACD: Momentum via EMA differences.",
        "macds": "MACD Signal: EMA smoothing of MACD line.",
        "macdh": "MACD Histogram: Gap between MACD and signal.",
        "rsi": "RSI: Momentum overbought/oversold indicator.",
        "boll": "Bollinger Middle: 20 SMA basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: 2 std devs above middle.",
        "boll_lb": "Bollinger Lower Band: 2 std devs below middle.",
        "atr": "ATR: Average true range volatility measure.",
        "vwma": "VWMA: Volume-weighted moving average.",
        "mfi": "MFI: Money Flow Index using price and volume.",
        "td_sequential": "TD Sequential (DeMark): Counter-trend exhaustion indicator.",
        "fib_channel": "Fibonacci Channel: Dynamic support/resistance levels.",
    }
    
    if indicator not in best_ind_params:
        raise ValueError(f"Indicator {indicator} not supported. Choose from: {list(best_ind_params.keys())}")
    
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # Fetch extra data for indicator calculation (need history for moving averages)
    fetch_start = (curr_date_dt - timedelta(days=look_back_days + 300)).strftime("%Y-%m-%d")
    
    safe_symbol = symbol.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"{safe_symbol}-{exchange_id}-{timeframe}-{fetch_start}-{curr_date}.csv")
    cache_ttl = config.get("cache_ttl_hours", 24)
    
    if os.path.exists(cache_file) and not is_cache_stale(cache_file, cache_ttl):
        data = pd.read_csv(cache_file)
    else:
        try:
            exchange = _get_exchange(exchange_id)
            since_ms = exchange.parse8601(f"{fetch_start}T00:00:00Z")
            end_ms = exchange.parse8601(f"{curr_date}T23:59:59Z")
            
            candles = _fetch_ohlcv_all(exchange, symbol, timeframe, since_ms, end_ms)
            
            if not candles:
                return f"No data found for '{symbol}'"
            
            data = pd.DataFrame(candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
            data.to_csv(cache_file, index=False)
            
        except Exception as e:
            raise DataFetchError(f"Error fetching crypto data for indicators: {e}") from e
    
    # Calculate indicator using stockstats
    _CUSTOM_INDICATORS = {"td_sequential", "fib_channel"}
    
    df = wrap(data.copy())
    
    if indicator in _CUSTOM_INDICATORS:
        if indicator == "td_sequential":
            df[indicator] = _calculate_td_sequential(df)
        elif indicator == "fib_channel":
            df[indicator] = _calculate_fibonacci_channel(df)
    else:
        _ = df[indicator]  # Trigger stockstats calculation
    
    # Build results for the look-back window
    before = curr_date_dt - timedelta(days=look_back_days)
    
    ind_string = ""
    for _, row in df.iterrows():
        date_str = row["Date"]
        try:
            row_date = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        if before <= row_date <= curr_date_dt:
            val = row[indicator] if pd.notna(row[indicator]) else "N/A"
            ind_string += f"{date_str}: {val}\n"
    
    result_str = (
        f"## {indicator} values for {symbol} from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )
    
    return result_str


def get_crypto_funding_rate(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    exchange_id: str = "bybit",
) -> str:
    """Get current and recent funding rates for a perpetual contract."""
    try:
        exchange = _get_exchange(exchange_id)
        
        # Fetch funding rate history
        funding_rates = exchange.fetch_funding_rate_history(symbol, limit=100)
        
        if not funding_rates:
            return f"No funding rate data for {symbol}"
        
        lines = [f"# Funding Rate History for {symbol} ({exchange_id})\n"]
        for fr in funding_rates[-20:]:  # Last 20 entries
            timestamp = datetime.fromtimestamp(fr['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
            rate = fr.get('fundingRate', 'N/A')
            if isinstance(rate, (int, float)):
                rate = f"{rate:.6f} ({rate * 100:.4f}%)"
            lines.append(f"{timestamp}: {rate}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise DataFetchError(f"Error fetching funding rates: {e}") from e


def get_crypto_open_interest(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    exchange_id: str = "bybit",
) -> str:
    """Get open interest data for a perpetual contract."""
    try:
        exchange = _get_exchange(exchange_id)
        
        # Fetch open interest
        oi = exchange.fetch_open_interest(symbol)
        
        if not oi:
            return f"No open interest data for {symbol}"
        
        lines = [f"# Open Interest for {symbol} ({exchange_id})\n"]
        lines.append(f"Open Interest: {oi.get('openInterestAmount', 'N/A')} contracts")
        lines.append(f"Value: ${oi.get('openInterestValue', 'N/A')}")
        lines.append(f"Timestamp: {datetime.fromtimestamp(oi['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M') if oi.get('timestamp') else 'N/A'}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise DataFetchError(f"Error fetching open interest: {e}") from e


def get_crypto_orderbook(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    depth: int = 20,
    exchange_id: str = "bybit",
) -> str:
    """Get order book depth for analysis of support/resistance."""
    try:
        exchange = _get_exchange(exchange_id)
        orderbook = exchange.fetch_order_book(symbol, limit=depth)
        
        if not orderbook:
            return f"No order book data for {symbol}"
        
        lines = [f"# Order Book for {symbol} ({exchange_id}) - Top {depth} levels\n"]
        
        lines.append("## Asks (Sell Orders - ascending):")
        for price, amount in orderbook['asks'][:depth]:
            lines.append(f"  ${price:,.2f} | {amount:.4f} BTC | ${price * amount:,.2f}")
        
        lines.append("\n## Bids (Buy Orders - descending):")
        for price, amount in orderbook['bids'][:depth]:
            lines.append(f"  ${price:,.2f} | {amount:.4f} BTC | ${price * amount:,.2f}")
        
        # Calculate bid/ask imbalance
        total_bid_value = sum(p * a for p, a in orderbook['bids'][:depth])
        total_ask_value = sum(p * a for p, a in orderbook['asks'][:depth])
        imbalance = total_bid_value / total_ask_value if total_ask_value > 0 else 0
        
        lines.append(f"\n## Summary:")
        lines.append(f"Spread: ${orderbook['asks'][0][0] - orderbook['bids'][0][0]:,.2f}")
        lines.append(f"Total Bid Value: ${total_bid_value:,.2f}")
        lines.append(f"Total Ask Value: ${total_ask_value:,.2f}")
        lines.append(f"Bid/Ask Imbalance: {imbalance:.4f} ({'Bullish' if imbalance > 1 else 'Bearish'})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise DataFetchError(f"Error fetching order book: {e}") from e


def get_fear_greed_index() -> str:
    """Fetch the Crypto Fear & Greed Index from Alternative.me (free, no key needed)."""
    import requests
    
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=30", timeout=10)
        data = resp.json()
        
        if "data" not in data:
            return "No Fear & Greed data available"
        
        lines = ["# Crypto Fear & Greed Index (last 30 days)\n"]
        for entry in data["data"]:
            date_str = datetime.fromtimestamp(int(entry["timestamp"])).strftime("%Y-%m-%d")
            value = entry["value"]
            classification = entry["value_classification"]
            lines.append(f"{date_str}: {value} ({classification})")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error fetching Fear & Greed Index: {e}"


def get_crypto_liquidations_summary(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
) -> str:
    """
    Get a summary of recent liquidation activity.
    Note: Detailed liquidation data may require CryptoQuant or Coinglass API.
    This provides what's available from exchange data.
    """
    # Placeholder — detailed liquidation data requires premium APIs
    return (
        f"# Liquidation Summary for {symbol}\n\n"
        "Note: Detailed liquidation data requires CryptoQuant or Coinglass API integration.\n"
        "For now, use funding rate extremes and open interest changes as proxy indicators:\n"
        "- Extreme positive funding (>0.05%) → overleveraged longs, liquidation risk\n"
        "- Extreme negative funding (<-0.05%) → overleveraged shorts, squeeze risk\n"
        "- Sudden OI drops → liquidation cascade occurred\n"
    )
