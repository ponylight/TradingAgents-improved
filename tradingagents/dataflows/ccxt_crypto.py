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
from .candle_cache import get_cached, save_cache


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
    """Fetch all OHLCV data between since and end, with file caching."""
    # Try cache first
    cached = get_cached(symbol, timeframe, since_ms, limit)
    if cached is not None:
        return cached

    all_candles = []
    current_since = since_ms
    
    while current_since < end_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        current_since = candles[-1][0] + 1
        if len(candles) < limit:
            break
    
    # Save to cache
    if all_candles:
        save_cache(symbol, timeframe, since_ms, limit, all_candles)
    
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
    
    df = wrap(data.copy())
    
    if False:  # No custom indicators
        pass
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
        lines.append("## Funding Rate Reference Scale")
        lines.append("  Normal:              -0.01% to +0.01%  (typical neutral market)")
        lines.append("  Elevated:            ±0.01% to ±0.03%  (directional bias, watch for reversal)")
        lines.append("  Extreme/Squeeze:     beyond ±0.03%     (squeeze potential — longs/shorts overextended)")
        lines.append("NOTE: A rate like -0.0042% is NORMAL, not 'significant negative'. Real squeezes require -0.03%+.\n")
        for fr in funding_rates[-20:]:  # Last 20 entries
            timestamp = datetime.fromtimestamp(fr['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
            rate = fr.get('fundingRate', 'N/A')
            if isinstance(rate, (int, float)):
                rate_pct = rate * 100
                if abs(rate_pct) > 0.03:
                    regime = "EXTREME"
                elif abs(rate_pct) > 0.01:
                    regime = "ELEVATED"
                else:
                    regime = "normal"
                rate = f"{rate:.6f} ({rate_pct:.4f}%) [{regime}]"
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


def get_crypto_oi_timeseries(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    exchange_id: str = "bybit",
    periods: int = 6,
) -> str:
    """Get OI time-series over recent intervals (default: last 6 × 4H candles).

    Returns OI at each timestamp plus:
    - Change rate (% per period)
    - Direction: BUILDING (OI rising), UNWINDING (OI falling), FLAT
    - Total change over the window
    """
    import urllib.request

    bybit_symbol = symbol.replace("/", "").replace(":USDT", "")
    try:
        # Fetch 5min-interval OI snapshots — 6 periods × 48 intervals/4h = 288 for 6 × 4h
        # But Bybit /open-interest endpoint has limited granularity. Use 1h interval, fetch enough for ~24h.
        limit = min(periods * 4, 48)  # 4 × 1h snapshots per 4H candle
        r = urllib.request.urlopen(
            f"https://api.bybit.com/v5/market/open-interest"
            f"?category=linear&symbol={bybit_symbol}&intervalTime=1h&limit={limit}",
            timeout=10,
        )
        data = json.loads(r.read())

        if data["retCode"] != 0 or not data["result"]["list"]:
            return f"No OI time-series data for {symbol}"

        oi_list = data["result"]["list"]  # newest first

        lines = [f"# Open Interest Time-Series for {symbol} ({exchange_id})\n"]
        lines.append(f"## Last {len(oi_list)} hourly snapshots (newest first)\n")

        # Parse all OI values with timestamps
        oi_points = []
        for entry in oi_list:
            ts_ms = int(entry["timestamp"])
            oi_val = float(entry["openInterest"])
            ts_str = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M")
            oi_points.append((ts_str, oi_val, ts_ms))

        # Display each point with period-over-period change
        for i, (ts_str, oi_val, _) in enumerate(oi_points):
            if i < len(oi_points) - 1:
                prev_oi = oi_points[i + 1][1]
                change_pct = ((oi_val - prev_oi) / prev_oi * 100) if prev_oi else 0
                lines.append(f"  {ts_str}: {oi_val:,.2f} BTC ({change_pct:+.2f}%)")
            else:
                lines.append(f"  {ts_str}: {oi_val:,.2f} BTC (baseline)")

        # Compute summary statistics
        newest_oi = oi_points[0][1]
        oldest_oi = oi_points[-1][1]
        total_change_pct = ((newest_oi - oldest_oi) / oldest_oi * 100) if oldest_oi else 0

        # Determine direction
        if total_change_pct > 2.0:
            direction = "BUILDING"
            interpretation = "OI rising — new positions entering, conviction building"
        elif total_change_pct < -2.0:
            direction = "UNWINDING"
            interpretation = "OI falling — positions closing, conviction fading"
        else:
            direction = "FLAT"
            interpretation = "OI stable — no significant positioning shift"

        # Calculate change rate per 4H period
        hours_span = (oi_points[0][2] - oi_points[-1][2]) / 1000 / 3600
        periods_4h = max(hours_span / 4, 1)
        rate_per_4h = total_change_pct / periods_4h

        lines.append(f"\n## Summary")
        lines.append(f"Current OI: {newest_oi:,.2f} BTC")
        lines.append(f"Window OI: {oldest_oi:,.2f} → {newest_oi:,.2f} BTC")
        lines.append(f"Total change: {total_change_pct:+.2f}%")
        lines.append(f"Rate: {rate_per_4h:+.2f}% per 4H candle")
        lines.append(f"Direction: {direction}")
        lines.append(f"Interpretation: {interpretation}")

        # Flag significant moves
        if abs(total_change_pct) > 5:
            lines.append(f"\n⚠️ SIGNIFICANT OI {'BUILD-UP' if total_change_pct > 0 else 'FLUSH'}: {total_change_pct:+.1f}% — watch for squeeze")
        elif abs(total_change_pct) > 3:
            lines.append(f"\n📊 Notable OI shift: {total_change_pct:+.1f}%")

        lines.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    except Exception as e:
        raise DataFetchError(f"Error fetching OI time-series: {e}") from e


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
    exchange_id: str = "bybit",
) -> str:
    """
    Get derivatives positioning summary: funding, OI, long/short ratio.
    Uses Bybit API directly — no third-party key needed.
    """
    import urllib.request

    lines = [f"# Derivatives Positioning for {symbol}\n"]

    try:
        exchange = _get_exchange(exchange_id)

        # 1. Current funding rate
        fr = exchange.fetch_funding_rate(symbol)
        rate = fr.get("fundingRate", 0)
        lines.append(f"## Funding Rate")
        lines.append(f"Current: {rate:.6f} ({rate*100:.4f}%)")
        if abs(rate) > 0.0005:
            lines.append(f"⚠️ EXTREME funding — {'longs' if rate > 0 else 'shorts'} paying heavy")
        lines.append("")

        # 2. Open interest
        bybit_symbol = symbol.replace("/", "").replace(":USDT", "")
        try:
            r = urllib.request.urlopen(
                f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={bybit_symbol}&intervalTime=5min&limit=6",
                timeout=5,
            )
            oi_data = __import__("json").loads(r.read())
            if oi_data["retCode"] == 0 and oi_data["result"]["list"]:
                oi_list = oi_data["result"]["list"]
                latest_oi = float(oi_list[0]["openInterest"])
                oldest_oi = float(oi_list[-1]["openInterest"]) if len(oi_list) > 1 else latest_oi
                oi_delta = ((latest_oi - oldest_oi) / oldest_oi * 100) if oldest_oi else 0
                lines.append(f"## Open Interest")
                lines.append(f"Current: {latest_oi:,.2f} BTC")
                lines.append(f"30m change: {oi_delta:+.2f}%")
                if abs(oi_delta) > 3:
                    lines.append(f"⚠️ Significant OI {'build-up' if oi_delta > 0 else 'flush'}")
                lines.append("")
        except Exception as e:
            lines.append(f"## Open Interest\nUnavailable: {e}\n")

        # 3. Long/short account ratio
        try:
            r = urllib.request.urlopen(
                f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={bybit_symbol}&period=4h&limit=6",
                timeout=5,
            )
            ls_data = __import__("json").loads(r.read())
            if ls_data["retCode"] == 0 and ls_data["result"]["list"]:
                ls_list = ls_data["result"]["list"]
                latest = ls_list[0]
                lines.append(f"## Long/Short Account Ratio (4h)")
                lines.append(f"Longs: {float(latest['buyRatio'])*100:.1f}% | Shorts: {float(latest['sellRatio'])*100:.1f}%")
                # Trend over last 6 periods
                if len(ls_list) > 1:
                    prev = ls_list[-1]
                    delta = float(latest["buyRatio"]) - float(prev["buyRatio"])
                    lines.append(f"Trend: {'more longs' if delta > 0 else 'more shorts'} ({delta*100:+.1f}pp over {len(ls_list)} periods)")
                lines.append("")
        except Exception as e:
            lines.append(f"## Long/Short Ratio\nUnavailable: {e}\n")

        # 4. Interpretation
        lines.append("## Positioning Interpretation")
        if rate > 0.0003:
            lines.append("- Longs paying shorts — crowded long, liquidation risk to downside")
        elif rate < -0.0003:
            lines.append("- Shorts paying longs — crowded short, squeeze risk to upside")
        else:
            lines.append("- Funding neutral — no strong directional crowding")

        return "\n".join(lines)

    except Exception as e:
        return f"Derivatives positioning unavailable: {e}"
