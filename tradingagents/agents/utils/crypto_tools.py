"""
Crypto-specific tools for the trading agents framework.
These replace/supplement the stock-focused tools for BTC/USDT trading.
"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.ccxt_crypto import (
    get_crypto_ohlcv,
    get_crypto_indicators,
    get_crypto_funding_rate,
    get_crypto_open_interest,
    get_crypto_oi_timeseries,
    get_crypto_orderbook,
    get_fear_greed_index,
    get_crypto_liquidations_summary,
)
from tradingagents.dataflows.macd_divergence import check_macd_divergence_for_symbol
from tradingagents.dataflows.pattern_scanner import scan_all_patterns


@tool
def get_crypto_price_data(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    timeframe: Annotated[str, "candle timeframe: 1h, 4h, 1d"] = "1d",
) -> str:
    """
    Retrieve crypto OHLCV price data from Bybit exchange.
    Args:
        symbol: Trading pair e.g. BTC/USDT
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        timeframe: Candle timeframe (1h, 4h, 1d)
    Returns:
        Formatted OHLCV data with volume.
    """
    return get_crypto_ohlcv(symbol, start_date, end_date, timeframe=timeframe)


@tool
def get_crypto_technical_indicators(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    indicator: Annotated[str, "indicator name: rsi, macd, macds, macdh, boll, boll_ub, boll_lb, atr, close_50_sma, close_200_sma, close_10_ema, vwma, mfi"],
    curr_date: Annotated[str, "current date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Calculate technical indicators for crypto assets.
    Supports: rsi, macd, macds, macdh, boll, boll_ub, boll_lb, atr,
    close_50_sma, close_200_sma, close_10_ema, vwma, mfi.
    """
    return get_crypto_indicators(symbol, indicator, curr_date, look_back_days)


@tool
def get_funding_rate(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT:USDT"],
) -> str:
    """
    Get current and recent funding rates for a perpetual contract.
    High positive rates indicate overleveraged longs; negative rates indicate overleveraged shorts.
    """
    return get_crypto_funding_rate(symbol)


@tool
def get_open_interest(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT:USDT"],
) -> str:
    """
    Get current open interest for a perpetual contract.
    Rising OI + rising price = strong trend. Rising OI + falling price = bearish pressure.
    """
    return get_crypto_open_interest(symbol)


@tool
def get_oi_timeseries(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT:USDT"],
) -> str:
    """
    Get open interest TIME-SERIES over the last ~24 hours (hourly snapshots).
    Shows OI at each timestamp, period-over-period change, total change rate,
    and direction (BUILDING/UNWINDING/FLAT).
    PREFERRED over get_open_interest — gives trend context, not just a snapshot.
    """
    return get_crypto_oi_timeseries(symbol)


@tool
def get_orderbook_depth(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
    depth: Annotated[int, "number of price levels to fetch"] = 20,
) -> str:
    """
    Get order book depth showing bid/ask levels and imbalance.
    Useful for identifying support/resistance levels and market sentiment.
    """
    return get_crypto_orderbook(symbol, depth=depth)


@tool
def get_crypto_fear_greed() -> str:
    """
    Get the Crypto Fear & Greed Index (last 30 days).
    A lagged composite of volatility, momentum, social volume, and surveys.
    0-25: Extreme Fear, 25-50: Fear, 50-75: Greed, 75-100: Extreme Greed.
    NOTE: This is a lagging indicator — weight it LOW relative to funding rates and OI.
    """
    return get_fear_greed_index()


@tool
def get_liquidation_info(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"],
) -> str:
    """
    Get liquidation activity summary and risk indicators.
    """
    return get_crypto_liquidations_summary(symbol)


@tool
def get_macro_signal_radar() -> str:
    """
    Get 7-signal macro radar with composite BUY/CASH verdict.
    Signals: Fear & Greed, BTC Technical Trend (SMA50/200/VWAP/Mayer),
    JPY Liquidity, QQQ vs XLP Macro Regime, BTC-QQQ Flow Structure,
    Hash Rate momentum, Mining Cost floor.
    ≥57% bullish = BUY, otherwise CASH.
    IMPORTANT: This is a high-signal macro overlay. Weight it heavily in position decisions.
    """
    from tradingagents.dataflows.macro_radar import get_macro_radar_cached
    result = get_macro_radar_cached()
    return result["summary"]


@tool
def get_stablecoin_peg_health() -> str:
    """
    Monitor stablecoin peg health (USDT, USDC, DAI).
    Depeg events signal systemic risk — if any stablecoin shows >0.5% deviation,
    treat it as a WARNING for risk management.
    """
    from tradingagents.dataflows.macro_radar import get_stablecoin_health_cached
    result = get_stablecoin_health_cached()
    return f"Stablecoin Health: {result['overall']}\n{result['summary']}"


@tool
def check_macd_divergence(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"] = "BTC/USDT",
    timeframe: Annotated[str, "candle timeframe: 4h or 1d"] = "1d",
) -> str:
    """
    Check for MACD Triple Divergence (半木夏 strategy).
    Detects when price makes 3 successive peaks/troughs while MACD histogram
    shrinks twice consecutively — a high-probability reversal signal.
    Best on Daily and 4H timeframes. Occurs ~1-2 times per year on daily.
    Returns BEARISH_DIVERGENCE, BULLISH_DIVERGENCE, or NONE with confidence score.
    """
    return check_macd_divergence_for_symbol(symbol, timeframe)


@tool
def run_pattern_scan(
    symbol: Annotated[str, "trading pair e.g. BTC/USDT"] = "BTC/USDT",
) -> str:
    """
    Run all pattern detectors against current market data.
    Checks 10+ verified trader strategies (Kyle Williams, Minervini, Hong Inki,
    BNF, 半木夏, 比特皇, Sykes, Bonde, Qullamaggie) and returns structured signals
    with confidence scores. Call this FIRST before making any trade decision.
    """
    return scan_all_patterns(symbol)


def get_crisis_impact_index() -> str:
    """
    Get the CryptoMonitor Crisis Impact Index (CII) — geopolitical risk score 0-100.
    Sources: GDELT global events, BBC/Al Jazeera headlines, mining region disruption tracking.
    Levels: LOW (<20), MODERATE (20-39), ELEVATED (40-59), HIGH (60-79), SEVERE (80+).
    Use this to assess geopolitical risk impact on crypto markets.
    """
    from tradingagents.dataflows.crypto_monitor import get_crisis_impact_index as _get_cii
    result = _get_cii()
    lines = [
        f"Crisis Impact Index: {result['cii_score']}/100 ({result['level']})",
        f"Crypto Impact: {result['crypto_impact']}",
        f"Components: GDELT={result['components']['gdelt_score']:.0f} Headlines={result['components']['headline_score']:.0f} Mining={result['components']['mining_region_score']:.0f}",
    ]
    if result.get("top_events"):
        lines.append("Top Events:")
        for e in result["top_events"]:
            lines.append(f"  - {e}")
    return "\n".join(lines)
