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
    get_crypto_orderbook,
    get_fear_greed_index,
    get_crypto_liquidations_summary,
)


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
