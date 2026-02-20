"""
Macro economy tools for the Macro Analyst agent.
"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.macro_data import (
    get_dxy_data,
    get_treasury_yields,
    get_sp500_data,
    get_fred_data,
    get_economic_calendar_summary,
)


@tool
def get_dollar_index(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """
    Get US Dollar Index (DXY) data. DXY rising = USD strengthening = typically bearish for BTC.
    """
    return get_dxy_data(start_date, end_date)


@tool
def get_yields(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """
    Get US Treasury yields (10Y and short-term). Rising yields = tighter conditions = risk-off.
    """
    return get_treasury_yields(start_date, end_date)


@tool
def get_sp500(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """
    Get S&P 500 data as a risk sentiment proxy. BTC often correlates with equities in macro regimes.
    """
    return get_sp500_data(start_date, end_date)


@tool
def get_economic_data(
    series_id: Annotated[str, "FRED series: FEDFUNDS, CPIAUCSL, M2SL, UNRATE, T10Y2Y"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """
    Get economic data from FRED. Key series:
    - FEDFUNDS: Fed Funds Rate (monetary policy)
    - CPIAUCSL: CPI (inflation)
    - M2SL: M2 Money Supply (liquidity, bullish for BTC when expanding)
    - UNRATE: Unemployment Rate
    - T10Y2Y: Yield curve spread (negative = recession signal)
    """
    return get_fred_data(series_id, start_date, end_date)


@tool
def get_economic_calendar() -> str:
    """
    Get summary of key upcoming economic events that impact BTC price.
    Includes FOMC, CPI, NFP, options expiry, and crypto-specific events.
    """
    return get_economic_calendar_summary()
