"""
Macro economic data provider using FRED API and yfinance.
For the Macro Analyst agent.
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Annotated
from .exceptions import DataFetchError


def get_dxy_data(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get US Dollar Index (DXY) data via yfinance."""
    try:
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if dxy.empty:
            return f"No DXY data found for {start_date} to {end_date}"
        
        dxy = dxy.reset_index()
        csv_string = dxy.to_csv(index=False)
        header = f"# US Dollar Index (DXY) from {start_date} to {end_date}\n"
        header += f"# DXY rising = USD strengthening = typically bearish for BTC\n\n"
        return header + csv_string
    except Exception as e:
        raise DataFetchError(f"Error fetching DXY: {e}") from e


def get_treasury_yields(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get US Treasury yields (10Y and 2Y) via yfinance."""
    try:
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        twy = yf.download("^IRX", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        
        lines = [f"# US Treasury Yields from {start_date} to {end_date}\n"]
        lines.append("# Rising yields = tighter monetary conditions = typically bearish for risk assets\n")
        
        if not tnx.empty:
            lines.append("## 10-Year Treasury Yield:")
            tnx = tnx.reset_index()
            lines.append(tnx[["Date", "Close"]].tail(20).to_string(index=False))
        
        if not twy.empty:
            lines.append("\n## 13-Week Treasury Bill Rate:")
            twy = twy.reset_index()
            lines.append(twy[["Date", "Close"]].tail(20).to_string(index=False))
        
        return "\n".join(lines)
    except Exception as e:
        raise DataFetchError(f"Error fetching treasury yields: {e}") from e


def get_sp500_data(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get S&P 500 data as risk sentiment proxy."""
    try:
        spx = yf.download("^GSPC", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if spx.empty:
            return f"No S&P 500 data found for {start_date} to {end_date}"
        
        spx = spx.reset_index()
        csv_string = spx.to_csv(index=False)
        header = f"# S&P 500 from {start_date} to {end_date}\n"
        header += "# BTC often correlates with S&P 500 in risk-on/risk-off environments\n\n"
        return header + csv_string
    except Exception as e:
        raise DataFetchError(f"Error fetching S&P 500: {e}") from e


def get_fred_data(
    series_id: Annotated[str, "FRED series ID e.g. FEDFUNDS, CPIAUCSL, M2SL, UNRATE"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """
    Get economic data from FRED (Federal Reserve Economic Data).
    Common series:
    - FEDFUNDS: Federal Funds Rate
    - CPIAUCSL: Consumer Price Index (CPI)
    - M2SL: M2 Money Supply
    - UNRATE: Unemployment Rate
    - T10Y2Y: 10Y-2Y Treasury Spread (yield curve)
    - DTWEXBGS: Trade Weighted US Dollar Index
    """
    try:
        from fredapi import Fred
        
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return "FRED_API_KEY not set. Please provide your FRED API key."
        
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        
        if data.empty:
            return f"No FRED data found for {series_id}"
        
        series_info = {
            "FEDFUNDS": "Federal Funds Effective Rate — Fed's target rate, higher = tighter monetary policy",
            "CPIAUCSL": "Consumer Price Index — inflation measure, rising CPI = hawkish Fed",
            "M2SL": "M2 Money Supply — liquidity proxy, expanding M2 historically bullish for BTC",
            "UNRATE": "Unemployment Rate — labor market health",
            "T10Y2Y": "10Y-2Y Spread — yield curve, negative = recession signal",
            "DTWEXBGS": "Trade Weighted USD — broad dollar strength index",
        }
        
        header = f"# FRED Data: {series_id}\n"
        header += f"# {series_info.get(series_id, 'Economic data series')}\n"
        header += f"# Period: {start_date} to {end_date}\n\n"
        
        df = data.reset_index()
        df.columns = ["Date", "Value"]
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        
        return header + df.to_csv(index=False)
        
    except ImportError:
        return "fredapi not installed. Run: pip install fredapi"
    except Exception as e:
        raise DataFetchError(f"Error fetching FRED data for {series_id}: {e}") from e


def get_economic_calendar_summary() -> str:
    """
    Provide a summary of key upcoming economic events that impact BTC.
    Note: For a full calendar, consider integrating with an economic calendar API.
    """
    return (
        "# Key Economic Events to Watch for BTC Impact\n\n"
        "## High Impact Events:\n"
        "- FOMC Rate Decision & Statement (8x/year)\n"
        "- CPI Report (monthly, ~10th-15th)\n"
        "- Non-Farm Payrolls (monthly, first Friday)\n"
        "- PCE Price Index (monthly, ~last week)\n"
        "- GDP Report (quarterly)\n"
        "- Fed Chair Press Conferences\n\n"
        "## Medium Impact Events:\n"
        "- PPI Report (monthly)\n"
        "- Retail Sales (monthly)\n"
        "- ISM Manufacturing/Services PMI (monthly)\n"
        "- Initial Jobless Claims (weekly, Thursday)\n\n"
        "## Crypto-Specific Events:\n"
        "- BTC Options Expiry (monthly/quarterly, last Friday)\n"
        "- Futures Expiry (quarterly)\n"
        "- Major Protocol Upgrades\n"
        "- Regulatory Announcements (SEC, CFTC)\n"
        "- Exchange Reserve Changes\n\n"
        "Note: Check an economic calendar (e.g., forexfactory.com) for exact dates and times.\n"
    )
