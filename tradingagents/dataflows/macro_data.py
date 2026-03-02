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
    Provide economic calendar context with current-month awareness.
    Uses date math to identify upcoming events.
    """
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    day = now.day
    weekday = now.weekday()  # 0=Mon
    month = now.strftime("%B %Y")
    
    lines = [f"# Economic Calendar Context — {month}\n"]
    
    # FOMC dates 2026 (approximate — 8 meetings)
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    if now.month in fomc_months:
        lines.append(f"⚠️ FOMC meeting month — rate decision expected this month")
    
    # CPI typically 10th-15th
    if 8 <= day <= 16:
        lines.append(f"⚠️ CPI release window (typically 10th-15th) — may have just released or upcoming")
    elif day < 8:
        lines.append(f"CPI release expected in ~{10-day} days")
    
    # NFP: first Friday
    if day <= 7 and weekday <= 4:
        days_to_friday = (4 - weekday) % 7
        if days_to_friday == 0 and day <= 7:
            lines.append(f"⚠️ Non-Farm Payrolls likely TODAY (first Friday of month)")
        elif days_to_friday > 0 and day + days_to_friday <= 7:
            lines.append(f"⚠️ NFP expected in {days_to_friday} days (first Friday)")
    
    # Options expiry: last Friday of month
    import calendar
    last_day = calendar.monthrange(now.year, now.month)[1]
    last_friday = last_day
    while datetime(now.year, now.month, last_friday).weekday() != 4:
        last_friday -= 1
    days_to_expiry = last_friday - day
    if 0 <= days_to_expiry <= 3:
        lines.append(f"⚠️ BTC Options/Futures expiry in {days_to_expiry} days — expect volatility")
    elif days_to_expiry < 0:
        lines.append(f"Options expiry passed this month")
    else:
        lines.append(f"Options expiry in ~{days_to_expiry} days ({now.month}/{last_friday})")
    
    # Weekly: Jobless claims Thursday
    if weekday <= 3:
        days_to_thurs = 3 - weekday
        lines.append(f"Initial Jobless Claims: {'TODAY' if days_to_thurs == 0 else f'in {days_to_thurs} days'} (Thursday)")
    
    lines.append("")
    lines.append("## Standing Calendar:")
    lines.append("- FOMC: ~8x/year (Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)")
    lines.append("- CPI: monthly, ~10th-15th")
    lines.append("- NFP: first Friday of month")
    lines.append("- PCE: last week of month")
    lines.append("- GDP: quarterly")
    lines.append("- Jobless Claims: weekly Thursday")
    lines.append("- BTC Options Expiry: last Friday of month")
    
    return "\n".join(lines)
