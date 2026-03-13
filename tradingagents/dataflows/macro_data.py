"""
Macro economic data provider using FRED API and yfinance.
For the Macro Analyst agent.

Each function tries a primary source and falls back to alternatives.
"""

import os
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Annotated
from .exceptions import DataFetchError

log = logging.getLogger("macro_data")


def get_dxy_data(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get US Dollar Index (DXY) data. Primary: yfinance DX-Y.NYB, fallback: FRED DTWEXBGS."""
    header = f"# US Dollar Index (DXY) from {start_date} to {end_date}\n"
    header += f"# DXY rising = USD strengthening = typically bearish for BTC\n\n"

    # Primary: yfinance
    try:
        dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if not dxy.empty:
            log.info("DXY: loaded from yfinance DX-Y.NYB")
            dxy = dxy.reset_index()
            return header + f"# Source: yfinance DX-Y.NYB\n" + dxy.to_csv(index=False)
        log.warning("DXY: yfinance returned empty — trying FRED fallback")
    except Exception as e:
        log.warning(f"DXY: yfinance failed ({e}) — trying FRED fallback")

    # Fallback: FRED DTWEXBGS (Trade Weighted USD)
    try:
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            data = fred.get_series("DTWEXBGS", observation_start=start_date, observation_end=end_date)
            if not data.empty:
                log.info("DXY: loaded from FRED DTWEXBGS (trade-weighted USD)")
                df = data.reset_index()
                df.columns = ["Date", "Close"]
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                return header + f"# Source: FRED DTWEXBGS (trade-weighted USD, proxy)\n" + df.to_csv(index=False)
    except Exception as e2:
        log.warning(f"DXY: FRED fallback also failed: {e2}")

    return f"DXY data unavailable for {start_date} to {end_date}. Both yfinance and FRED sources failed. Use other macro signals."


def get_treasury_yields(
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Get US Treasury yields. Primary: yfinance ^TNX/^IRX, fallback: FRED DGS10/DGS2."""
    lines = [f"# US Treasury Yields from {start_date} to {end_date}\n"]
    lines.append("# Rising yields = tighter monetary conditions = typically bearish for risk assets\n")

    got_10y = False
    got_short = False

    # Primary: yfinance
    try:
        tnx = yf.download("^TNX", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if not tnx.empty:
            lines.append("## 10-Year Treasury Yield (yfinance ^TNX):")
            tnx = tnx.reset_index()
            lines.append(tnx[["Date", "Close"]].tail(20).to_string(index=False))
            got_10y = True
            log.info("Treasury 10Y: loaded from yfinance ^TNX")
    except Exception as e:
        log.warning(f"Treasury 10Y yfinance failed: {e}")

    try:
        twy = yf.download("^IRX", start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if not twy.empty:
            lines.append("\n## 13-Week Treasury Bill Rate (yfinance ^IRX):")
            twy = twy.reset_index()
            lines.append(twy[["Date", "Close"]].tail(20).to_string(index=False))
            got_short = True
            log.info("Treasury 13W: loaded from yfinance ^IRX")
    except Exception as e:
        log.warning(f"Treasury 13W yfinance failed: {e}")

    # Fallback: FRED for anything that failed
    if not got_10y or not got_short:
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=api_key)
                if not got_10y:
                    data = fred.get_series("DGS10", observation_start=start_date, observation_end=end_date)
                    if not data.empty:
                        df = data.reset_index()
                        df.columns = ["Date", "Yield"]
                        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                        lines.append("\n## 10-Year Treasury Yield (FRED DGS10):")
                        lines.append(df.tail(20).to_string(index=False))
                        got_10y = True
                        log.info("Treasury 10Y: loaded from FRED DGS10")
                if not got_short:
                    data = fred.get_series("DGS2", observation_start=start_date, observation_end=end_date)
                    if not data.empty:
                        df = data.reset_index()
                        df.columns = ["Date", "Yield"]
                        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                        lines.append("\n## 2-Year Treasury Yield (FRED DGS2):")
                        lines.append(df.tail(20).to_string(index=False))
                        got_short = True
                        log.info("Treasury 2Y: loaded from FRED DGS2")
            except Exception as e2:
                log.warning(f"Treasury FRED fallback failed: {e2}")

    if not got_10y and not got_short:
        return f"Treasury yields unavailable for {start_date} to {end_date}. Both yfinance and FRED sources failed. Use other macro signals."

    return "\n".join(lines)


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
        log.warning(f"S&P 500 fetch failed: {e}")
        return f"S&P 500 data unavailable for {start_date} to {end_date}: {e}"


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
            return f"FRED API key not configured — economic data for {series_id} unavailable. Set FRED_API_KEY env var."
        
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
        log.warning(f"FRED data fetch failed for {series_id}: {e}")
        return f"FRED data for {series_id} unavailable: {e}. Using fallback sources."


def get_economic_calendar_summary() -> str:
    """
    Provide economic calendar context with current-month awareness.
    Uses date math to identify upcoming events.
    Hardcoded FOMC/CPI/NFP dates as fallback — no external API needed.
    """
    import calendar as _cal

    now = datetime.utcnow()
    day = now.day
    weekday = now.weekday()  # 0=Mon
    month = now.strftime("%B %Y")

    lines = [f"# Economic Calendar Context — {month}\n"]

    # Known FOMC meeting dates (Wednesday announcements)
    # This is the reliable fallback — no external API needed.
    # TODO: Add 2028 FOMC dates when the Fed publishes them (typically late 2027).
    fomc_dates_2027 = [
        (1, 29), (3, 19), (5, 7), (6, 18),
        (7, 30), (9, 17), (11, 4), (12, 16),
    ]
    fomc_dates_2026 = [
        (1, 29), (3, 18), (4, 29), (6, 17),
        (7, 29), (9, 16), (10, 28), (12, 9),
    ]
    fomc_dates_2025 = [
        (1, 29), (3, 19), (5, 7), (6, 18),
        (7, 30), (9, 17), (12, 17),
    ]
    fomc_by_year = {2025: fomc_dates_2025, 2026: fomc_dates_2026, 2027: fomc_dates_2027}
    fomc_dates = fomc_by_year.get(now.year, fomc_dates_2027)

    next_fomc = None
    for m, d in fomc_dates:
        try:
            fdate = datetime(now.year, m, d)
            delta = (fdate - now).days
            if delta >= -1:  # Include yesterday (just happened)
                if delta <= 0:
                    lines.append(f"⚠️ FOMC rate decision TODAY/YESTERDAY ({m}/{d})")
                elif delta <= 3:
                    lines.append(f"FOMC rate decision in {delta} days ({m}/{d}) — see FedWatch probabilities below")
                elif delta <= 14:
                    lines.append(f"FOMC in {delta} days ({m}/{d})")
                next_fomc = (m, d, delta)
                break
        except ValueError:
            continue

    # CPI typically 10th-15th
    if 8 <= day <= 16:
        lines.append(f"⚠️ CPI release window (typically 10th-15th) — may have just released or upcoming")
    elif day < 8:
        lines.append(f"CPI release expected in ~{10 - day} days")

    # NFP: first Friday
    if day <= 7 and weekday <= 4:
        days_to_friday = (4 - weekday) % 7
        if days_to_friday == 0 and day <= 7:
            lines.append(f"⚠️ Non-Farm Payrolls likely TODAY (first Friday of month)")
        elif days_to_friday > 0 and day + days_to_friday <= 7:
            lines.append(f"⚠️ NFP expected in {days_to_friday} days (first Friday)")

    # Options expiry: last Friday of month
    last_day = _cal.monthrange(now.year, now.month)[1]
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

    # Fed Funds Futures implied probabilities
    fedwatch = get_fedwatch_probabilities()
    if fedwatch:
        lines.append("")
        lines.append(fedwatch)

    return "\n".join(lines)



def get_fedwatch_probabilities() -> str | None:
    """Derive rate cut/hike probabilities from 30-day Fed Funds futures (Yahoo Finance).

    Uses CME FedWatch methodology with day-weighted post-meeting rate isolation:
    - For early/mid-month meetings: isolate post-rate from month average
    - For late-month meetings (<=7 post-days): use next month's contract directly
    - Chain meetings: each meeting's post-rate becomes the next meeting's pre-rate
    - Cumulative ease = (effective_rate - post_rate) / 0.25

    Reference: https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html
    """
    import requests
    import logging
    import calendar as _cal
    log = logging.getLogger(__name__)

    try:
        STEP = 0.25

        # Fetch live rates from FRED
        CURRENT_EFF = None
        CURRENT_UPPER = 3.75
        CURRENT_LOWER = 3.50
        fred_key = os.environ.get("FRED_API_KEY", "")
        if fred_key:
            for series, attr in [("DFF", "eff"), ("DFEDTARU", "upper"), ("DFEDTARL", "lower")]:
                try:
                    r = requests.get(
                        f"https://api.stlouisfed.org/fred/series/observations"
                        f"?series_id={series}&api_key={fred_key}&file_type=json"
                        f"&sort_order=desc&limit=1",
                        timeout=8,
                    )
                    if r.status_code == 200:
                        val = r.json().get("observations", [{}])[0].get("value")
                        if val and val != ".":
                            if attr == "eff":
                                CURRENT_EFF = float(val)
                            elif attr == "upper":
                                CURRENT_UPPER = float(val)
                            elif attr == "lower":
                                CURRENT_LOWER = float(val)
                except Exception:
                    pass
        if CURRENT_EFF is None:
            CURRENT_EFF = (CURRENT_UPPER + CURRENT_LOWER) / 2

        now = datetime.utcnow()

        # FOMC meeting dates
        fomc_dates_2026 = [
            (1, 29), (3, 18), (4, 29), (6, 17),
            (7, 29), (9, 16), (10, 28), (12, 9),
        ]
        fomc_dates_2027 = [
            (1, 29), (3, 19), (5, 7), (6, 18),
            (7, 30), (9, 17), (11, 4), (12, 16),
        ]
        fomc_by_year = {2026: fomc_dates_2026, 2027: fomc_dates_2027}

        month_codes = {
            1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
            7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
        }
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # Find upcoming FOMC meetings (next 6)
        upcoming = []
        for year in [now.year, now.year + 1]:
            for m, d in fomc_by_year.get(year, []):
                if datetime(year, m, d) > now and len(upcoming) < 6:
                    upcoming.append((year, m, d))

        if not upcoming:
            return None

        # Fetch ALL month futures (need non-meeting months for chaining)
        futures = {}
        year = now.year
        for m_num in range(now.month, 13):
            code = month_codes.get(m_num)
            if not code:
                continue
            yr_short = str(year)[-2:]
            ticker = f"ZQ{code}{yr_short}.CBT"
            try:
                r = requests.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                )
                if r.status_code == 200:
                    data = r.json()
                    result = data.get("chart", {}).get("result", [])
                    if result:
                        price = result[0].get("meta", {}).get("regularMarketPrice")
                        if price:
                            futures[(year, m_num)] = 100 - price
            except Exception:
                continue
        # Also fetch Jan-Jun of next year if needed
        if any(y > year for y, _, _ in upcoming):
            for m_num in range(1, 7):
                code = month_codes.get(m_num)
                yr_short = str(year + 1)[-2:]
                ticker = f"ZQ{code}{yr_short}.CBT"
                try:
                    r = requests.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        result = data.get("chart", {}).get("result", [])
                        if result:
                            price = result[0].get("meta", {}).get("regularMarketPrice")
                            if price:
                                futures[(year + 1, m_num)] = 100 - price
                except Exception:
                    continue

        if not futures:
            return None

        output = ["## Fed Funds Futures — Rate Probabilities (FedWatch-style)"]
        output.append(f"Current target: {CURRENT_LOWER:.2f}%-{CURRENT_UPPER:.2f}% | Effective: {CURRENT_EFF:.2f}%")
        output.append("")

        prev_post_rate = CURRENT_EFF

        for year, m, d in upcoming:
            key = (year, m)
            if key not in futures:
                continue

            N = _cal.monthrange(year, m)[1]
            post_days = N - d + 1
            avg = futures[key]
            pre_rate = prev_post_rate

            # Post-meeting rate calculation
            next_key = (year, m + 1) if m < 12 else (year + 1, 1)
            if post_days <= 7 and next_key in futures:
                # Late-month meeting: next month\'s implied IS the post-meeting rate
                post_rate = futures[next_key]
            else:
                # Day-weighted isolation from this month\'s contract
                # Find the prior non-meeting month\'s implied for pre_rate
                prior_key = (year, m - 1) if m > 1 else (year - 1, 12)
                pre_for_calc = futures.get(prior_key, CURRENT_EFF)
                post_rate = (avg * N - pre_for_calc * (d - 1)) / post_days

            # Meeting-specific ease probability
            ease = max(0, min(1, (pre_rate - post_rate) / STEP))

            # Cumulative ease from effective rate
            cum_ease = max(0, min(1, (CURRENT_EFF - post_rate) / STEP))
            cum_ease_pct = cum_ease * 100
            no_change_pct = max(0, 100 - cum_ease_pct)

            label = f"{month_names[m]} {d}, {year}"
            if cum_ease_pct > 50:
                emoji = "\U0001f7e2"
            elif cum_ease_pct > 20:
                emoji = "\U0001f7e1"
            else:
                emoji = "\U0001f534"

            output.append(
                f"  {emoji} {label}: Ease {cum_ease_pct:.1f}% / No Change {no_change_pct:.1f}% | "
                f"Implied post-mtg: {post_rate:.3f}% | Cuts priced: {cum_ease:.2f}"
            )

            prev_post_rate = post_rate

        if len(output) <= 2:
            return None

        return "\n".join(output)

    except Exception as e:
        log.debug(f"FedWatch probabilities failed: {e}")
        return None
