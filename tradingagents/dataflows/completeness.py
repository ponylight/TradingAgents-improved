"""Validates market data completeness before analysis."""

import logging
from datetime import datetime, timedelta

log = logging.getLogger("completeness")


def check_ohlcv_completeness(candles: list, timeframe: str = "4h", symbol: str = "BTC/USDT") -> dict:
    """Check OHLCV data for gaps, staleness, and anomalies.
    
    Returns dict with: ok (bool), issues (list of str), stats (dict).
    """
    issues = []
    
    if not candles:
        return {"ok": False, "issues": ["No candle data returned"], "stats": {}}
    
    # Check staleness
    last_ts = candles[-1][0] / 1000  # ccxt returns ms
    age_hours = (datetime.utcnow().timestamp() - last_ts) / 3600
    tf_hours = {"1h": 1, "4h": 4, "1d": 24}.get(timeframe, 4)
    
    if age_hours > tf_hours * 2:
        issues.append(f"Stale data: last candle is {age_hours:.1f}h old (expected <{tf_hours*2}h)")
    
    # Check for gaps
    expected_interval_ms = tf_hours * 3600 * 1000
    gaps = 0
    for i in range(1, len(candles)):
        actual = candles[i][0] - candles[i-1][0]
        if actual > expected_interval_ms * 1.5:
            gaps += 1
    
    if gaps > 0:
        issues.append(f"{gaps} gaps detected in {len(candles)} candles")
    
    # Check for zero volume candles (exchange issues)
    zero_vol = sum(1 for c in candles[-20:] if c[5] == 0)
    if zero_vol > 3:
        issues.append(f"{zero_vol}/20 recent candles have zero volume")
    
    # Check for identical OHLC (frozen feed)
    frozen = 0
    for c in candles[-10:]:
        if c[1] == c[2] == c[3] == c[4]:
            frozen += 1
    if frozen > 2:
        issues.append(f"{frozen}/10 recent candles have identical OHLC (frozen feed?)")
    
    stats = {
        "candle_count": len(candles),
        "age_hours": round(age_hours, 1),
        "gaps": gaps,
        "zero_volume_recent": zero_vol,
    }
    
    if issues:
        for issue in issues:
            log.warning(f"⚠️ [{symbol}] {issue}")
    
    return {"ok": len(issues) == 0, "issues": issues, "stats": stats}
