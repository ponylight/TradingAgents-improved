"""
Green Lane Scanner — Phase 1
Pure deterministic scanner. No LLM calls.

Detects the "Green Lane" setup:
  1. EMA confluence zone (daily EMA9/21 + nearest AVWAP)
  2. Liquidity sweep into the zone (wick, not close)
  3. V-reversal with strong volume
  4. Entry above reversal candle high, stop below sweep low, 3:1/5:1 TPs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

from .ta_schema import GreenLaneSignal
from .crypto_technical_brief import (
    _ema,
    _calc_avwap,
    _find_swing_points,
    compute_indicators,
    compute_avwap_levels,
    detect_ema_alignment,
)

log = logging.getLogger("green_lane_scanner")

# ── BTC event anchors
BTC_EVENT_ANCHORS = {
    "halving_2024": "2024-04-19",
    "etf_approval": "2024-01-10",
    "cycle_ath": "2024-03-14",
    "cycle_low": "2022-11-21",
}


def _fetch_5m(symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
    try:
        ex = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
        raw = ex.fetch_ohlcv(f"{symbol}:USDT", timeframe="5m", limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ema_9"] = _ema(df["close"], 9)
        df["ema_21"] = _ema(df["close"], 21)
        df["ema_50"] = _ema(df["close"], 50)
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        return df
    except Exception as e:
        log.error(f"fetch_5m failed for {symbol}: {e}")
        return None


def _detect_avwap_pinch(df: pd.DataFrame) -> Tuple[bool, float]:
    sh_idx, sl_idx = _find_swing_points(df, lookback=60)
    avwap_sh = _calc_avwap(df, sh_idx) if sh_idx is not None else None
    avwap_sl = _calc_avwap(df, sl_idx) if sl_idx is not None else None
    if avwap_sh is None or avwap_sl is None:
        return False, 0.0
    mid = (avwap_sh + avwap_sl) / 2
    if mid == 0:
        return False, 0.0
    width_pct = abs(avwap_sh - avwap_sl) / mid * 100
    return width_pct <= 3.0, round(width_pct, 2)


def _compute_confluence_zone(df_daily: pd.DataFrame) -> Tuple[float, float, float]:
    e9 = float(df_daily["ema_9"].iloc[-1])
    e21 = float(df_daily["ema_21"].iloc[-1])
    price = float(df_daily["close"].iloc[-1])
    avwap_levels = compute_avwap_levels(df_daily)
    nearest_avwap = None
    if avwap_levels:
        nearest_avwap = min(avwap_levels, key=lambda a: abs(a.price - price)).price
    candidates = [e9, e21]
    if nearest_avwap is not None:
        candidates.append(nearest_avwap)
    zone_bottom = min(candidates)
    zone_top = max(candidates)
    mid = (zone_bottom + zone_top) / 2
    zone_width_pct = (zone_top - zone_bottom) / mid * 100 if mid > 0 else 0.0
    return zone_bottom, zone_top, round(zone_width_pct, 2)


def _detect_sweep_long(df5m: pd.DataFrame, zone_bottom: float) -> Tuple[Optional[int], float]:
    min_d, max_d = 0.003, 0.020
    lookback = min(50, len(df5m) - 5)
    subset = df5m.iloc[-(lookback + 5):-1]
    for i in range(len(subset) - 4, -1, -1):
        candle = subset.iloc[i]
        depth = (zone_bottom - candle["low"]) / zone_bottom
        if not (min_d < depth < max_d):
            continue
        for j in range(i + 1, min(i + 4, len(subset))):
            if subset.iloc[j]["close"] > zone_bottom:
                actual_idx = df5m.index[-(lookback + 5) + i]
                return actual_idx, round(depth * 100, 3)
    return None, 0.0


def _detect_sweep_short(df5m: pd.DataFrame, zone_top: float) -> Tuple[Optional[int], float]:
    min_d, max_d = 0.003, 0.020
    lookback = min(50, len(df5m) - 5)
    subset = df5m.iloc[-(lookback + 5):-1]
    for i in range(len(subset) - 4, -1, -1):
        candle = subset.iloc[i]
        depth = (candle["high"] - zone_top) / zone_top
        if not (min_d < depth < max_d):
            continue
        for j in range(i + 1, min(i + 4, len(subset))):
            if subset.iloc[j]["close"] < zone_top:
                actual_idx = df5m.index[-(lookback + 5) + i]
                return actual_idx, round(depth * 100, 3)
    return None, 0.0


def _classify_v_reversal_long(df5m: pd.DataFrame, sweep_idx: int) -> Tuple[bool, float, float, float]:
    try:
        pos = df5m.index.get_loc(sweep_idx)
    except KeyError:
        return False, 0.0, 0.0, 0.0
    if pos + 6 > len(df5m):
        return False, 0.0, 0.0, 0.0
    sweep = df5m.iloc[pos]
    sweep_low = float(sweep["low"])
    sweep_open = float(sweep["open"])
    vol_sma = float(df5m["vol_sma_20"].iloc[pos]) if not pd.isna(df5m["vol_sma_20"].iloc[pos]) else 1.0
    if sweep_open == 0:
        return False, 0.0, 0.0, 0.0
    down_pct = (sweep_open - sweep_low) / sweep_open
    if down_pct < 0.005:
        return False, 0.0, 0.0, 0.0
    reversal_high, reversal_vol, up_pct, candles_up = 0.0, 0.0, 0.0, 0
    for j in range(pos + 1, min(pos + 4, len(df5m))):
        c = df5m.iloc[j]
        candidate = (c["close"] - sweep_low) / sweep_low
        if candidate > up_pct:
            up_pct = candidate
            reversal_high = float(c["high"])
            reversal_vol = float(c["volume"])
            candles_up = j - pos
    if up_pct < 0.005:
        return False, 0.0, 0.0, 0.0
    reclaimed = any(df5m.iloc[j]["close"] > sweep_open for j in range(pos + 1, min(pos + 6, len(df5m))))
    if not reclaimed:
        return False, 0.0, 0.0, 0.0
    up_vel = up_pct / max(candles_up, 1)
    if up_vel < down_pct * 0.8:
        return False, 0.0, 0.0, 0.0
    vol_ratio = reversal_vol / vol_sma if vol_sma > 0 else 1.0
    if vol_ratio < 1.5:
        return False, 0.0, 0.0, 0.0
    return True, round(up_vel * 100, 4), round(vol_ratio, 2), reversal_high


def _build_mtf_alignment(df_5m: pd.DataFrame, df_daily: pd.DataFrame) -> Tuple[str, str]:
    a5 = detect_ema_alignment(df_5m)
    ad = detect_ema_alignment(df_daily)
    parts = []
    if a5: parts.append(f"{a5} on 5m")
    if ad: parts.append(f"{ad} on daily")
    if not parts:
        return "Mixed signals", "mixed"
    bull = sum(1 for x in [a5, ad] if x == "bullish")
    bear = sum(1 for x in [a5, ad] if x == "bearish")
    total = sum(1 for x in [a5, ad] if x is not None)
    cat = "all_bullish" if bull == total else "all_bearish" if bear == total else "partial"
    return ", ".join(parts), cat


def _quality_score(zone_w, sweep_d, rev_v, drop_v, vol_r, mtf_cat, pinch) -> int:
    s = 0
    if zone_w < 1.0: s += 2
    elif zone_w < 2.0: s += 1
    if 0.3 <= sweep_d <= 1.0: s += 2
    elif 1.0 < sweep_d <= 2.0: s += 1
    if drop_v > 0 and rev_v >= 2 * drop_v: s += 2
    elif drop_v > 0 and rev_v >= drop_v: s += 1
    if vol_r >= 2.0: s += 2
    elif vol_r >= 1.5: s += 1
    if "all_bullish" in mtf_cat or "all_bearish" in mtf_cat: s += 2
    elif "partial" in mtf_cat: s += 1
    if pinch: s += 2
    return min(s, 10)


def _no_signal(reasoning: str, **kw) -> GreenLaneSignal:
    base = dict(
        triggered=False, quality_score=0, direction="long",
        entry_price=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
        trail_ema="daily_ema9", pinch_active=False, pinch_width_pct=0.0,
        zone_width_pct=0.0, sweep_depth_pct=0.0, reversal_velocity=0.0,
        volume_ratio=0.0, mtf_alignment="", timestamp=datetime.now(timezone.utc).isoformat(),
        reasoning=reasoning,
    )
    base.update(kw)
    return GreenLaneSignal(**base)


def scan_green_lane(symbol: str = "BTC/USDT") -> GreenLaneSignal:
    """Run the green lane scanner. Pure deterministic, no LLM."""
    log.info(f"Scanning {symbol}")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    df5m = _fetch_5m(symbol, limit=200)
    if df5m is None:
        return _no_signal("Failed to fetch 5-min data", timestamp=ts)

    df_daily = compute_indicators(symbol, "1d")
    if df_daily is None:
        return _no_signal("Failed to fetch daily data", timestamp=ts)

    price = float(df5m["close"].iloc[-1])
    pinch_active, pinch_width_pct = _detect_avwap_pinch(df_daily)
    zone_bottom, zone_top, zone_width_pct = _compute_confluence_zone(df_daily)
    mtf_summary, mtf_cat = _build_mtf_alignment(df5m, df_daily)

    log.debug(f"zone=[{zone_bottom:.2f},{zone_top:.2f}] w={zone_width_pct:.2f}% price={price:.2f} MTF={mtf_summary}")

    # ── LONG ──
    sweep_idx, sweep_depth = _detect_sweep_long(df5m, zone_bottom)
    if sweep_idx is not None:
        valid, rev_v, vol_r, rev_high = _classify_v_reversal_long(df5m, sweep_idx)
        if valid and price > rev_high:
            entry = price
            sweep_low = float(df5m.loc[sweep_idx, "low"]) * 0.9995
            risk = entry - sweep_low
            stop_pct = risk / entry * 100 if entry > 0 else 0
            if risk > 0 and 1.5 <= stop_pct <= 3.0:
                tp1 = entry + 3 * risk
                tp2 = entry + 5 * risk
                sweep_open = float(df5m.loc[sweep_idx, "open"])
                drop_v = (sweep_open - float(df5m.loc[sweep_idx, "low"])) / sweep_open * 100 if sweep_open > 0 else 0
                quality = _quality_score(zone_width_pct, sweep_depth, rev_v, drop_v, vol_r, mtf_cat, pinch_active)
                reasoning = (
                    f"Long setup: zone=[{zone_bottom:.2f},{zone_top:.2f}] ({zone_width_pct:.2f}% wide), "
                    f"sweep depth={sweep_depth:.2f}%, V-rev velocity={rev_v:.4f}%/bar, "
                    f"vol_ratio={vol_r:.2f}x, MTF={mtf_summary}, pinch={pinch_active}"
                )
                log.info(f"TRIGGERED LONG q={quality}: {reasoning}")
                return GreenLaneSignal(
                    triggered=True, quality_score=quality, direction="long",
                    entry_price=round(entry, 4), stop_loss=round(sweep_low, 4),
                    tp1=round(tp1, 4), tp2=round(tp2, 4), trail_ema="daily_ema9",
                    pinch_active=pinch_active, pinch_width_pct=pinch_width_pct,
                    zone_width_pct=zone_width_pct, sweep_depth_pct=sweep_depth,
                    reversal_velocity=rev_v, volume_ratio=vol_r,
                    mtf_alignment=mtf_summary, timestamp=ts, reasoning=reasoning,
                )

    # ── SHORT ──
    sweep_idx_s, sweep_depth_s = _detect_sweep_short(df5m, zone_top)
    if sweep_idx_s is not None:
        try:
            pos = df5m.index.get_loc(sweep_idx_s)
        except KeyError:
            pos = None
        if pos is not None and pos + 6 <= len(df5m):
            sc = df5m.iloc[pos]
            sh, so = float(sc["high"]), float(sc["open"])
            vol_sma = float(df5m["vol_sma_20"].iloc[pos]) if not pd.isna(df5m["vol_sma_20"].iloc[pos]) else 1.0
            up_pct = (sh - so) / so if so > 0 else 0
            if up_pct >= 0.005:
                rev_low, rev_vol, down_pct, can_down = sh, 0.0, 0.0, 0
                for j in range(pos + 1, min(pos + 4, len(df5m))):
                    c = df5m.iloc[j]
                    cand = (sh - c["close"]) / sh
                    if cand > down_pct:
                        down_pct = cand; rev_low = float(c["low"]); rev_vol = float(c["volume"]); can_down = j - pos
                if down_pct >= 0.005:
                    vol_r = rev_vol / vol_sma if vol_sma > 0 else 1.0
                    if vol_r >= 1.5 and price < rev_low:
                        entry = price; stop = sh * 1.0005; risk = stop - entry
                        stop_pct = risk / entry * 100 if entry > 0 else 0
                        if risk > 0 and 1.5 <= stop_pct <= 3.0:
                            tp1 = entry - 3 * risk; tp2 = entry - 5 * risk
                            drop_v = up_pct * 100; rev_v = down_pct / max(can_down, 1) * 100
                            quality = _quality_score(zone_width_pct, sweep_depth_s, rev_v, drop_v, vol_r, mtf_cat, pinch_active)
                            reasoning = (
                                f"Short setup: zone=[{zone_bottom:.2f},{zone_top:.2f}] ({zone_width_pct:.2f}% wide), "
                                f"sweep depth={sweep_depth_s:.2f}%, V-rev down={rev_v:.4f}%/bar, "
                                f"vol_ratio={vol_r:.2f}x, MTF={mtf_summary}, pinch={pinch_active}"
                            )
                            log.info(f"TRIGGERED SHORT q={quality}: {reasoning}")
                            return GreenLaneSignal(
                                triggered=True, quality_score=quality, direction="short",
                                entry_price=round(entry, 4), stop_loss=round(stop, 4),
                                tp1=round(tp1, 4), tp2=round(tp2, 4), trail_ema="daily_ema9",
                                pinch_active=pinch_active, pinch_width_pct=pinch_width_pct,
                                zone_width_pct=zone_width_pct, sweep_depth_pct=sweep_depth_s,
                                reversal_velocity=rev_v, volume_ratio=vol_r,
                                mtf_alignment=mtf_summary, timestamp=ts, reasoning=reasoning,
                            )

    # ── No setup ──
    reason = (
        f"No Green Lane setup. zone=[{zone_bottom:.2f},{zone_top:.2f}] ({zone_width_pct:.2f}% wide), "
        f"price={price:.2f}, MTF={mtf_summary}, pinch={pinch_active} ({pinch_width_pct:.2f}%)"
    )
    log.info(f"Not triggered: {reason}")
    return GreenLaneSignal(
        triggered=False, quality_score=0, direction="long",
        entry_price=round(price, 4), stop_loss=0.0, tp1=0.0, tp2=0.0,
        trail_ema="daily_ema9", pinch_active=pinch_active, pinch_width_pct=pinch_width_pct,
        zone_width_pct=zone_width_pct, sweep_depth_pct=0.0, reversal_velocity=0.0,
        volume_ratio=0.0, mtf_alignment=mtf_summary, timestamp=ts, reasoning=reason,
    )
