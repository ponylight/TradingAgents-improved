"""
Tier 1 -- Deterministic Crypto Technical Analysis Engine.

Computes indicators across 1h/4h/1d from Bybit via ccxt, detects regime,
extracts key levels, produces a standardized TechnicalBrief JSON.

**No LLM calls.** Pure math. Fast. Reliable.
Adapted from AlpacaTradingAgent for crypto/ccxt.
"""

from __future__ import annotations
import logging
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

_log = logging.getLogger("crypto_technical_brief")

from .ta_schema import (
    AVWAPLevel, Direction, KeyLevel, MarketStructure, MomentumState,
    SignalSummary, Strength, TechnicalBrief, TimeframeBrief,
    TrendState, VolatilityState, VolumeState, VWAPState,
)

warnings.filterwarnings("ignore", category=FutureWarning)

TIMEFRAMES = {"1h": ("1h", 200), "4h": ("4h", 200), "1d": ("1d", 200)}

# ── Indicator helpers ──

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(window=p).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def _stoch_rsi(close, period=14, k=3, d=3):
    rsi = _rsi(close, period)
    mn, mx = rsi.rolling(period).min(), rsi.rolling(period).max()
    st = ((rsi - mn) / (mx - mn)) * 100
    return st.rolling(k).mean(), st.rolling(k).mean().rolling(d).mean()

def _adx(high, low, close, period=14):
    pdm = high.diff().clip(lower=0)
    mdm = low.diff().abs(); mdm[low.diff() > 0] = 0
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100*(pdm.ewm(alpha=1/period).mean()/atr)
    mdi = 100*(mdm.ewm(alpha=1/period).mean()/atr)
    dx = (abs(pdi-mdi)/abs(pdi+mdi))*100
    return dx.ewm(alpha=1/period).mean()

def _macd(close, fast=12, slow=26, sig=9):
    ml = _ema(close, fast) - _ema(close, slow)
    sl = _ema(ml, sig)
    return ml, sl, ml - sl

def _atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _bollinger(close, period=20, nstd=2.0):
    s = _sma(close, period); sd = close.rolling(period).std()
    return s+nstd*sd, s-nstd*sd, (2*nstd*sd)/s

def _obv(close, vol): return (np.sign(close.diff()).fillna(0)*vol).cumsum()

# ── OHLCV fetch ──

# Module-level exchange instance (reuse across calls)
_exchange = None

def _get_exchange():
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    return _exchange

def _fetch_ohlcv(symbol, tf_key):
    ccxt_tf, limit = TIMEFRAMES[tf_key]
    try:
        ex = _get_exchange()
        raw = ex.fetch_ohlcv(f"{symbol}:USDT", timeframe=ccxt_tf, limit=limit)
        if not raw or len(raw) < 30: return None
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"[TA-BRIEF] Error {symbol}@{tf_key}: {e}")
        return None

def compute_indicators(symbol, tf_key):
    df = _fetch_ohlcv(symbol, tf_key)
    if df is None: return None
    df["ema_8"]=_ema(df["close"],8); df["ema_9"]=_ema(df["close"],9); df["ema_21"]=_ema(df["close"],21); df["ema_50"]=_ema(df["close"],50)
    df["sma_50"]=_sma(df["close"],50); df["sma_200"]=_sma(df["close"],200)
    df["adx_14"]=_adx(df["high"],df["low"],df["close"],14)
    df["rsi_14"]=_rsi(df["close"],14)
    df["stoch_k"],df["stoch_d"]=_stoch_rsi(df["close"])
    df["macd"],df["macds"],df["macdh"]=_macd(df["close"])
    df["atr_14"]=_atr(df["high"],df["low"],df["close"],14)
    df["boll_ub"],df["boll_lb"],df["boll_bw"]=_bollinger(df["close"])
    df["obv"]=_obv(df["close"],df["volume"])
    df["vol_sma_20"]=_sma(df["volume"],20)
    # Rolling VWAP (20-bar) — cumulative from bar 0 drifts, producing nonsensical z-scores
    _tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (_tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    return df

# ── Detectors ──


# ── Anchored VWAP ──

def _find_swing_points(df, lookback=40):
    """Find recent swing high and swing low indices."""
    r = df.tail(lookback)
    if len(r) < 10:
        return None, None
    h, l = r["high"].values, r["low"].values
    idx = r.index.values
    sh_idx, sl_idx = None, None
    for i in range(len(r)-3, 2, -1):
        if sh_idx is None and h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            sh_idx = idx[i]
        if sl_idx is None and l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            sl_idx = idx[i]
        if sh_idx is not None and sl_idx is not None:
            break
    return sh_idx, sl_idx

def _find_volume_spike(df, lookback=40, threshold=2.0):
    """Find most recent bar with volume > threshold * SMA20."""
    r = df.tail(lookback)
    if "vol_sma_20" not in r.columns:
        return None
    for i in range(len(r)-1, -1, -1):
        v = r["volume"].iloc[i]
        vs = r["vol_sma_20"].iloc[i]
        if not pd.isna(vs) and vs > 0 and v > threshold * vs:
            return r.index[i]
    return None

def _calc_avwap(df, anchor_idx):
    """Calculate AVWAP from anchor_idx to end of df."""
    if anchor_idx is None or anchor_idx not in df.index:
        return None
    subset = df.loc[anchor_idx:]
    if len(subset) < 2:
        return None
    tp = (subset["high"] + subset["low"] + subset["close"]) / 3
    cv = (tp * subset["volume"]).cumsum()
    vv = subset["volume"].cumsum()
    avwap = cv / vv.replace(0, np.nan)
    return float(avwap.iloc[-1]) if not pd.isna(avwap.iloc[-1]) else None

def compute_avwap_levels(df):
    """Compute AVWAP from swing high, swing low, and volume spike anchors."""
    levels = []
    last_idx = df.index[-1]
    sh_idx, sl_idx = _find_swing_points(df)
    vol_idx = _find_volume_spike(df)
    
    for label, idx in [("swing_high", sh_idx), ("swing_low", sl_idx), ("volume_spike", vol_idx)]:
        price = _calc_avwap(df, idx)
        if price is not None:
            bars_ago = int(last_idx - idx)
            levels.append(AVWAPLevel(anchor=label, price=round(price, 2), anchor_bar_ago=bars_ago))
    return levels

# ── EMA Convergence Zone ──

def detect_ema_convergence(df):
    """Check if EMA 9/21/50 are within 2% of each other."""
    e9 = float(df["ema_9"].iloc[-1]) if not pd.isna(df["ema_9"].iloc[-1]) else None
    e21 = float(df["ema_21"].iloc[-1]) if not pd.isna(df["ema_21"].iloc[-1]) else None
    e50 = float(df["ema_50"].iloc[-1]) if not pd.isna(df["ema_50"].iloc[-1]) else None
    if e9 is None or e21 is None or e50 is None:
        return False, 0.0
    mid = (e9 + e21 + e50) / 3
    if mid == 0:
        return False, 0.0
    spread_pct = (max(e9, e21, e50) - min(e9, e21, e50)) / mid * 100
    return spread_pct <= 2.0, round(spread_pct, 2)

# ── Liquidity Sweep Detection ──

def detect_liquidity_sweep(df, lookback=30):
    """Detect if last candle wicked below swing low (bullish) or above swing high (bearish) then closed back."""
    if len(df) < lookback:
        return None
    sh_idx, sl_idx = _find_swing_points(df.iloc[:-1], lookback=lookback)
    last = df.iloc[-1]
    
    if sl_idx is not None:
        sl_price = float(df.loc[sl_idx, "low"])
        if last["low"] < sl_price and last["close"] > sl_price:
            return "bullish_sweep"
    
    if sh_idx is not None:
        sh_price = float(df.loc[sh_idx, "high"])
        if last["high"] > sh_price and last["close"] < sh_price:
            return "bearish_sweep"
    
    return None

# ── EMA Alignment ──

def detect_ema_alignment(df):
    """Check if EMA 9 > 21 > 50 (bullish) or 9 < 21 < 50 (bearish)."""
    e9 = float(df["ema_9"].iloc[-1]) if not pd.isna(df["ema_9"].iloc[-1]) else None
    e21 = float(df["ema_21"].iloc[-1]) if not pd.isna(df["ema_21"].iloc[-1]) else None
    e50 = float(df["ema_50"].iloc[-1]) if not pd.isna(df["ema_50"].iloc[-1]) else None
    if e9 is None or e21 is None or e50 is None:
        return None
    if e9 > e21 > e50:
        return "bullish"
    elif e9 < e21 < e50:
        return "bearish"
    return None

def detect_trend(df):
    c=df["close"].iloc[-1]; e8=df["ema_8"].iloc[-1]; e21=df["ema_21"].iloc[-1]; s50=df["sma_50"].iloc[-1]
    s200=float(df["sma_200"].iloc[-1]) if not pd.isna(df["sma_200"].iloc[-1]) else 0.0
    er=df["ema_8"].iloc[-6:]
    es=(er.iloc[-1]/er.iloc[0]-1)*100 if len(er)>=2 and er.iloc[0]!=0 else 0.0
    bu=int(e8>e21)+int(e21>s50)+int(c>e8)
    be=int(e8<e21)+int(e21<s50)+int(c<e8)
    d=Direction.BULLISH if bu>=2 else Direction.BEARISH if be>=2 else Direction.NEUTRAL
    st=Strength.STRONG if abs(es)>1.5 else Strength.MODERATE if abs(es)>0.5 else Strength.WEAK
    hh,hl=_detect_hh_hl(df)
    av=float(df["adx_14"].iloc[-1]) if not pd.isna(df["adx_14"].iloc[-1]) else 0.0
    at="very_strong" if av>40 else "strong" if av>25 else "weak"
    sd=((c-s200)/s200)*100 if s200>0 else 0.0

    # === Indicator Reconciliation ===
    # EMA/SMA scoring can conflict with structural signals (higher_highs, higher_lows, ema_slope).
    # When they conflict, override to NEUTRAL to avoid false directional bias.
    if d == Direction.BEARISH and hh and hl and es > 0.5:
        _log.warning(
            f"⚠️ TREND RECONCILIATION: EMA/SMA scored BEARISH but structure is bullish "
            f"(higher_highs=True, higher_lows=True, ema_slope={es:.2f}) — overriding to NEUTRAL"
        )
        d = Direction.NEUTRAL
    elif d == Direction.BULLISH and not hh and not hl and es < -0.5:
        _log.warning(
            f"⚠️ TREND RECONCILIATION: EMA/SMA scored BULLISH but structure is bearish "
            f"(higher_highs=False, higher_lows=False, ema_slope={es:.2f}) — overriding to NEUTRAL"
        )
        d = Direction.NEUTRAL

    return TrendState(direction=d,strength=st,ema_slope=round(es,4),higher_highs=hh,higher_lows=hl,
                     adx=round(av,2),trend_strength_adx=at,sma_200=round(s200,2),sma_200_dist=round(sd,2))

def _detect_hh_hl(df, lookback=20):
    r=df.tail(lookback)
    if len(r)<10: return False,False
    h,l=r["high"].values,r["low"].values
    sh,sl=[],[]
    for i in range(2,len(r)-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]: sh.append(h[i])
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]: sl.append(l[i])
    return (len(sh)>=2 and sh[-1]>sh[-2]),(len(sl)>=2 and sl[-1]>sl[-2])

def detect_momentum(df):
    rv=float(df["rsi_14"].iloc[-1]) if not pd.isna(df["rsi_14"].iloc[-1]) else 50.0
    rz="oversold" if rv<30 else "overbought" if rv>70 else "neutral"
    rd=False
    if len(df)>14:
        r=df.tail(14)
        ph=r["close"].iloc[-1]>=r["close"].max()*0.99
        rl=r["rsi_14"].iloc[-1]<r["rsi_14"].iloc[:7].max()
        pl=r["close"].iloc[-1]<=r["close"].min()*1.01
        rh=r["rsi_14"].iloc[-1]>r["rsi_14"].iloc[:7].min()
        rd=bool((ph and rl)or(pl and rh))
    mc="none"
    if len(df)>1:
        m1,s1=df["macd"].iloc[-1],df["macds"].iloc[-1]
        m0,s0=df["macd"].iloc[-2],df["macds"].iloc[-2]
        if not(pd.isna(m1)or pd.isna(s1)):
            if m0<=s0 and m1>s1: mc="bullish"
            elif m0>=s0 and m1<s1: mc="bearish"
    hi=df["macdh"].iloc[-3:]
    if len(hi.dropna())>=2:
        dd=hi.diff().dropna()
        ht="flat" if(dd.abs()<0.001).all() else "expanding" if(dd>0).all() else "contracting"
    else: ht="flat"
    sk=float(df["stoch_k"].iloc[-1]) if not pd.isna(df["stoch_k"].iloc[-1]) else 50.0
    sd=float(df["stoch_d"].iloc[-1]) if not pd.isna(df["stoch_d"].iloc[-1]) else 50.0
    ss="overbought" if sk>80 else "oversold" if sk<20 else "neutral"
    return MomentumState(rsi_value=round(rv,1),rsi_zone=rz,rsi_divergence=rd,macd_cross=mc,
                        macd_histogram_trend=ht,stoch_k=round(sk,2),stoch_d=round(sd,2),stoch_state=ss)

def detect_vwap_state(df):
    c,v=df["close"].iloc[-1],df["vwap"].iloc[-1]
    if pd.isna(v) or v==0: return VWAPState(position="at",zscore_distance=0.0)
    s=df["close"].iloc[-20:].std()
    z=(c-v)/s if s and s>0 else 0.0
    p="at" if abs(z)<0.3 else "above" if z>0 else "below"
    avwaps = compute_avwap_levels(df)
    return VWAPState(position=p,zscore_distance=round(z,2),anchored_vwaps=avwaps)

def detect_volatility(df):
    av=float(df["atr_14"].iloc[-1]) if not pd.isna(df["atr_14"].iloc[-1]) else 0.0
    at=df["atr_14"].dropna().tail(90)
    ap=float((at<av).sum()/len(at)*100) if len(at)>5 else 50.0
    bs=df["boll_bw"].dropna().tail(90)
    bc=df["boll_bw"].iloc[-1] if not pd.isna(df["boll_bw"].iloc[-1]) else 0.0
    bp=float((bs<bc).sum()/len(bs)*100) if len(bs)>5 else 50.0
    gp=((df["open"].iloc[-1]-df["close"].iloc[-2])/df["close"].iloc[-2])*100 if len(df)>=2 else 0.0
    return VolatilityState(atr_value=round(av,4),atr_percentile=round(ap,1),
                          squeeze=bp<20,breakout=bp>80,gap_percent=round(gp,2))

def detect_volume(df):
    # Use second-to-last bar to avoid incomplete candle skewing ratios
    idx = -2 if len(df) >= 3 else -1
    vl=float(df["volume"].iloc[idx])
    vs=float(df["vol_sma_20"].iloc[idx]) if not pd.isna(df["vol_sma_20"].iloc[idx]) else 1.0
    vr=vl/vs if vs>0 else 1.0
    sr=df["vol_sma_20"].iloc[-6:-1] if len(df) >= 7 else df["vol_sma_20"].tail(5)
    sl=(sr.iloc[-1]/sr.iloc[0])-1 if len(sr)>=5 and sr.iloc[0]>0 else 0
    vt="up" if sl>0.05 else "down" if sl<-0.05 else "flat"
    ob=df["obv"].iloc[-6:-1] if len(df) >= 7 else df["obv"].tail(5)
    os_=float(ob.iloc[-1]-ob.iloc[0]) if len(ob)>=5 else 0.0
    return VolumeState(vol_ma_ratio=round(vr,2),vol_trend=vt,obv_slope=round(os_,2))

def detect_market_structure(df):
    r=df.tail(40); h,l=r["high"].values,r["low"].values
    sh,sl=[],[]
    for i in range(3,len(r)-3):
        if all(h[i]>h[i-j] for j in range(1,4)) and all(h[i]>h[i+j] for j in range(1,4)): sh.append(float(h[i]))
        if all(l[i]<l[i-j] for j in range(1,4)) and all(l[i]<l[i+j] for j in range(1,4)): sl.append(float(l[i]))
    lsh=sh[-1] if sh else float(df["high"].iloc[-1])
    lsl=sl[-1] if sl else float(df["low"].iloc[-1])
    bos=df["close"].iloc[-1]>sh[-2] if len(sh)>=2 else False
    choch=False
    if len(sh)>=2 and len(sl)>=2:
        choch=(sh[-1]>sh[-2] and sl[-1]<sl[-2])or(sh[-1]<sh[-2] and sl[-1]>sl[-2])
    return MarketStructure(bos=bos,choch=choch,last_swing_high=round(lsh,2),last_swing_low=round(lsl,2))

def extract_key_levels(dfs):
    levels=[]
    daily=dfs.get("1d"); daily=daily if daily is not None else next((v for v in dfs.values() if v is not None),None)
    if daily is None: return levels
    c=float(daily["close"].iloc[-1])
    if len(daily)>=2:
        ph,pl,pc=float(daily["high"].iloc[-2]),float(daily["low"].iloc[-2]),float(daily["close"].iloc[-2])
        pv=(ph+pl+pc)/3
        levels+=[KeyLevel(label="Prev High",price=round(ph,2),type="resistance" if ph>c else "support"),
                 KeyLevel(label="Prev Low",price=round(pl,2),type="support" if pl<c else "resistance"),
                 KeyLevel(label="Pivot",price=round(pv,2),type="pivot"),
                 KeyLevel(label="R1",price=round(2*pv-pl,2),type="resistance"),
                 KeyLevel(label="S1",price=round(2*pv-ph,2),type="support")]
    if len(daily)>=30:
        hi,lo=daily["high"].max(),daily["low"].min()
        d=hi-lo
        if d>0:
            for f,lb in [(0.236,"Fib .236"),(0.382,"Fib .382"),(0.5,"Fib .5"),(0.618,"Fib .618")]:
                p=round(hi-f*d,2)
                levels.append(KeyLevel(label=lb,price=p,type="support" if c>p else "resistance"))
    for tf in ("4h","1h"):
        td=dfs.get(tf)
        if td is not None and "boll_ub" in td.columns:
            if not pd.isna(td["boll_ub"].iloc[-1]):
                levels.append(KeyLevel(label=f"BB Upper({tf})",price=round(float(td["boll_ub"].iloc[-1]),2),type="resistance"))
            if not pd.isna(td["boll_lb"].iloc[-1]):
                levels.append(KeyLevel(label=f"BB Lower({tf})",price=round(float(td["boll_lb"].iloc[-1]),2),type="support"))
            break
    levels.sort(key=lambda l:l.price)
    deduped=[levels[0]] if levels else []
    for lv in levels[1:]:
        if c>0 and abs(lv.price-deduped[-1].price)/c*100<0.3:
            if len(lv.label)>len(deduped[-1].label): deduped[-1]=lv
        else: deduped.append(lv)
    deduped.sort(key=lambda l:abs(l.price-c))
    return deduped[:5]

def generate_signal_summary(briefs, levels):
    if not briefs: return SignalSummary(setup="none",confidence="low",description="Insufficient data.")
    dirs=[b.trend.direction for b in briefs]
    bu=sum(1 for d in dirs if d==Direction.BULLISH)
    be=sum(1 for d in dirs if d==Direction.BEARISH)
    bias="bullish" if bu>=2 else "bearish" if be>=2 else "neutral"
    sq=any(b.volatility.squeeze for b in briefs)
    bo=any(b.volatility.breakout for b in briefs)
    bs=any(b.market_structure.bos for b in briefs)
    rz=[b.momentum.rsi_zone for b in briefs]
    if bo and bs: setup,desc="breakout",f"Volatility breakout+BOS, {bias}"
    elif sq: setup,desc="breakout",f"BB squeeze, potential breakout, {bias}"
    elif bias=="bullish" and "oversold" in rz: setup,desc="mean_reversion","Bullish+oversold RSI"
    elif bias=="bearish" and "overbought" in rz: setup,desc="mean_reversion","Bearish+overbought RSI"
    elif bias!="neutral":
        if briefs[0].trend.direction.value!=bias: setup,desc="pullback",f"{bias.title()} HTF pullback"
        else: setup,desc="trend_continuation",f"Multi-TF {bias} continuation"
    else: setup,desc="none","Mixed signals"
    al=bu==len(briefs) or be==len(briefs)
    mc=sum(1 for b in briefs if(bias=="bullish" and b.momentum.macd_cross=="bullish")or(bias=="bearish" and b.momentum.macd_cross=="bearish"))
    conf="high" if al and mc>=1 else "medium" if bu>=2 or be>=2 else "low"
    return SignalSummary(setup=setup,confidence=conf,description=desc)

# ── Main ──

def build_crypto_technical_brief(symbol="BTC/USDT"):
    """Build complete TechnicalBrief. No LLM calls. ~5 seconds."""
    dfs={}; tbs=[]
    for tf in ("1h","4h","1d"):
        df=compute_indicators(symbol,tf)
        if df is None: continue
        dfs[tf]=df
        conv, conv_pct = detect_ema_convergence(df)
        sweep = detect_liquidity_sweep(df) if tf == "1h" else None
        alignment = detect_ema_alignment(df)
        b=TimeframeBrief(timeframe=tf,trend=detect_trend(df),momentum=detect_momentum(df),
                        vwap_state=detect_vwap_state(df),volatility=detect_volatility(df),
                        volume=detect_volume(df),market_structure=detect_market_structure(df),
                        ema_convergence=conv,ema_convergence_pct=conv_pct,
                        liquidity_sweep=sweep,ema_alignment=alignment)
        tbs.append(b)
    levels=extract_key_levels(dfs)
    signal=generate_signal_summary(tbs,levels)
    rp={"last_close":0.0,"prev_close":0.0,"daily_change_pct":0.0}
    d=dfs.get("1d")
    if d is not None and len(d)>=2:
        lc,pc=float(d["close"].iloc[-1]),float(d["close"].iloc[-2])
        rp={"last_close":round(lc,2),"prev_close":round(pc,2),"daily_change_pct":round((lc/pc-1)*100,2) if pc else 0.0}
    # Build MTF EMA alignment summary
    mtf_parts = []
    for b in tbs:
        if b.ema_alignment:
            mtf_parts.append(f"{b.ema_alignment.title()} on {b.timeframe}")
    mtf_summary = "MTF EMA Alignment: " + ", ".join(mtf_parts) if mtf_parts else "MTF EMA Alignment: Mixed"

    return TechnicalBrief(symbol=symbol,generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         timeframes=tbs,key_levels=levels,signal_summary=signal,raw_prices=rp,
                         mtf_ema_alignment=mtf_summary)
