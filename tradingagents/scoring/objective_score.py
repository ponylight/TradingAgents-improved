"""
Objective Scoring Guardrail

Rule-based mechanical scoring that runs alongside LLM agent decisions.
Can veto or flag absurd decisions (e.g. holding a short through 4 green candles).

Score range: -100 (strong bearish) to +100 (strong bullish)
  >= +25: BUY signal
  <= -25: SELL signal
  between: NEUTRAL

The score does NOT make the decision — it flags conflicts with the LLM decision.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

log = logging.getLogger("objective_score")


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of the objective score."""
    technical: float = 0.0
    momentum: float = 0.0
    volume: float = 0.0
    sentiment: float = 0.0
    macro: float = 0.0
    overall: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def signal(self) -> str:
        if self.overall >= 25:
            return "BUY"
        elif self.overall <= -25:
            return "SELL"
        return "NEUTRAL"

    @property
    def strength(self) -> str:
        a = abs(self.overall)
        if a >= 70:
            return "STRONG"
        elif a >= 40:
            return "MODERATE"
        elif a >= 25:
            return "WEAK"
        return "NEUTRAL"

    def conflicts_with(self, agent_decision: str) -> bool:
        """Check if objective score conflicts with agent's decision."""
        agent = agent_decision.upper().strip()
        sig = self.signal

        # Direct conflicts
        if sig == "BUY" and agent in ("SELL", "OPEN_SHORT", "REVERSE_TO_SHORT"):
            return True
        if sig == "SELL" and agent in ("BUY", "OPEN_LONG", "REVERSE_TO_LONG"):
            return True
        # Holding against a strong signal only counts as a conflict when the signal is truly strong.
        # Moderate scores are common in noisy/choppy conditions and should not force action by themselves.
        if self.strength == "STRONG" and agent in ("HOLD", "STAY_NEUTRAL"):
            return True
        return False


def score_technical(brief) -> float:
    """Score from technical brief data. Returns -100 to +100."""
    score = 0.0
    weights = 0.0

    for tf in brief.timeframes:
        # Weight: 1d=3, 4h=2, 1h=1
        w = {"1d": 3.0, "4h": 2.0, "1h": 1.0}.get(tf.timeframe, 1.0)

        tf_score = 0.0

        # Trend direction
        if tf.trend.direction == "bullish":
            tf_score += 20
        elif tf.trend.direction == "bearish":
            tf_score -= 20

        # EMA slope sign
        if tf.trend.ema_slope > 0:
            tf_score += 10
        elif tf.trend.ema_slope < 0:
            tf_score -= 10

        # BOS (break of structure) — market_structure.bos is bool
        if tf.market_structure.bos:
            # BOS in trend direction
            if tf.trend.direction == "bullish":
                tf_score += 15
            elif tf.trend.direction == "bearish":
                tf_score -= 15

        # Higher highs / lower lows
        if tf.trend.higher_highs and tf.trend.higher_lows:
            tf_score += 10
        elif not tf.trend.higher_highs and not tf.trend.higher_lows:
            tf_score -= 10

        # ADX trend strength
        adx = tf.trend.adx
        if adx and adx > 25:
            # Strong trend — amplify direction
            if tf.trend.direction == "bullish":
                tf_score += 5
            elif tf.trend.direction == "bearish":
                tf_score -= 5

        score += tf_score * w
        weights += w

    if weights > 0:
        score = score / weights
    return max(-100, min(100, score))


def score_momentum(brief) -> float:
    """Score from momentum indicators. Returns -100 to +100."""
    score = 0.0
    weights = 0.0

    for tf in brief.timeframes:
        w = {"1d": 3.0, "4h": 2.0, "1h": 1.0}.get(tf.timeframe, 1.0)
        tf_score = 0.0

        # RSI
        rsi = tf.momentum.rsi_value
        if rsi > 70:
            tf_score -= 30  # Overbought
        elif rsi > 60:
            tf_score -= 10
        elif rsi < 30:
            tf_score += 30  # Oversold
        elif rsi < 40:
            tf_score += 10

        # Stochastic RSI
        k = tf.momentum.stoch_k
        if k > 80:
            tf_score -= 15
        elif k < 20:
            tf_score += 15

        # MACD cross direction
        macd_cross = tf.momentum.macd_cross
        if macd_cross == "bullish":
            tf_score += 15
        elif macd_cross == "bearish":
            tf_score -= 15

        # MACD histogram trend
        hist_trend = tf.momentum.macd_histogram_trend
        if hist_trend == "expanding" and macd_cross == "bullish":
            tf_score += 5
        elif hist_trend == "expanding" and macd_cross == "bearish":
            tf_score -= 5

        score += tf_score * w
        weights += w

    if weights > 0:
        score = score / weights
    return max(-100, min(100, score))


def score_volume(brief) -> float:
    """Score from volume analysis. Returns -100 to +100."""
    score = 0.0
    weights = 0.0

    for tf in brief.timeframes:
        w = {"1d": 3.0, "4h": 2.0, "1h": 1.0}.get(tf.timeframe, 1.0)
        tf_score = 0.0

        # Volume/MA ratio — high volume confirms trend
        vr = tf.volume.vol_ma_ratio
        if vr > 1.5:
            tf_score += 15  # High volume — confirms current trend
        elif vr < 0.5:
            tf_score -= 10  # Low volume — suspicious

        # OBV slope
        obv = tf.volume.obv_slope
        if obv > 0:
            tf_score += 10
        elif obv < 0:
            tf_score -= 10

        # Volume trend
        if tf.volume.vol_trend == "up":
            tf_score += 5
        elif tf.volume.vol_trend == "down":
            tf_score -= 5

        score += tf_score * w
        weights += w

    if weights > 0:
        score = score / weights
    return max(-100, min(100, score))


def score_sentiment(sentiment_data: Optional[Dict] = None) -> float:
    """Score from sentiment data. Returns -100 to +100."""
    if not sentiment_data:
        return 0.0

    score = 0.0

    # Funding rate — most important real-time signal
    funding = sentiment_data.get("funding_rate", 0)
    if funding > 0.01:
        score -= 20  # Longs paying heavily — overextended
    elif funding > 0.005:
        score -= 10
    elif funding < -0.005:
        score += 20  # Shorts paying — squeeze potential
    elif funding < 0:
        score += 10

    # OI change
    oi_change = sentiment_data.get("oi_change_pct", 0)
    if oi_change > 10:
        score += 10  # Building positions
    elif oi_change < -10:
        score -= 10  # Unwinding

    return max(-100, min(100, score))


def score_macro(macro_data: Optional[Dict] = None) -> float:
    """Score from macro environment. Returns -100 to +100."""
    if not macro_data:
        return 0.0

    score = 0.0

    # VIX
    vix = macro_data.get("vix", 0)
    if vix > 35:
        score -= 30
    elif vix > 25:
        score -= 15
    elif vix < 15:
        score += 10

    # DXY change
    dxy_change = macro_data.get("dxy_change_pct", 0)
    if dxy_change > 1:
        score -= 15  # Strong dollar = bearish crypto
    elif dxy_change < -1:
        score += 15

    return max(-100, min(100, score))


def calculate_objective_score(
    brief=None,
    sentiment_data: Optional[Dict] = None,
    macro_data: Optional[Dict] = None,
) -> ScoreBreakdown:
    """
    Calculate the full objective score from all available data.

    Args:
        brief: CryptoTechnicalBrief from build_crypto_technical_brief()
        sentiment_data: dict with funding_rate, oi_change_pct, etc.
        macro_data: dict with vix, dxy_change_pct, etc.

    Returns:
        ScoreBreakdown with per-component and overall scores.
    """
    tech = score_technical(brief) if brief else 0.0
    mom = score_momentum(brief) if brief else 0.0
    vol = score_volume(brief) if brief else 0.0
    sent = score_sentiment(sentiment_data)
    macro = score_macro(macro_data)

    # Weighted overall: tech 25%, momentum 25%, volume 15%, sentiment 20%, macro 15%
    overall = (
        tech * 0.25
        + mom * 0.25
        + vol * 0.15
        + sent * 0.20
        + macro * 0.15
    )

    breakdown = ScoreBreakdown(
        technical=round(tech, 1),
        momentum=round(mom, 1),
        volume=round(vol, 1),
        sentiment=round(sent, 1),
        macro=round(macro, 1),
        overall=round(overall, 1),
        details={
            "signal": "BUY" if overall >= 25 else "SELL" if overall <= -25 else "NEUTRAL",
            "strength": "STRONG" if abs(overall) >= 70 else "MODERATE" if abs(overall) >= 40 else "WEAK" if abs(overall) >= 25 else "NEUTRAL",
        },
    )

    log.info(
        f"📊 Objective Score: {breakdown.overall:+.1f} ({breakdown.signal} {breakdown.strength}) "
        f"| tech={breakdown.technical:+.1f} mom={breakdown.momentum:+.1f} vol={breakdown.volume:+.1f} "
        f"sent={breakdown.sentiment:+.1f} macro={breakdown.macro:+.1f}"
    )

    return breakdown
