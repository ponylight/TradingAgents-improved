# Task: Implement MACD Triple Divergence Detector

## Context
Implementing 半木夏's MACD Triple Divergence strategy as a standalone indicator module.
This detects when price makes 3 successive peaks/troughs while MACD histogram shrinks twice consecutively — a high-probability reversal signal.

## Strategy Rules (from 半木夏):
- **Parameters:** Fast=12, Slow=26, Signal=9 (standard MACD)
- **Triple Top Divergence (Bearish):** Price makes 3 highs, MACD histogram green bars shrink twice
- **Triple Bottom Divergence (Bullish):** Price makes 3 lows, MACD histogram red bars shrink twice
- Each histogram segment must have opposite-color bars between them (validates separate waves)
- Aligns with Elliott Wave 5-wave completion
- **Best timeframes:** Daily and 4H
- **Frequency:** ~1-2 times per year on daily, more on 4H
- **Exit if next candle doesn't continue the histogram direction**

## Implementation Plan:

### 1. Create new module: `tradingagents/dataflows/macd_divergence.py`

```python
def detect_triple_divergence(ohlcv_data, fast=12, slow=26, signal=9) -> dict:
    """
    Detect MACD triple divergence on OHLCV data.
    
    Returns:
        {
            "signal": "BEARISH_DIVERGENCE" | "BULLISH_DIVERGENCE" | "NONE",
            "confidence": 0-10,
            "peaks_or_troughs": [(idx, price, macdh), ...],  # The 3 points
            "histogram_shrinkage": [pct1, pct2],  # How much histogram shrank
            "timeframe": str,
            "description": str,  # Human-readable explanation
        }
    """
```

### 2. Logic:
- Calculate MACD, Signal, Histogram from close prices
- Segment histogram into positive (green) and negative (red) runs
- For bearish: find 3 positive histogram segments where peak values decrease twice consecutively, AND price at each segment's peak increases
- For bullish: find 3 negative histogram segments where trough values get less negative twice consecutively, AND price at each segment's trough decreases
- Score confidence based on: how much histogram shrunk, timeframe level, how clean the pattern is

### 3. Integration hook in `tradingagents/agents/analysts/crypto_market_analyst.py`:
- Add a tool function `check_macd_divergence(symbol, timeframe)` that the market analyst can call
- It should use the existing `get_crypto_ohlcv()` to fetch data and then run the detector
- Register it as a tool the analyst can use

### 4. Also add to `tradingagents/agents/utils/crypto_tools.py`:
- Register the tool so it's available to agents

### 5. Test with current BTC data:
```python
cd ~/TradingAgents-improved
.venv/bin/python3 -c "
from tradingagents.dataflows.macd_divergence import detect_triple_divergence
from tradingagents.dataflows.ccxt_crypto import get_crypto_ohlcv
import pandas as pd
# Get 6 months of daily data
data = get_crypto_ohlcv('BTC/USDT', '1d', days_back=180)
print(data[:200])  # Show raw data format first
"
```

### Important:
- Use only standard libraries + pandas + numpy (already in the venv)
- Make it a standalone module that can be tested independently
- The detector should work on any OHLCV DataFrame
- Include logging for when divergences are detected
- Handle edge cases: not enough data, no clear segments, etc.

When completely finished, run: openclaw system event --text "Done: MACD Triple Divergence detector implemented and tested" --mode now
