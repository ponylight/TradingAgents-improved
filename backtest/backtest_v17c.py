"""V17c — Smart Aggressive with WIDER stops.

Changes from V17b:
  - BASE trailing: 3x ATR (was 2x) — let winners run
  - ADD trailing: 2.5x ATR (was 2x) — more breathing room
  - Entry cooldown: 2 weeks (was 1)
  - SMA proximity: 3% (tighter filter for add quality)
  
Same as V17b:
  - 3x base leverage, 5x/10x adds
  - 30% margin
  - Capital + profits for adds
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backtest.fetch_data import fetch_ohlcv
from backtest.indicators import compute_macd, compute_atr, compute_ma

# Just import and modify V17b
from backtest.backtest_v17b import (
    SmartAggressiveBacktest, resample_to_weekly,
    CAPITAL, Trade
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_v17c")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_4h = fetch_ohlcv(use_cache=True)
    weekly = resample_to_weekly(df_4h)
    print(f"Loaded {len(df_4h)} 4H → {len(weekly)} weekly candles")

    # Monkey-patch the constants for V17c
    import backtest.backtest_v17b as v17b
    v17b.TRAILING_ATR_MULT = 3.0   # wider base trailing
    v17b.ADD_ATR_MULT = 2.5        # wider add stops
    v17b.MIN_COOLDOWN = 2
    v17b.SMA_PROXIMITY_PCT = 0.03

    bt = SmartAggressiveBacktest(weekly)
    metrics = bt.run()
    metrics["system"] = "v17c_wide_stops"

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Compact report
    lines = ["=" * 70, "  V17c — SMART AGGRESSIVE + WIDE STOPS", "=" * 70]
    lines.append(f"\nBASE: 3x, 30% margin | TRAIL: 3x ATR | ADD STOP: 2.5x ATR")
    lines.append(f"ADD LEV: 5x/10x | PYRAMID: max 5 adds\n")
    for k, v in metrics.items():
        if k == "system": continue
        lines.append(f"  {k:35s}: {v:>12,.2f}" if isinstance(v, float) else f"  {k:35s}: {v:>12}")
    lines.append("\n── TRADE LOG ──")
    for e in bt.trade_log: lines.append(f"  {e}")
    
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*50}")
    print(f"  V17c WIDE STOPS")
    print(f"{'='*50}")
    print(f"  Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"  Max DD:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Trades:       {metrics['total_trades']} ({metrics['pyramid_adds']} adds)")
    print(f"  Win Rate:     {metrics['win_rate_pct']:.1f}%")
    print(f"  Largest Win:  ${metrics['largest_win']:,.2f}")
    print(f"  Funding:      ${metrics['total_funding_cost']:,.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
