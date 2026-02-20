"""Main entry point: fetch data, run backtest, generate report."""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.fetch_data import fetch_ohlcv
from backtest.engine import BacktestEngine, BacktestConfig


def main():
    # Fetch or load cached data
    df = fetch_ohlcv(use_cache=True)
    print(f"\nLoaded {len(df)} candles: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Configure and run backtest
    config = BacktestConfig(
        initial_capital=10_000.0,
        commission_rate=0.001,  # 0.1%
    )

    engine = BacktestEngine(df, config)
    metrics = engine.run()
    engine.generate_report(metrics)

    print("\nBacktest complete!")
    return metrics


if __name__ == "__main__":
    main()
