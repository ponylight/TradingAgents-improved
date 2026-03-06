"""
Backtesting tools for agents — wraps ClawQuant CLI for strategy validation.

Lets the Market Analyst test hypotheses like "would a breakout strategy have
worked in the last 30 days?" with actual backtested numbers.
"""

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("backtest_tools")

VENV_PYTHON = str(Path(__file__).parent.parent.parent.parent / ".venv" / "bin" / "clawquant")


def _run_clawquant(args: list, timeout: int = 120) -> dict:
    """Run a clawquant command and return JSON output."""
    cmd = [VENV_PYTHON, "--json"] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(Path.home())
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip() or f"Exit code {result.returncode}"}
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON output: {result.stdout[:500]}"}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except FileNotFoundError:
        return {"error": "clawquant not installed. Run: pip install clawquant"}
    except Exception as e:
        return {"error": str(e)}


def backtest_strategy(strategy: str, symbol: str = "BTC/USDT", days: int = 30, interval: str = "1h") -> str:
    """
    Run a backtest for a specific strategy on a symbol.

    Available strategies: dca, ma_crossover, macd, breakout, rsi_reversal, bollinger_bands, grid.

    Args:
        strategy: Strategy name (e.g. "ma_crossover", "breakout")
        symbol: Trading pair (e.g. "BTC/USDT")
        days: Number of days of historical data
        interval: Bar interval (e.g. "1h", "4h", "1d")

    Returns:
        JSON string with backtest results including return, sharpe, drawdown, win rate.
    """
    # First pull data
    _run_clawquant(["data", "pull", symbol, "--days", str(days), "--interval", interval], timeout=60)

    # Run backtest
    result = _run_clawquant(
        ["backtest", "run", strategy, "--symbols", symbol, "--days", str(days), "--interval", interval],
        timeout=120,
    )

    if "error" in result:
        return json.dumps(result)

    # Extract key metrics
    summary = {
        "strategy": strategy,
        "symbol": symbol,
        "days": days,
        "interval": interval,
        "total_return_pct": result.get("total_return_pct", 0),
        "sharpe_ratio": result.get("sharpe_ratio", 0),
        "max_drawdown_pct": result.get("max_drawdown_pct", 0),
        "win_rate": result.get("win_rate", 0),
        "total_trades": result.get("total_trades", 0),
        "profit_factor": result.get("profit_factor", 0),
        "stability_score": result.get("stability_score", 0),
        "score_breakdown": result.get("score_breakdown", {}),
    }

    return json.dumps(summary, indent=2)


def compare_strategies(strategies: str = "ma_crossover,macd,breakout,rsi_reversal", symbol: str = "BTC/USDT", days: int = 30) -> str:
    """
    Compare multiple strategies side-by-side via batch backtest.

    Args:
        strategies: Comma-separated strategy names
        symbol: Trading pair
        days: Number of days

    Returns:
        JSON comparison with key metrics for each strategy.
    """
    # Pull data
    _run_clawquant(["data", "pull", symbol, "--days", str(days)], timeout=60)

    # Batch backtest
    result = _run_clawquant(
        ["backtest", "batch", strategies, "--symbols", symbol, "--days", str(days)],
        timeout=300,
    )

    if "error" in result:
        return json.dumps(result)

    return json.dumps(result, indent=2)


def scan_opportunities(symbols: str = "BTC/USDT,ETH/USDT,SOL/USDT") -> str:
    """
    Scan symbols for current trading opportunities across all strategies.

    Args:
        symbols: Comma-separated symbols to scan

    Returns:
        JSON with detected signals and context.
    """
    result = _run_clawquant(
        ["radar", "scan", "--symbols", symbols],
        timeout=120,
    )

    if "error" in result:
        return json.dumps(result)

    return json.dumps(result, indent=2)
