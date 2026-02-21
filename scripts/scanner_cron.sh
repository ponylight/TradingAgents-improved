#!/usr/bin/env bash
# scanner_cron.sh — V20 Signal Scanner wrapper for cron
#
# Schedule (every 4 hours, offset by 5 min to let candle close):
#   5 */4 * * * /Users/michaelhuang/TradingAgents-improved/scripts/scanner_cron.sh
#
# Runs live_scanner.py, logs output to logs/scanner_YYYY-MM-DD.log

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_ACTIVATE="$HOME/trading-agents-env/bin/activate"
LOG_DIR="$PROJECT_DIR/logs"

# ── Create log dir if needed ───────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Log file: scanner_YYYY-MM-DD.log ──────────────────────────────
DATE_STR="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/scanner_${DATE_STR}.log"

# ── Timestamp banner ───────────────────────────────────────────────
{
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  V20 Scanner run: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "════════════════════════════════════════════════════════════"
} >> "$LOG_FILE" 2>&1

# ── Activate venv and run scanner ─────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

cd "$PROJECT_DIR"

python scripts/live_scanner.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "  ⚠️  Scanner exited with code $EXIT_CODE" >> "$LOG_FILE"
fi

echo "  Done: $(date '+%H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
