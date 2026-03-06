## Position Scaling (Future)

**Status:** Deferred — revisit when stability score > 70 and 20+ trades tracked.

**Current:** One position at a time (committee) + optional green lane side position.

**Future design when ready:**
- Add-on-pullback: if confidence >= 8/10 and price retraces to EMA/VWAP, add 30%
- Trader agent needs structured output: ADD_TO_LONG/ADD_TO_SHORT
- Risk manager must enforce combined margin limits
- Blended entry price tracking for SL/TP
- Prerequisite: proven base system with consistent win rate

**Why not now:** Fix fundamentals first, then layer complexity.
