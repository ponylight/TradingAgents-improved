# TradingAgents/graph/signal_processing.py

import re


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm):
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Extract LONG/SHORT/NEUTRAL or BUY/SELL/HOLD from signal text."""
        content = full_signal.upper()

        # Try deterministic extraction first (avoid LLM call)
        # Trading mode: LONG/SHORT/NEUTRAL
        for action in ["LONG", "SHORT", "NEUTRAL"]:
            if f"FINAL TRANSACTION PROPOSAL: **{action}**" in content:
                return action

        # Investment mode: BUY/SELL/HOLD
        for action in ["BUY", "SELL", "HOLD"]:
            if f"FINAL TRANSACTION PROPOSAL: **{action}**" in content:
                return action

        # Fallback: check last 200 chars with word boundaries
        tail = content[-200:]
        for action in ["LONG", "SHORT", "NEUTRAL", "BUY", "SELL", "HOLD"]:
            if re.search(rf"\b{action}\b", tail):
                return action

        # LLM fallback
        messages = [
            (
                "system",
                "Extract the investment decision from the text. "
                "Output ONLY one word: LONG, SHORT, NEUTRAL, BUY, SELL, or HOLD.",
            ),
            ("human", full_signal),
        ]

        raw = self.quick_thinking_llm.invoke(messages).content.strip().upper()

        valid = {"LONG", "SHORT", "NEUTRAL", "BUY", "SELL", "HOLD"}
        if raw in valid:
            return raw
        for signal in valid:
            if signal in raw:
                return signal

        return "NEUTRAL"
