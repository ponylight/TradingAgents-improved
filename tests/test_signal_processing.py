"""Tests for SignalProcessor BUY/SELL/HOLD extraction logic.

Since SignalProcessor uses an LLM, we mock the LLM to test the
post-processing validation logic that was added.
"""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.signal_processing import SignalProcessor


def _make_processor(llm_response: str) -> SignalProcessor:
    """Create a SignalProcessor with a mocked LLM that returns the given text."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_llm.invoke.return_value = mock_response
    return SignalProcessor(mock_llm)


class TestSignalExtraction:
    def test_clean_buy(self):
        p = _make_processor("BUY")
        assert p.process_signal("some signal text") == "BUY"

    def test_clean_sell(self):
        p = _make_processor("SELL")
        assert p.process_signal("some signal text") == "SELL"

    def test_clean_hold(self):
        p = _make_processor("HOLD")
        assert p.process_signal("some signal text") == "HOLD"

    def test_lowercase_normalized(self):
        p = _make_processor("buy")
        assert p.process_signal("some signal text") == "BUY"

    def test_mixed_case(self):
        p = _make_processor("Buy")
        assert p.process_signal("some signal text") == "BUY"

    def test_embedded_buy(self):
        p = _make_processor("I recommend BUY for this stock")
        assert p.process_signal("some signal text") == "BUY"

    def test_embedded_sell(self):
        p = _make_processor("The recommendation is to SELL")
        assert p.process_signal("some signal text") == "SELL"

    def test_with_whitespace(self):
        p = _make_processor("  BUY  ")
        assert p.process_signal("some signal text") == "BUY"

    def test_garbage_defaults_to_hold(self):
        p = _make_processor("I'm not sure what to do")
        assert p.process_signal("some signal text") == "HOLD"

    def test_empty_defaults_to_hold(self):
        p = _make_processor("")
        assert p.process_signal("some signal text") == "HOLD"
