"""Tests for FinancialSituationMemory."""

from tradingagents.agents.utils.memory import (
    FinancialSituationMemory,
    STOP_WORDS,
    FINANCIAL_SYNONYMS,
)


class TestTokenization:
    def test_stop_words_removed(self):
        m = FinancialSituationMemory("test")
        tokens = m._tokenize("The stock is in a market with analysis data")
        for stop in ["the", "is", "in", "with"]:
            assert stop not in tokens

    def test_financial_synonyms_normalized(self):
        m = FinancialSituationMemory("test")

        tokens = m._tokenize("P/E ratio is high")
        assert "price_earnings" in tokens

        tokens = m._tokenize("RSI at 75")
        assert "relative_strength_index" in tokens

        tokens = m._tokenize("MACD crossover signal")
        assert "moving_average_convergence_divergence" in tokens

        tokens = m._tokenize("EPS growth")
        assert "earnings_per_share" in tokens

    def test_slash_and_hyphen_tokens_captured(self):
        m = FinancialSituationMemory("test")
        tokens = m._tokenize("debt-to-equity ratio D/E")
        assert "debt_equity" in tokens

    def test_single_char_tokens_removed(self):
        m = FinancialSituationMemory("test")
        tokens = m._tokenize("a b c hello")
        assert "hello" in tokens
        assert "a" not in tokens
        assert "b" not in tokens

    def test_case_insensitive(self):
        m = FinancialSituationMemory("test")
        tokens_lower = m._tokenize("rsi overbought")
        tokens_upper = m._tokenize("RSI OVERBOUGHT")
        assert tokens_lower == tokens_upper

    def test_bullish_bearish_normalization(self):
        m = FinancialSituationMemory("test")
        tokens = m._tokenize("bullish momentum with bearish divergence")
        assert "bull_signal" in tokens
        assert "bear_signal" in tokens


class TestMemoryOperations:
    def test_empty_memory_returns_empty(self):
        m = FinancialSituationMemory("test")
        results = m.get_memories("some query", n_matches=3)
        assert results == []

    def test_add_and_retrieve(self):
        m = FinancialSituationMemory("test")
        m.add_situations([
            ("High volatility with RSI overbought", "Reduce position size"),
            ("Low volatility consolidation pattern", "Wait for breakout"),
        ])
        results = m.get_memories("RSI showing overbought volatility", n_matches=1)
        assert len(results) == 1
        assert "recommendation" in results[0]
        assert "matched_situation" in results[0]
        assert "similarity_score" in results[0]

    def test_multiple_matches(self):
        m = FinancialSituationMemory("test")
        m.add_situations([
            ("Situation A", "Advice A"),
            ("Situation B", "Advice B"),
            ("Situation C", "Advice C"),
        ])
        results = m.get_memories("Some query", n_matches=2)
        assert len(results) == 2

    def test_n_matches_capped_at_corpus_size(self):
        m = FinancialSituationMemory("test")
        m.add_situations([("Only situation", "Only advice")])
        results = m.get_memories("query", n_matches=5)
        # Should return at most the number of documents
        assert len(results) <= 1

    def test_clear(self):
        m = FinancialSituationMemory("test")
        m.add_situations([("test situation", "test advice")])
        assert len(m.documents) == 1

        m.clear()
        assert len(m.documents) == 0
        assert len(m.recommendations) == 0
        assert len(m.timestamps) == 0
        assert m.bm25 is None
        assert m.get_memories("anything") == []

    def test_similarity_score_normalized_0_to_1(self):
        m = FinancialSituationMemory("test")
        m.add_situations([
            ("Strong RSI overbought with MACD bearish crossover", "Sell signal"),
            ("Low volatility with no clear direction", "Hold position"),
            ("Earnings beat expectations with revenue growth", "Consider buying"),
        ])
        results = m.get_memories("RSI overbought MACD crossover", n_matches=3)
        for r in results:
            assert 0 <= r["similarity_score"] <= 1.0

    def test_timestamps_recorded(self):
        m = FinancialSituationMemory("test")
        m.add_situations([("test", "advice")])
        assert len(m.timestamps) == 1
        assert m.timestamps[0] > 0


class TestStopWordsAndSynonyms:
    def test_stop_words_is_frozenset(self):
        assert isinstance(STOP_WORDS, frozenset)
        assert "the" in STOP_WORDS
        assert "market" in STOP_WORDS
        assert "analysis" in STOP_WORDS

    def test_financial_synonyms_is_dict(self):
        assert isinstance(FINANCIAL_SYNONYMS, dict)
        assert "pe" in FINANCIAL_SYNONYMS
        assert "rsi" in FINANCIAL_SYNONYMS
        assert FINANCIAL_SYNONYMS["pe"] == "price_earnings"
        assert FINANCIAL_SYNONYMS["p/e"] == "price_earnings"
