"""Financial situation memory using BM25 for lexical similarity matching.

Uses BM25 (Best Matching 25) algorithm for retrieval - no API calls,
no token limits, works offline with any LLM provider.
Enhanced with financial term normalization, stop-word removal, and recency weighting.
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple
import re
import time
import math


# Common stop words that add no signal for BM25 matching in financial contexts
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "and", "or", "if", "while", "that", "this", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "up", "down", "about", "because", "also", "report", "data",
    "analysis", "based", "using", "shows", "company", "stock", "market",
})

# Financial term synonyms for normalization
FINANCIAL_SYNONYMS = {
    "pe": "price_earnings", "p/e": "price_earnings", "price-to-earnings": "price_earnings",
    "pb": "price_book", "p/b": "price_book", "price-to-book": "price_book",
    "eps": "earnings_per_share",
    "roe": "return_on_equity", "roa": "return_on_assets",
    "rsi": "relative_strength_index",
    "macd": "moving_average_convergence_divergence",
    "sma": "simple_moving_average", "ema": "exponential_moving_average",
    "atr": "average_true_range", "mfi": "money_flow_index",
    "fcf": "free_cash_flow",
    "d/e": "debt_equity", "debt-to-equity": "debt_equity",
    "bullish": "bull_signal", "bearish": "bear_signal",
    "overbought": "overbought_signal", "oversold": "oversold_signal",
    "breakout": "breakout_signal", "breakdown": "breakdown_signal",
    "uptrend": "trend_up", "downtrend": "trend_down",
    "buy": "buy_signal", "sell": "sell_signal", "hold": "hold_signal",
}


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situations using BM25.

    Enhanced with financial term normalization, stop-word removal, and recency weighting.
    """

    # Half-life for recency decay in days (score halves every 30 days)
    RECENCY_HALF_LIFE_DAYS = 30

    def __init__(self, name: str, config: dict = None):
        """Initialize the memory system.

        Args:
            name: Name identifier for this memory instance
            config: Configuration dict (kept for API compatibility)
        """
        self.name = name
        self.documents: List[str] = []
        self.recommendations: List[str] = []
        self.timestamps: List[float] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing with financial awareness.

        - Captures tokens with slashes and hyphens (P/E, debt-to-equity)
        - Normalizes financial abbreviations to canonical forms
        - Removes common stop words
        """
        tokens = re.findall(r'\b[\w/\-]+\b', text.lower())
        processed = []
        for token in tokens:
            normalized = FINANCIAL_SYNONYMS.get(token, token)
            if normalized not in STOP_WORDS and len(normalized) > 1:
                processed.append(normalized)
        return processed

    def _rebuild_index(self):
        """Rebuild the BM25 index after adding documents."""
        if self.documents:
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add financial situations and their corresponding advice.

        Args:
            situations_and_advice: List of tuples (situation, recommendation)
        """
        for situation, recommendation in situations_and_advice:
            self.documents.append(situation)
            self.recommendations.append(recommendation)
            self.timestamps.append(time.time())

        # Rebuild BM25 index with new documents
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 1) -> List[dict]:
        """Find matching recommendations using BM25 similarity with recency weighting.

        Args:
            current_situation: The current financial situation to match against
            n_matches: Number of top matches to return

        Returns:
            List of dicts with matched_situation, recommendation, and similarity_score
        """
        if not self.documents or self.bm25 is None:
            return []

        # Tokenize query
        query_tokens = self._tokenize(current_situation)

        # Get BM25 scores for all documents
        scores = list(self.bm25.get_scores(query_tokens))

        # Apply recency weighting: 70% BM25 relevance + 30% recency boost
        now = time.time()
        half_life_seconds = self.RECENCY_HALF_LIFE_DAYS * 86400
        for i in range(len(scores)):
            age_seconds = now - self.timestamps[i]
            recency_factor = math.exp(-0.693 * age_seconds / half_life_seconds)
            scores[i] *= (0.7 + 0.3 * recency_factor)

        # Get top-n indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_matches]

        # Build results
        max_score = max(scores) if max(scores) > 0 else 1

        results = []
        for idx in top_indices:
            normalized_score = scores[idx] / max_score if max_score > 0 else 0
            results.append({
                "matched_situation": self.documents[idx],
                "recommendation": self.recommendations[idx],
                "similarity_score": normalized_score,
            })

        return results

    def clear(self):
        """Clear all stored memories."""
        self.documents = []
        self.recommendations = []
        self.timestamps = []
        self.bm25 = None

    def save(self, path: str):
        """Persist memory to disk as JSON."""
        import json
        data = {
            "name": self.name,
            "documents": self.documents,
            "recommendations": self.recommendations,
            "timestamps": self.timestamps,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load memory from disk. Returns True if loaded, False if file missing."""
        import json, os
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.documents = data.get("documents", [])
        self.recommendations = data.get("recommendations", [])
        self.timestamps = data.get("timestamps", [])
        self._rebuild_index()
        return True


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory("test_memory")

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
