"""
LLM Fallback Wrapper — switches to backup provider on rate limit (429) errors.

Usage:
    primary = anthropic_client.get_llm()
    fallback = openrouter_client.get_llm()
    llm = FallbackLLM(primary, fallback, fallback_label="openrouter/minimax-m2.5")
"""

import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

log = logging.getLogger("llm_fallback")


def _is_rate_limit(exc: Exception) -> bool:
    """Check if exception is a rate limit error (HTTP 429 or similar)."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True
    if "overloaded" in msg or "capacity" in msg:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return True
    return False


class FallbackLLM(BaseChatModel):
    """Wraps a primary LLM and falls back to a secondary on rate limit errors."""

    primary: Any
    fallback: Any
    fallback_label: str = "fallback"

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fallback"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        try:
            return self.primary._generate(messages, stop=stop, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                log.warning(f"⚡ Rate limited on primary, falling back to {self.fallback_label}: {e}")
                return self.fallback._generate(messages, stop=stop, **kwargs)
            raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        try:
            return await self.primary._agenerate(messages, stop=stop, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                log.warning(f"⚡ Rate limited on primary, falling back to {self.fallback_label}: {e}")
                return await self.fallback._agenerate(messages, stop=stop, **kwargs)
            raise
