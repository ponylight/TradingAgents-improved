"""
LLM Fallback Wrapper — switches to backup provider on rate limit (429) errors.

Wraps primary + fallback LLMs. On 429/rate-limit/overloaded from primary,
transparently retries with fallback. Supports bind_tools for LangChain agents.
"""

import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig

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

    @property
    def _identifying_params(self):
        return {"primary": str(self.primary), "fallback": self.fallback_label}

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

    def bind_tools(self, *args, **kwargs):
        """Delegate bind_tools — returns a FallbackRunnable wrapping both bound results."""
        primary_bound = self.primary.bind_tools(*args, **kwargs)
        fallback_bound = self.fallback.bind_tools(*args, **kwargs)
        return FallbackRunnable(
            primary=primary_bound,
            fallback=fallback_bound,
            fallback_label=self.fallback_label,
        )

    def bind(self, *args, **kwargs):
        """Delegate bind — returns a FallbackRunnable wrapping both bound results."""
        primary_bound = self.primary.bind(*args, **kwargs)
        fallback_bound = self.fallback.bind(*args, **kwargs)
        return FallbackRunnable(
            primary=primary_bound,
            fallback=fallback_bound,
            fallback_label=self.fallback_label,
        )


class FallbackRunnable(Runnable):
    """Wraps two bound Runnables (e.g. from bind_tools) with rate-limit fallback."""

    def __init__(self, primary, fallback, fallback_label="fallback"):
        self._primary = primary
        self._fallback = fallback
        self._fallback_label = fallback_label

    def invoke(self, input, config: Optional[RunnableConfig] = None, **kwargs):
        try:
            return self._primary.invoke(input, config=config, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                log.warning(f"⚡ Rate limited on primary, falling back to {self._fallback_label}: {e}")
                return self._fallback.invoke(input, config=config, **kwargs)
            raise

    async def ainvoke(self, input, config: Optional[RunnableConfig] = None, **kwargs):
        try:
            return await self._primary.ainvoke(input, config=config, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                log.warning(f"⚡ Rate limited on primary, falling back to {self._fallback_label}: {e}")
                return await self._fallback.ainvoke(input, config=config, **kwargs)
            raise

    def batch(self, inputs, config=None, **kwargs):
        try:
            return self._primary.batch(inputs, config=config, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                log.warning(f"⚡ Rate limited on primary, falling back to {self._fallback_label}: {e}")
                return self._fallback.batch(inputs, config=config, **kwargs)
            raise

    # Pass through attributes to primary (for LangGraph compatibility)
    def __getattr__(self, name):
        return getattr(self._primary, name)
