import json
import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient
from .validators import validate_model

# OpenClaw gateway config path
OPENCLAW_CONFIG = os.path.expanduser("~/.openclaw/openclaw.json")


def _get_openclaw_gateway():
    """Read OpenClaw gateway URL and auth token from config."""
    try:
        with open(OPENCLAW_CONFIG) as f:
            cfg = json.load(f)
        gw = cfg.get("gateway", {})
        port = gw.get("port", 18789)
        token = gw.get("auth", {}).get("token", "")
        return f"http://127.0.0.1:{port}/v1", token
    except Exception:
        return None, None


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models via OpenClaw gateway (OpenAI-compatible proxy)."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return ChatOpenAI routed through OpenClaw gateway."""
        gw_url, gw_token = _get_openclaw_gateway()

        if not gw_url or not gw_token:
            raise RuntimeError(
                "OpenClaw gateway not configured. Ensure ~/.openclaw/openclaw.json exists "
                "with gateway.port and gateway.auth.token, and chatCompletions endpoint is enabled."
            )

        # Route through OpenClaw gateway as OpenAI-compatible endpoint
        model_name = self.model
        if not model_name.startswith("anthropic/"):
            model_name = f"anthropic/{model_name}"

        llm_kwargs = {
            "model": model_name,
            "base_url": gw_url,
            "api_key": gw_token,
        }

        for key in ("timeout", "max_retries", "max_tokens", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
