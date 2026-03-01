"""LLM call telemetry - logs timing, prompt size, response size."""

import time
import logging

log = logging.getLogger("telemetry")


class TelemetryWrapper:
    """Wraps a ChatModel to log timing and token estimates on every invoke."""

    def __init__(self, llm, label="LLM"):
        self._llm = llm
        self._label = label

    def invoke(self, *args, **kwargs):
        start = time.time()
        input_str = str(args[0] if args else kwargs.get("input", ""))
        input_chars = len(input_str)

        result = self._llm.invoke(*args, **kwargs)

        elapsed = time.time() - start
        output_chars = len(result.content) if hasattr(result, "content") else 0

        log.info(
            f"[{self._label}] {elapsed:.1f}s | "
            f"in: ~{input_chars//4:,}tok | out: ~{output_chars//4:,}tok"
        )
        return result

    def bind_tools(self, *args, **kwargs):
        bound = self._llm.bind_tools(*args, **kwargs)
        return TelemetryWrapper(bound, self._label)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def wrap_with_telemetry(llm, label="LLM"):
    return TelemetryWrapper(llm, label)
