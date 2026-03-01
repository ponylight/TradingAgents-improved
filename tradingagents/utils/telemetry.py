"""LLM call telemetry - logs timing, prompt size, response size."""

import time
import logging
from langchain_core.runnables import RunnableSerializable

log = logging.getLogger("telemetry")


class TelemetryWrapper(RunnableSerializable):
    """Wraps a ChatModel to log timing and token estimates. LangChain Runnable-compatible."""

    class Config:
        arbitrary_types_allowed = True

    llm: object
    label: str = "LLM"

    def invoke(self, input, config=None, **kwargs):
        start = time.time()
        input_chars = len(str(input))

        result = self.llm.invoke(input, config=config, **kwargs)

        elapsed = time.time() - start
        output_chars = len(result.content) if hasattr(result, "content") else 0

        log.info(
            f"[{self.label}] {elapsed:.1f}s | "
            f"in: ~{input_chars//4:,}tok | out: ~{output_chars//4:,}tok"
        )
        return result

    def bind_tools(self, *args, **kwargs):
        bound = self.llm.bind_tools(*args, **kwargs)
        return TelemetryWrapper(llm=bound, label=self.label)

    def __getattr__(self, name):
        return getattr(self.llm, name)


def wrap_with_telemetry(llm, label="LLM"):
    return TelemetryWrapper(llm=llm, label=label)
