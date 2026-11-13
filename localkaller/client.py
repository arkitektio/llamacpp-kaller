"""The OpenAI client kaller talks to, pointed at the local llama-server.

kaller calls ``client.chat.completions.create(model=, messages=, tools=, stream=True)``
and nothing else. Two things a hosted model forgives and a local chat template may
not are adjusted on the way:

- **A system message after the first.** kaller's last round appends "Stop calling
  tools…" as a system message. DeepSeek's template accepts that; some templates
  (Qwen's among them) refuse any system message that is not first. With
  ``fold_system`` such a message is sent as a user message marked as an instruction.
- **No token limit.** A reasoning model can think until the context runs out. Every
  completion is given ``max_tokens`` unless kaller set one.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

INSTRUCTION_PREFIX = "[Instruction from the system] "


def fold_late_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` with every system message after the first turned into a user instruction."""
    folded: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "system" and index > 0:
            folded.append({"role": "user", "content": INSTRUCTION_PREFIX + str(message.get("content") or "")})
        else:
            folded.append(message)
    return folded


class _Completions:
    def __init__(self, inner: Any, max_tokens: int, fold_system: bool) -> None:
        self._inner = inner
        self._max_tokens = max_tokens
        self._fold_system = fold_system

    async def create(self, **kwargs: Any) -> Any:
        if self._fold_system and "messages" in kwargs:
            kwargs["messages"] = fold_late_system_messages(list(kwargs["messages"]))
        kwargs.setdefault("max_tokens", self._max_tokens)
        return await self._inner.create(**kwargs)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class LocalClient:
    """Enough of ``AsyncOpenAI`` for kaller: ``chat.completions.create``, adjusted as above."""

    def __init__(self, base_url: str, *, max_tokens: int = 8192, fold_system: bool = True, api_key: str = "local") -> None:
        self.openai = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.chat = _Chat(_Completions(self.openai.chat.completions, max_tokens, fold_system))
