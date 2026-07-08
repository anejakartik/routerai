"""Provider abstraction.

A `Provider` turns a chat request into a completion. v0.1 ships a deterministic
STUB provider so the gateway runs with NO API key. Real providers (OpenAI /
Anthropic) can drop in behind an env key; `get_provider()` selects based on
ROUTERAI_PROVIDER (default: "stub").
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), min 1."""
    return max(1, len(text) // 4)


def _last_user_prompt(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


class Provider(Protocol):
    name: str

    def complete(self, model: str, messages: List[Message]) -> Completion:
        ...


class StubProvider:
    """Deterministic mock provider — same input always yields the same output.

    Produces a plausible, prompt-aware completion without any network call so the
    whole gateway is runnable and testable offline.
    """

    name = "stub"

    def complete(self, model: str, messages: List[Message]) -> Completion:
        prompt = _last_user_prompt(messages)
        digest = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:8]
        text = (
            f"[stub:{model}] Acknowledged your request. "
            f"Here is a concise, helpful answer to: \"{prompt.strip()[:120]}\". "
            f"(deterministic id={digest})"
        )
        prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_estimate_tokens(text),
        )


class OpenAIProvider:
    """Placeholder real provider. Requires `openai` + OPENAI_API_KEY.

    Intentionally lazy-imported so the stub path never needs the dependency.
    """

    name = "openai"

    def complete(self, model: str, messages: List[Message]) -> Completion:  # pragma: no cover
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return Completion(
            text=choice,
            prompt_tokens=getattr(usage, "prompt_tokens", _estimate_tokens("".join(m.content for m in messages))),
            completion_tokens=getattr(usage, "completion_tokens", _estimate_tokens(choice)),
        )


def get_provider(name: str | None = None) -> Provider:
    """Select a provider. Defaults to the stub so the app runs key-free."""
    name = (name or os.getenv("ROUTERAI_PROVIDER") or "stub").lower()
    if name == "openai":
        return OpenAIProvider()
    return StubProvider()
