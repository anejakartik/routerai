"""Provider abstraction.

A `Provider` turns a chat request into a completion. routerai ships a
deterministic STUB provider so the gateway runs with NO API key. Real providers
(OpenAI / Anthropic) drop in behind an env key; `get_provider()` selects based on
ROUTERAI_PROVIDER (default: "stub").

v0.2: the real adapters are wired end-to-end (import-guarded so the stub path
never needs the dependency). Each real adapter enforces a request **timeout**,
does **one retry**, and on any failure **falls back gracefully to the stub** so a
provider outage degrades to offline mode instead of a 500.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger("routerai.providers")

# defaults; all overridable via env so nothing is hardcoded to a secret.
DEFAULT_TIMEOUT_S = float(os.getenv("ROUTERAI_PROVIDER_TIMEOUT", "30"))
DEFAULT_RETRIES = int(os.getenv("ROUTERAI_PROVIDER_RETRIES", "1"))


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    # which provider actually produced this (may differ from the configured one
    # if we fell back to the stub on error).
    provider: str = "stub"

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
            provider=self.name,
        )


# shared stub instance used for fallback
_STUB = StubProvider()


class _RealProvider:
    """Base for real network providers: timeout + one retry + stub fallback.

    Subclasses implement `_call(model, messages) -> Completion`. The base handles
    retries and the graceful degrade so a provider hiccup never 500s the gateway.
    """

    name = "real"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_S, retries: int = DEFAULT_RETRIES):
        self.timeout = timeout
        self.retries = max(0, retries)

    def _call(self, model: str, messages: List[Message]) -> Completion:  # pragma: no cover
        raise NotImplementedError

    def complete(self, model: str, messages: List[Message]) -> Completion:
        attempts = self.retries + 1
        last_err: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._call(model, messages)
            except Exception as err:  # noqa: BLE001 - degrade on any provider error
                last_err = err
                logger.warning(
                    "provider %s attempt %d/%d failed: %s",
                    self.name, attempt, attempts, err,
                )
        logger.error(
            "provider %s exhausted %d attempt(s); falling back to stub (last error: %s)",
            self.name, attempts, last_err,
        )
        # graceful fallback: still serve a deterministic answer, tagged as stub.
        return _STUB.complete(model, messages)


class OpenAIProvider(_RealProvider):
    """Real OpenAI adapter. Requires `openai` + OPENAI_API_KEY (read from env).

    Lazy-imports `openai` so the stub path never needs the dependency.
    """

    name = "openai"

    def _call(self, model: str, messages: List[Message]) -> Completion:
        from openai import OpenAI  # type: ignore  # lazy import

        # OpenAI() reads OPENAI_API_KEY from the environment; never hardcoded.
        client = OpenAI(timeout=self.timeout, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens if prompt_tokens is not None
            else sum(_estimate_tokens(m.content) for m in messages),
            completion_tokens=completion_tokens if completion_tokens is not None
            else _estimate_tokens(text),
            provider=self.name,
        )


class AnthropicProvider(_RealProvider):
    """Real Anthropic adapter. Requires `anthropic` + ANTHROPIC_API_KEY (from env).

    Lazy-imports `anthropic`. The routed model id is forwarded as-is; override the
    fallback default via ROUTERAI_ANTHROPIC_MODEL if a route emits a non-Anthropic
    id. max_tokens is bounded (ROUTERAI_ANTHROPIC_MAX_TOKENS, default 1024).
    """

    name = "anthropic"

    def _call(self, model: str, messages: List[Message]) -> Completion:
        import anthropic  # type: ignore  # lazy import

        # Anthropic() reads ANTHROPIC_API_KEY from the environment; never hardcoded.
        client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)

        # Anthropic wants a system string separate from the message turns.
        system = "\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        max_tokens = int(os.getenv("ROUTERAI_ANTHROPIC_MAX_TOKENS", "1024"))
        model_id = model or os.getenv("ROUTERAI_ANTHROPIC_MODEL", "claude-opus-5-5")

        kwargs = {"model": model_id, "max_tokens": max_tokens, "messages": turns}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)

        # concatenate any text blocks in the response content
        text = "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens if prompt_tokens is not None
            else sum(_estimate_tokens(m.content) for m in messages),
            completion_tokens=completion_tokens if completion_tokens is not None
            else _estimate_tokens(text),
            provider=self.name,
        )


_PROVIDERS = {
    "stub": StubProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str | None = None) -> Provider:
    """Select a provider. Defaults to the stub so the app runs key-free."""
    name = (name or os.getenv("ROUTERAI_PROVIDER") or "stub").lower()
    factory = _PROVIDERS.get(name)
    if factory is None:
        logger.warning("unknown ROUTERAI_PROVIDER=%r; using stub", name)
        return _STUB
    return factory()


def get_providers(spec: str | None = None) -> Dict[str, Provider]:
    """Enabled providers keyed by name, from a comma list (ROUTERAI_PROVIDER).

    `ROUTERAI_PROVIDER=openai,anthropic` enables both; the router then picks the
    cheapest candidate either one can serve and dispatches to its owner.
    Unknown names are skipped with a warning; an empty result means stub.
    """
    spec = spec or os.getenv("ROUTERAI_PROVIDER") or "stub"
    enabled: Dict[str, Provider] = {}
    for name in (part.strip().lower() for part in spec.split(",")):
        if not name:
            continue
        factory = _PROVIDERS.get(name)
        if factory is None:
            logger.warning("unknown provider %r in ROUTERAI_PROVIDER; skipping", name)
            continue
        enabled[name] = factory()
    return enabled or {"stub": _STUB}
