"""Shadow eval hook.

A pluggable `evaluate(prompt, response) -> score` in [0.0, 1.0]. v0.1 uses a
simple heuristic judge (a stand-in for an evalstack judge). Swap `evaluate` for
an LLM-as-judge or evalstack call in a later version — the router only depends
on the signature.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+")


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 3}


def evaluate(prompt: str, response: str) -> float:
    """Heuristic quality score in [0.0, 1.0].

    Rewards: non-trivial length, keyword overlap with the prompt (relevance),
    and absence of obvious failure markers. Deterministic and dependency-free.
    """
    if not response or not response.strip():
        return 0.0

    resp = response.strip()

    # Length signal — very short answers score poorly, saturating around 40 words.
    n_words = len(resp.split())
    length_score = min(n_words / 40.0, 1.0)

    # Relevance signal — overlap between prompt keywords and response keywords.
    p_kw = _keywords(prompt)
    r_kw = _keywords(resp)
    if p_kw:
        relevance = len(p_kw & r_kw) / len(p_kw)
    else:
        relevance = 0.5

    # Failure markers drag the score down.
    lowered = resp.lower()
    penalty = 0.0
    for marker in ("i cannot", "i can't", "error", "as an ai", "i'm unable"):
        if marker in lowered:
            penalty += 0.3

    score = 0.5 * length_score + 0.5 * relevance - penalty
    return max(0.0, min(1.0, round(score, 4)))
