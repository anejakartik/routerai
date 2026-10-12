"""Policy configuration loading for routerai.

Loads `routes.yaml` (path overridable via ROUTERAI_ROUTES env var) into a typed
Policy object used by the router.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

DEFAULT_ROUTES_PATH = Path(__file__).resolve().parent.parent / "routes.yaml"


@dataclass
class Policy:
    baseline: str
    quality_bar: float
    shadow_sample_rate: float
    models: Dict[str, float]  # model name -> cost per 1K tokens
    tiers: Dict[str, List[str]]  # tier name -> ordered candidate models
    default_tier: str
    # model name -> provider that serves it. A model with no entry (legacy
    # float-only config) is treated as servable by any provider.
    model_providers: Dict[str, str] = field(default_factory=dict)
    _path: str = field(default="", repr=False)

    def cost_per_1k(self, model: str) -> float:
        """Cost in USD per 1K tokens for a model (0.0 if unknown)."""
        return float(self.models.get(model, 0.0))

    def candidates(self, tier: str) -> List[str]:
        """Ordered candidate models for a tier, falling back to the default tier."""
        return self.tiers.get(tier, self.tiers[self.default_tier])

    def provider_for(self, model: str) -> Optional[str]:
        """The provider that serves `model`, or None if the config doesn't say."""
        return self.model_providers.get(model)

    def cheapest(self, tier: str, providers: Optional[Iterable[str]] = None) -> str:
        """The cheapest candidate for the tier that an enabled provider can serve.

        `providers=None` considers every candidate (pure price ordering). With a
        provider set, candidates owned by some other provider are skipped, so the
        router never sends e.g. a stub or Claude model id to OpenAI.
        """
        cands = self.candidates(tier)
        if providers is not None:
            enabled = set(providers)
            cands = [
                m for m in cands
                if self.provider_for(m) is None or self.provider_for(m) in enabled
            ]
            if not cands:
                raise NoRouteError(
                    f"no candidate in tier '{tier}' is served by enabled "
                    f"provider(s) {sorted(enabled)}"
                )
        return min(cands, key=self.cost_per_1k)


class NoRouteError(ValueError):
    """No candidate model in a tier is servable by the enabled providers."""


def _parse_models(raw_models: dict) -> tuple[Dict[str, float], Dict[str, str]]:
    """Accept `name: cost` (legacy) or `name: {provider: ..., cost: ...}`."""
    costs: Dict[str, float] = {}
    owners: Dict[str, str] = {}
    for name, spec in (raw_models or {}).items():
        if isinstance(spec, dict):
            costs[name] = float(spec.get("cost", 0.0))
            if spec.get("provider"):
                owners[name] = str(spec["provider"]).lower()
        else:
            costs[name] = float(spec)
    return costs, owners


def load_policy(path: str | os.PathLike | None = None) -> Policy:
    """Load a Policy from a YAML file. Raises on missing required keys."""
    routes_path = Path(path or os.getenv("ROUTERAI_ROUTES") or DEFAULT_ROUTES_PATH)
    with open(routes_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    tiers = raw.get("tiers") or {}
    if not tiers:
        raise ValueError(f"routes config {routes_path} defines no tiers")

    default_tier = raw.get("default_tier") or next(iter(tiers))
    if default_tier not in tiers:
        raise ValueError(f"default_tier '{default_tier}' is not a defined tier")

    models, model_providers = _parse_models(raw.get("models") or {})

    return Policy(
        baseline=raw.get("baseline", "gpt-4o"),
        quality_bar=float(raw.get("quality_bar", 0.5)),
        shadow_sample_rate=float(raw.get("shadow_sample_rate", 1.0)),
        models=models,
        tiers={k: list(v) for k, v in tiers.items()},
        default_tier=default_tier,
        model_providers=model_providers,
        _path=str(routes_path),
    )


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    """Cached policy singleton for the running app."""
    return load_policy()
