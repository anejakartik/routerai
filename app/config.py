"""Policy configuration loading for routerai.

Loads `routes.yaml` (path overridable via ROUTERAI_ROUTES env var) into a typed
Policy object used by the router.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

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
    _path: str = field(default="", repr=False)

    def cost_per_1k(self, model: str) -> float:
        """Cost in USD per 1K tokens for a model (0.0 if unknown)."""
        return float(self.models.get(model, 0.0))

    def candidates(self, tier: str) -> List[str]:
        """Ordered candidate models for a tier, falling back to the default tier."""
        return self.tiers.get(tier, self.tiers[self.default_tier])

    def cheapest(self, tier: str) -> str:
        """The cheapest candidate model available for the given tier."""
        cands = self.candidates(tier)
        return min(cands, key=self.cost_per_1k)


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

    return Policy(
        baseline=raw.get("baseline", "gpt-4o"),
        quality_bar=float(raw.get("quality_bar", 0.5)),
        shadow_sample_rate=float(raw.get("shadow_sample_rate", 1.0)),
        models={k: float(v) for k, v in (raw.get("models") or {}).items()},
        tiers={k: list(v) for k, v in tiers.items()},
        default_tier=default_tier,
        _path=str(routes_path),
    )


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    """Cached policy singleton for the running app."""
    return load_policy()
