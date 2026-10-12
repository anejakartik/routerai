"""OpenAI-compatible request/response schemas (subset for v0.1)."""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model or tier hint. Accepts 'cheap'/'balanced'/'premium' as a routing tier, or any model name (routerai will still route by tier).")
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    # routerai extension: explicit quality tier (overrides model-derived hint)
    quality_tier: Optional[str] = Field(default=None, description="routerai extension: cheap | balanced | premium")


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RouterAIMeta(BaseModel):
    """routerai-specific metadata attached to each response (non-breaking)."""
    request_id: str
    tier: str
    chosen_model: str
    est_cost_usd: float
    baseline_cost_usd: float
    est_savings_usd: float
    eval_score: Optional[float] = None
    passed_quality_bar: Optional[bool] = None
    # the routed provider failed and the offline stub answered instead
    fallback: bool = False


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage
    routerai: RouterAIMeta
