"""FastAPI app — the OpenAI-compatible gateway surface."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from . import __version__
from .metrics import STORE
from .observability import get_sink
from .config import NoRouteError
from .providers import Message, get_providers
from .routing import get_db, route
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
    RouterAIMeta,
    Usage,
)

logger = logging.getLogger("routerai.main")

app = FastAPI(
    title="routerai",
    version=__version__,
    description="Cost-aware, quality-gated LLM routing gateway (govern control-plane).",
)


@app.on_event("startup")
def _reload_persisted_metrics() -> None:
    """Warm the in-memory metrics view from the durable SQLite log on startup."""
    try:
        db = get_db()
        persisted = db.load_all()
        if persisted:
            STORE.reset()
            for d in persisted:
                STORE.record(d)
            logger.info("reloaded %d persisted decision(s) into metrics", len(persisted))
    except Exception as err:  # noqa: BLE001 - never block startup on the log
        logger.warning("could not reload persisted metrics: %s", err)


@app.get("/")
def root():
    return {
        "name": "routerai",
        "version": __version__,
        "providers": sorted(get_providers()),
        "event_sink": getattr(get_sink(), "name", "unknown"),
        "endpoints": ["/v1/chat/completions", "/metrics", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    messages = [Message(role=m.role, content=m.content) for m in req.messages]
    # a model hint like "cheap"/"balanced"/"premium" is honored as a tier
    tier_hint = req.quality_tier or req.model

    try:
        result = route(messages, quality_tier=tier_hint)
    except NoRouteError as err:
        # routes.yaml has no model in this tier for the enabled providers
        raise HTTPException(status_code=503, detail=str(err)) from err

    return ChatCompletionResponse(
        model=result.chosen_model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=result.completion.text),
            )
        ],
        usage=Usage(
            prompt_tokens=result.completion.prompt_tokens,
            completion_tokens=result.completion.completion_tokens,
            total_tokens=result.completion.total_tokens,
        ),
        routerai=RouterAIMeta(
            request_id=result.request_id,
            tier=result.tier,
            chosen_model=result.chosen_model,
            est_cost_usd=result.est_cost,
            baseline_cost_usd=result.baseline_cost,
            est_savings_usd=round(result.baseline_cost - result.est_cost, 8),
            eval_score=result.eval_score,
            passed_quality_bar=result.passed_quality_bar,
            fallback=result.fallback,
        ),
    )


@app.get("/metrics")
def metrics():
    return STORE.summary()
