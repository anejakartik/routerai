# routerai

**Cost-aware, quality-gated LLM routing gateway.** Most of your LLM queries don't
need a GPT-class model. `routerai` is a drop-in, OpenAI-compatible gateway that
auto-routes each request to the **cheapest model that still clears your quality
bar** — and proves the savings with shadow evals.

> **The LLM platform trilogy — gate -> observe -> govern:**
> [**evalstack**](https://github.com/anejakartik/evalstack) proves quality (the *gate*),
> [**tracelens**](https://github.com/anejakartik/tracelens) measures cost/latency (the *observe* layer),
> and **routerai** is the **govern** control-plane that *acts* on both at request time.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### One-command demo (no API key, no server)

```bash
python demo.py
```

Runs several sample prompts through the router against a deterministic **stub
provider**, printing the tier chosen, the model picked, per-request cost, the
shadow eval score, and the aggregate estimated savings vs always using the
premium model.

### Run the server

```bash
uvicorn app.main:app --port 8000
# then, in another shell:
./scripts/demo_curl.sh
```

### OpenAI-compatible request

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"cheap","messages":[{"role":"user","content":"classify: I love it"}]}'
```

The `model` field doubles as a routing hint — pass `cheap`, `balanced`, or
`premium` (or use the `quality_tier` extension field). The response is standard
OpenAI shape plus a `routerai` metadata block (tier, chosen model, cost, savings,
eval score).

---

## How it works

1. **Proxy** — `POST /v1/chat/completions` accepts the standard OpenAI request.
2. **Routing** — a request is assigned a **quality tier** (explicit hint, or a
   heuristic complexity classifier), then the router picks the **cheapest
   candidate model that an enabled provider serves** for that tier from
   `routes.yaml` (v0.3: every model declares its provider).
3. **Providers** — a pluggable abstraction. Ships with a **stub** provider so it
   runs offline with no key; set `ROUTERAI_PROVIDER=openai` (+ `OPENAI_API_KEY`),
   `anthropic` (+ `ANTHROPIC_API_KEY`), or `openai,anthropic` to route across both.
   Real adapters enforce a timeout, retry once, and **fall back to the stub** on
   error so a provider outage degrades to offline mode instead of a 500 (v0.2).
4. **Shadow eval** — a pluggable `evaluate(prompt, response) -> score` judge
   scores each response and flags whether the cheap route cleared the
   `quality_bar` (v0.1 uses a heuristic judge; swap in an evalstack/LLM judge).
5. **Persistence** — every decision is appended to **SQLite** (`routerai.db`) and
   mirrored in an in-memory view; on startup the aggregate metrics are reloaded
   from the log so `/metrics` is continuous across restarts (v0.2).
6. **Observability emit** — one **tracelens-style trace event** per request
   (model, latency_ms, tokens, cost, eval_score, passed) to a pluggable sink;
   default appends JSONL, a webhook sink can feed a live **tracelens** (v0.2).
7. **Metrics** — `GET /metrics` returns total requests, tokens, est cost,
   **estimated savings vs always-premium**, avg eval score, and quality-bar pass
   rate.

### Configuration — `routes.yaml`

Maps quality tiers to candidate models, each with a provider and a per-1K-token
cost (`gpt-4o-mini: {provider: openai, cost: 0.0003}`; the legacy `name: cost`
form still works and means "any provider"), sets the
`baseline` (always-premium) model for savings math, the `quality_bar`, and the
`shadow_sample_rate`. Point `ROUTERAI_ROUTES` at your own file to override.

---

## What's new in v0.3 — routing that knows who serves what

v0.2 shipped real OpenAI and Anthropic adapters, but the router picked the
cheapest model in a tier with no idea which provider served it. With the shipped
`routes.yaml` the cheapest candidate in every tier was a stub model, so
`ROUTERAI_PROVIDER=openai` sent `model="stub-small"` to OpenAI on every request.
It 404'd, retried, fell back to the stub, and the request was still billed at the
routed price, counted as a saving, and scored as a quality pass. In other words,
the real providers never answered a request, and an outage looked like a 100%
pass rate.

- **Provider-aware routing.** Each model in `routes.yaml` names its provider. The
  router picks the cheapest candidate an *enabled* provider can serve and
  dispatches to that provider. `ROUTERAI_PROVIDER=openai,anthropic` routes across
  both. If no candidate in a tier is servable, the API returns `503` with the
  reason instead of guessing.
- **Honest fallbacks.** When a real provider fails and the stub answers, the
  decision is flagged `fallback: true` (in the response, the trace event, the
  SQLite log and `/metrics`). It is excluded from cost, savings and pass rate, and
  `/metrics` reports `fallback_count` / `fallback_rate`. The CI gate takes
  `--max-fallback-rate` (default `0`).
- **Current model ids.** Claude entries updated to `claude-haiku-4-5`,
  `claude-sonnet-5-5` and `claude-opus-5-5`.
- Shadow-eval sampling now hashes the request id with SHA-256. Python's built-in
  `hash()` is salted per process, so the sampled subset changed on every restart.

Existing `routerai.db` files are migrated in place (a `fallback` column is added
on startup).

---

## What's new in v0.2 — closing the *gate → observe → govern* loop

routerai is the **govern** plane; v0.2 wires it directly to the other two legs so
a routing change is *proven* (gate), *measured* (observe), and *acted on* (govern)
end to end.

### 1. Real provider adapters (stub stays the default)

| `ROUTERAI_PROVIDER` | Needs | Notes |
| --- | --- | --- |
| `stub` *(default)* | nothing | deterministic, offline, key-free |
| `openai` | `openai` lib + `OPENAI_API_KEY` | keys read from env, never hardcoded |
| `anthropic` | `anthropic` lib + `ANTHROPIC_API_KEY` | keys read from env, never hardcoded |

Real adapters are **import-guarded** (the lib is only imported when selected),
enforce a **timeout** (`ROUTERAI_PROVIDER_TIMEOUT`), do **one retry**
(`ROUTERAI_PROVIDER_RETRIES`), and **gracefully fall back to the stub** on any
error — the completion is then tagged `provider="stub"` so the trace tells the truth.

### 2. Persistence (SQLite + in-memory, reload on startup)

Every routing decision (model, tokens, cost, eval score, pass/fail, timestamp) is
appended to `routerai.db` **and** kept in the in-memory metrics view. On startup
the app **reloads the aggregate from the log**, so `/metrics` survives restarts.
Override the path with `ROUTERAI_DB`.

### 3. Observability emit — feeds **tracelens**

One structured **trace event** per request, matching the tracelens trace schema:

```json
{"request_id":"req-ab12cd34","ts":1751328000.0,"tier":"cheap","model":"stub-small",
 "provider":"stub","latency_ms":0.42,"prompt_tokens":6,"completion_tokens":34,
 "total_tokens":40,"cost":8e-06,"baseline_cost":0.0002,"eval_score":0.71,"passed":true}
```

The sink is pluggable via `ROUTERAI_SINK`:

- `jsonl` *(default)* — append one JSON line to `events.jsonl`
  (`ROUTERAI_EVENTS_PATH`); tail-able, trivially ingestible.
- `webhook` — POST each event to `ROUTERAI_WEBHOOK_URL` (feed a live **tracelens**).
- `null` — disable emit.

A sink failure never breaks the request path.

### 4. CI policy-gate (mirrors **evalstack**)

`.github/workflows/eval-gate.yml` runs the test suite then the eval gate, and
**fails the build** if the quality-bar pass-rate or estimated savings regresses
past a threshold — so a policy change that quietly degrades quality/savings can't
merge:

```bash
python scripts/eval_gate.py --min-pass-rate 0.8 --min-savings-pct 20 --max-fallback-rate 0 --json
```

Thresholds also read from `ROUTERAI_MIN_PASS_RATE` / `ROUTERAI_MIN_SAVINGS_PCT`.
The gate runs against the deterministic stub, so it's stable and key-free in CI.

---

## Tests

```bash
pytest -q
```

---

## Roadmap (v0.3+)

- Fine-tuned complexity classifier trained on the persisted decision log.
- Semantic caching (return cached answers for near-duplicate prompts).
- Postgres persistence backend (SQLite is the default today).
- Next.js playground + savings dashboard at `routerai.kartikaneja.com`.
- Latency-SLO-aware routing; per-tenant policies.

## License

MIT — see [LICENSE](LICENSE).
