# routerai

**Cost-aware, quality-gated LLM routing gateway.** Most of your LLM queries don't
need a GPT-class model. `routerai` is a drop-in, OpenAI-compatible gateway that
auto-routes each request to the **cheapest model that still clears your quality
bar** — and proves the savings with shadow evals.

> **The LLM platform trilogy — gate -> observe -> govern:**
> [**evalstack**](https://github.com/kartikaneja) proves quality (the *gate*),
> [**tracelens**](https://github.com/kartikaneja) measures cost/latency (the *observe* layer),
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
   candidate model** for that tier from `routes.yaml`.
3. **Providers** — a pluggable abstraction. Ships with a **stub** provider so it
   runs offline with no key; set `ROUTERAI_PROVIDER=openai` (+ `OPENAI_API_KEY`)
   to use a real backend.
4. **Shadow eval** — a pluggable `evaluate(prompt, response) -> score` judge
   scores each response and flags whether the cheap route cleared the
   `quality_bar` (v0.1 uses a heuristic judge; swap in an evalstack/LLM judge).
5. **Metrics** — `GET /metrics` returns total requests, tokens, est cost,
   **estimated savings vs always-premium**, avg eval score, and quality-bar pass
   rate.

### Configuration — `routes.yaml`

Maps quality tiers to ordered candidate models with per-1K-token costs, sets the
`baseline` (always-premium) model for savings math, the `quality_bar`, and the
`shadow_sample_rate`. Point `ROUTERAI_ROUTES` at your own file to override.

---

## Tests

```bash
pytest -q
```

---

## Roadmap (v0.2+)

- Real OpenAI/Anthropic providers wired end-to-end (stub is the default today).
- Fine-tuned complexity classifier trained on the decision log.
- Semantic caching (return cached answers for near-duplicate prompts).
- Persist the decision log (Postgres) + emit events to **tracelens**.
- CI policy gate: a routing change that regresses shadow-eval quality fails the build.
- Next.js playground + savings dashboard at `routerai.kartikaneja.com`.
- Latency-SLO-aware routing; per-tenant policies.

## License

MIT — see [LICENSE](LICENSE).
