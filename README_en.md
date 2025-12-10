# AI-Assisted Migration Prototype

FastAPI prototype that ingests a relational DB schema, performs baseline and heuristic domain decomposition, and produces a draft migration plan (service boundaries, table ownership, sharding plan).

Essentials:
- Input: schema JSON (`tables`, `relations`, `usage`), validated/normalized.
- Analysis: baseline (graph connectivity) and extended “AI” (heuristics + domain rules); metrics, explanations, LLM summaries; timing and token usage.
- Output: migration draft with service boundaries, ownership map, and sharding plan.

## Quick start
- Install deps: `pip install -r requirements.txt`
- Run server: `uvicorn main:app --reload`
- Docs: `http://127.0.0.1:8000/docs`

## Endpoints
- `GET /health` — health check.
- `POST /analysis` — returns baseline vs AI domains, metrics, comparison, timings, and LLM summaries (if enabled).
- `POST /suggestions` — consumes `/analysis` output, returns draft migration plan (services, ownership map, sharding plan).

## Sample request
```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H "Content-Type: application/json" \
  -d @app/data/example_schema.json
```

## Architecture (formal steps)
1. Normalize/validate schema (Pydantic).
2. Build FK + co-usage graph; usage threshold = median of positive counts.
3. Baseline: connected components of FK graph.
4. AI: cut weak edges by usage, seed domains (identity/auth, catalog/inventory, orders/checkout, payments, promo, reference), enforce domain rules (identity/audit/payments/reference separated), attach small components, limited merge on strong co-usage.
5. Post-process join/utility tables.
6. Metrics: domain_count, avg_internal_coupling, cross_domain_edges, over_segmentation, ownership_conflicts. Domain graph: cross-domain edges/weights.
7. Labeling: heuristic summaries + LLM (optional).

## Files of interest
- `main.py` — FastAPI app and routing.
- `app/core/ai_engine.py` — baseline + heuristic engine, metrics, comparison.
- `app/core/llm_adapter.py` — OpenAI-based domain descriptions; enable via `LLM_ENABLED=1` and `OPENAI_API_KEY`.
- `app/utils/db_tools.py` — schema normalization.
- `app/services/migration_service.py` — migration plan from AI domains.
- `app/data/example_schema.json` — sample schema.

## LLM setup (OpenAI)
- Set `LLM_ENABLED=1`, `OPENAI_API_KEY=...`.
- Optional: `OPENAI_MODEL` (default gpt-4o-mini), `OPENAI_BASE_URL` (proxy/custom).
- If missing/failed, `llm_summaries.note` explains the fallback; `token_usage` is returned when available.

## Platform
- Python 3.x, FastAPI + uvicorn, custom heuristic AI-engine; OpenAI (optional) for labeling.
- Example dataset: e-commerce schema (~18 tables, FK + usage).

## Roadmap / research add-ons
- ML/LLM-based clustering; SQL DDL parser; richer metrics/visuals (heatmap cross-domain usage, graph, baseline vs AI splits).
- Case studies: failure cases, where baseline wins.
- Benchmarking: baseline (benchmark) vs AI-engine (tested) on multiple schemas, report deltas and expert alignment.
