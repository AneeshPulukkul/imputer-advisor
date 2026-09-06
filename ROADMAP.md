# Roadmap

## Done

- [x] Linear Regression + feature engineering tutorial (`linear_regression_tutorial.py`)
- [x] Heuristic Imputer Advisor: simple / knn / mice with scores + rationale
- [x] 4-dataset benchmark harness with 4/4 advisor hit-rate (`BENCHMARK_RESULTS.md`)
- [x] FastAPI advisor API with Postgres-backed history (`backend/`)
- [x] React + Vite UI (`frontend/`)
- [x] Docker Compose + Kubernetes manifests (`docker-compose.yml`, `k8s/`)

## Next

- [ ] **More benchmark datasets** (high-missingness, MNAR, text-heavy, wide `p >> n`).
  Prerequisite for anything learned: 4 datasets cannot train a model without
  overfitting theater.
- [ ] **Learned scoring/decision agent (AI-based).** Replace the hand-set
  weights in `score_imputers()` with a meta-model:
  1. Collect training pairs — `(describe_data(X), best imputer by CV)` — across
     dozens of diverse datasets (an extension of `benchmark_imputers.py` that
     logs rows instead of just a hit-rate).
  2. Train a small classifier (e.g. gradient boosting) mapping characteristics
     → `simple | knn | mice`, with calibrated probabilities as confidence.
  3. Serve it behind the same `recommend_imputer()` interface so the API, UI,
     and stored history are unaffected; keep the heuristic as the fallback when
     the agent is uncertain or unavailable, and keep publishing the rationale
     (top contributing characteristics) so recommendations stay auditable.
  4. Optionally expose the agent as its own endpoint (`POST /api/recommend-agent`)
     so heuristic vs learned recommendations can be A/B-compared on the stored
     history before promoting it to default.
- [ ] **Compute-budget knob.** Let callers pass a time budget; MICE is ~5–10×
  slower, so under tight budgets Simple/KNN should win ties automatically.
- [ ] **Non-linear structure signal.** Pearson correlation misses non-linear
  relationships; add a distance-correlation or mutual-information feature to
  `describe_data()` for the KNN-vs-MICE call.
- [ ] **Auth + per-user history** on the API (currently single-tenant).
