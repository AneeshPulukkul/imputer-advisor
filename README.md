# Imputer Advisor — Simple, KNN, or MICE?

An explainable decision maker that recommends the right scikit-learn imputer
from your data's characteristics — plus the tutorial, benchmark, API, and UI
around it. **New here? Start with the Quickstart tutorial below**, then point
the advisor at your own data.

> 📖 **Adopting this project? Read [`USER_GUIDE.md`](USER_GUIDE.md)** — run it
> with Docker or natively, click through your first recommendation (with
> screenshots), and find troubleshooting + FAQ.

## Quickstart tutorial (start here)

A hands-on, 10-minute tutorial that teaches **Linear Regression** by building
a complete **feature engineering pipeline** in scikit-learn — including a fair
head-to-head comparison of **KNN vs MICE imputation**.

- 📖 **Read it:** [`TUTORIAL.md`](TUTORIAL.md)
- ▶️ **Run it:** `linear_regression_tutorial.py` (same content, executable)

## What you will learn

- How Linear Regression works (`y = Xw + b`, MSE objective, key assumptions)
- **Encoding** — `OneHotEncoder` for nominal features, `OrdinalEncoder` for ordered features
- **Feature scaling** — `StandardScaler` vs `MinMaxScaler`, and why scaling matters for coefficients, regularization, and distance-based steps like `KNNImputer`
- **Feature creation** — domain features via `FunctionTransformer` + non-linear expansion via `PolynomialFeatures`
- **Feature selection** — `SelectKBest(f_regression)`, `VarianceThreshold`, done *inside* the `Pipeline` to avoid leakage
- **Missing-value imputation** — `KNNImputer` vs `IterativeImputer` (MICE) with identical pipelines and 5-fold CV

### Run the tutorial

```bash
pip install scikit-learn pandas numpy matplotlib
python linear_regression_tutorial.py
```

This generates a synthetic housing dataset (so no download is needed), trains both pipelines, prints CV + test metrics, and saves three plots:

- `plot_actual_vs_pred.png`
- `plot_residuals.png`
- `plot_knn_vs_mice.png`

## Imputation comparison: KNN vs MICE

| Approach | How it works | Strengths | Watch out for |
|---|---|---|---|
| **KNN** (`KNNImputer`) | Fills each NaN with the mean of its K nearest complete rows | Simple, fast, good local baseline | Sensitive to scale — scale first; struggles if no similar rows exist |
| **MICE** (`IterativeImputer`) | Regress each missing column on all others, iterate | Stronger on correlated numerics | Slower, more hyperparameters |

Both are evaluated fairly: same `ColumnTransformer` + `SelectKBest` + `LinearRegression`, same 5-fold CV splits, same held-out test set.

Example run (1000 samples, ~10% missingness):

| Pipeline | CV RMSE | CV R² | Test RMSE | Test R² |
|---|---|---|---|---|
| KNN (k=5) | ~25,550 | ~0.83 | ~29,860 | ~0.77 |
| MICE | ~24,860 | ~0.84 | ~27,530 | ~0.80 |

> Numbers vary slightly per seed — the point is the *comparison setup*, not the exact values.

## Pipeline sketch

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([
            ("impute", KNNImputer() | IterativeImputer()),  # <-- swap here
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])),
    ("select", SelectKBest(f_regression, k=15)),
    ("model", LinearRegression()),
])
```

## Repo structure

```text
.
├── TUTORIAL.md                    # the tutorial, readable (start here)
├── linear_regression_tutorial.py  # the tutorial, runnable (run this)
├── USER_GUIDE.md                  # adoption walkthrough with screenshots
├── docs/screenshots/              # UI + API screenshots used by the guide
├── imputer_advisor.py             # intelligent imputer recommender (simple/knn/mice)
├── benchmark_imputers.py          # 4-dataset experiment harness
├── test_imputer_advisor.py        # 7 tests, no pytest needed
├── backend/                       # FastAPI advisor API (Postgres-backed history)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/ (main.py, advisor_service.py, models.py, schemas.py, database.py)
├── frontend/                      # React + Vite UI (paste CSV / upload / history)
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/ (App.jsx, api.js, main.jsx, styles.css)
├── docker-compose.yml             # db + backend + frontend, one command
├── .env.example                   # copy to .env for compose
├── k8s/                           # namespace, postgres, backend, frontend, ingress
├── ARCHITECTURE.md                # advisor design + decision rules
├── BENCHMARK_RESULTS.md           # experiment protocol, results, findings
├── benchmark_results.json         # raw machine-readable benchmark output
├── README.md
├── plot_actual_vs_pred.png
├── plot_residuals.png
└── plot_knn_vs_mice.png
```

## Imputer Advisor (automatic KNN vs MICE vs Simple recommender)

`imputer_advisor.py` picks the right imputer from measurable data
characteristics (missing rates, numeric correlation, numeric/categorical mix,
sample size, skew, and numeric scale disparity) — and explains itself:

```python
from imputer_advisor import recommend_imputer, ImputerAdvisor

rec = recommend_imputer(X)
print(rec["recommended"], rec["confidence"])  # e.g. mice 0.87
print(rec["rationale"])                       # human-readable why

X_filled = ImputerAdvisor().fit_transform(X)  # drop-in cleaning step
```

Benchmarked on 4 datasets (real diabetes + 3 synthetic): **hit-rate 4/4**
— see `BENCHMARK_RESULTS.md` for protocol and numbers, `ARCHITECTURE.md` for
the decision rules (including the scale-disparity lesson that fixed an
initial 3/4 miss).

```bash
python test_imputer_advisor.py   # 7 tests
python benchmark_imputers.py     # reproduces benchmark_results.json
```

## Full-stack app (FastAPI + Postgres + React/Vite)

The advisor is exposed as a REST API with a web UI; every recommendation is
stored in PostgreSQL history.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `POST /api/recommend` | JSON `{columns, rows, dataset_name?, target_column?}` → recommendation + saved record |
| `POST /api/recommend-csv` | multipart CSV upload (`?dataset_name=&target_column=`) |
| `GET /api/history` / `GET /api/history/{id}` | past recommendations |

Local dev (no Docker):

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000  # SQLite fallback, no Postgres needed
cd frontend && npm install && npm run dev           # http://localhost:5173
```

Docker Compose (Postgres included):

```bash
cp .env.example .env   # set a real POSTGRES_PASSWORD
docker compose up --build
# UI: http://localhost:8080   API: http://localhost:8000  Docs: http://localhost:8000/docs
```

Kubernetes (`k8s/`): namespace + Postgres (PVC) + backend (×2) + frontend (×2) + ingress (`advisor.local`, `/api` → backend, `/` → frontend):

```bash
kubectl apply -f k8s/
# build/push images first and set image: to your registry; create the
# postgres-credentials secret (template in k8s/postgres-secret.yaml) securely.
```

## Roadmap

See `ROADMAP.md` — next up: more benchmark datasets, then a learned
scoring/decision agent (meta-model over dataset characteristics, served behind
the same API with the heuristic as fallback), plus a compute-budget knob.

## Exercises

1. Swap `StandardScaler` → `MinMaxScaler`. Does test RMSE change? Why / why not?
2. Try `k=5` vs `k=30` in `SelectKBest`. Watch CV R² for overfitting.
3. Replace `LinearRegression` with `Ridge(alpha=1.0)`. When does it win?
4. Try `RobustScaler` on the outlier-heavy `age` / `dist` columns and compare.

## Requirements

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib

## Note on `IterativeImputer`

It is experimental in scikit-learn, so the tutorial enables it explicitly:

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
```
