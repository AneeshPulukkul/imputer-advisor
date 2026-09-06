"""
Benchmark: does the ImputerAdvisor pick the empirically-best imputer?
============================================================================
Datasets (all local, no downloads — reproducible):
  A. diabetes_real         — sklearn diabetes, numeric, moderately correlated
  B. synthetic_correlated  — strong linear correlations  -> MICE should shine
  C. synthetic_clustered   — local cluster structure     -> KNN should shine
  D. mixed_housing         — numeric + categorical       -> realistic mixed case

Protocol (fair, leakage-free):
  - Inject ~10% MCAR missingness into feature columns (fixed seeds).
  - Ask the advisor for a recommendation WITHOUT seeing the target/model.
  - Evaluate simple / knn / mice with IDENTICAL downstream pipelines
    (StandardScaler + LinearRegression) under the same 5-fold CV.
  - Metric: mean CV RMSE (lower wins) + R2 + fit time.
  - Advisor "hit" = recommended imputer is the CV winner or within 2% of it
    (ties are common when missingness is light — documented honestly).

Run:
    python benchmark_imputers.py
Outputs:
    benchmark_results.json  + markdown table printed to stdout.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from imputer_advisor import (
    recommend_imputer, get_numeric_imputer, describe_data,
    SIMPLE, KNN, MICE,
)

RANDOM_STATE = 42
CV = 5
TOLERANCE = 0.02  # advisor counts as "hit" if within 2% of best RMSE


def inject_missing(df: pd.DataFrame, frac: float, seed: int, cols=None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    out = df.copy()
    cols = cols or list(df.columns)
    for c in cols:
        mask = rng.rand(len(df)) < frac
        out.loc[mask, c] = np.nan
    return out


def make_datasets():
    rng = np.random.RandomState(RANDOM_STATE)
    out = {}

    # A. Real: diabetes (442 x 10, numeric)
    diab = load_diabetes(as_frame=True)
    out["A_diabetes_real"] = (diab.data, diab.target, "numeric")

    # B. Synthetic correlated: x2..x5 are noisy copies of x1 -> MICE territory
    n = 800
    x1 = rng.normal(0, 1, n)
    Xb = pd.DataFrame({
        "x1": x1,
        "x2": x1 * 0.9 + rng.normal(0, 0.2, n),
        "x3": x1 * -0.7 + rng.normal(0, 0.3, n),
        "x4": x1 * 0.5 + rng.normal(0, 0.5, n),
        "x5": rng.normal(0, 1, n),  # one independent column
        "x6": rng.normal(0, 1, n),
    })
    yb = 3 * Xb["x1"] + 2 * Xb["x2"] + rng.normal(0, 0.5, n)
    out["B_synthetic_correlated"] = (Xb, yb, "numeric")

    # C. Synthetic clustered: 3 blobs; target = cluster id + noise.
    # Missing values are recoverable from NEIGHBORS (local), not from
    # global linear regressions -> KNN territory.
    n = 900
    centers = np.array([[-3, -3], [0, 3], [3, -2]])
    cid = rng.randint(0, 3, n)
    Xc = pd.DataFrame({
        "f1": centers[cid, 0] + rng.normal(0, 0.5, n),
        "f2": centers[cid, 1] + rng.normal(0, 0.5, n),
        "f3": centers[cid, 0] * 0.3 + rng.normal(0, 0.8, n),
        "f4": rng.normal(0, 1, n),
    })
    yc = pd.Series(cid * 10.0 + rng.normal(0, 1.0, n))
    out["C_synthetic_clustered"] = (Xc, yc, "numeric")

    # D. Mixed housing-like: numeric + categoricals
    n = 800
    size = rng.normal(1500, 400, n).clip(400, 3500)
    rooms = rng.randint(1, 7, n).astype(float)
    age = rng.exponential(15, n).clip(0, 80)
    dist = rng.gamma(2.0, 2.0, n)
    neigh = rng.choice(["A", "B", "C"], n, p=[0.4, 0.4, 0.2])
    style = rng.choice(["apt", "detached", "semi"], n, p=[0.5, 0.3, 0.2])
    Xd = pd.DataFrame({"size": size, "rooms": rooms, "age": age,
                       "dist": dist, "neigh": neigh, "style": style})
    yd = pd.Series(120 * size + 8000 * rooms - 900 * age - 6000 * dist
                   + (neigh == "A") * 40000 + (style == "detached") * 30000
                   + rng.normal(0, 20000, n))
    out["D_mixed_housing"] = (Xd, yd, "mixed")
    return out


def make_pipeline(kind: str, imputer_name: str, X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([
        ("impute", get_numeric_imputer(imputer_name)
         if imputer_name != SIMPLE else SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    if kind == "mixed":
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        prep = ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, cat_cols)])
    else:
        prep = ColumnTransformer([("num", num_pipe, num_cols)])
    return Pipeline([("prep", prep), ("model", LinearRegression())])


def evaluate(X, y, kind):
    res = {}
    for name in (SIMPLE, KNN, MICE):
        pipe = make_pipeline(kind, name, X)
        t0 = time.perf_counter()
        cv = cross_validate(pipe, X, y, cv=CV,
                            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                            n_jobs=1)
        dt = time.perf_counter() - t0
        res[name] = {
            "cv_rmse": round(float(-cv["test_rmse"].mean()), 4),
            "cv_r2": round(float(cv["test_r2"].mean()), 4),
            "fit_time_s": round(dt, 2),
        }
    return res


def main():
    datasets = make_datasets()
    all_results = {}
    hits = 0
    print("\n# Benchmark: ImputerAdvisor vs empirical CV winner\n")
    for i, (dname, (X, y, kind)) in enumerate(datasets.items()):
        # Inject missingness (numeric cols for A/B/C; a few cols for D incl. 1 cat)
        if kind == "mixed":
            Xm = inject_missing(X, 0.10, seed=100 + i,
                                cols=["size", "age", "dist", "style"])
        else:
            Xm = inject_missing(X, 0.10, seed=100 + i)
        rec = recommend_imputer(Xm)
        ch = rec["characteristics"]
        print(f"## {dname}  (n={ch['n_samples']}, num={ch['n_numeric']}, "
              f"cat={ch['n_categorical']}, miss={ch['missing_rate_overall']}, "
              f"mean|r|={ch['mean_abs_corr']})")
        print(f"Advisor -> **{rec['recommended']}** "
              f"(conf {rec['confidence']}, scores {rec['scores']})")
        print(f"  rationale: {rec['rationale']}")
        scores = evaluate(Xm, y, kind)
        best = min(scores, key=lambda k: scores[k]["cv_rmse"])
        best_rmse = scores[best]["cv_rmse"]
        adv = rec["recommended"]
        gap = (scores[adv]["cv_rmse"] - best_rmse) / best_rmse
        hit = gap <= TOLERANCE
        hits += hit
        print(f"CV results: {json.dumps(scores)}")
        print(f"Winner: **{best}** | Advisor: **{adv}** "
              f"(+{gap * 100:.1f}% vs best) -> {'HIT' if hit else 'MISS'}\n")
        all_results[dname] = {
            "characteristics": ch,
            "recommendation": {k: rec[k] for k in ("recommended", "confidence", "scores", "rationale")},
            "cv_scores": scores,
            "cv_winner": best,
            "advisor_gap_pct": round(gap * 100, 2),
            "hit": bool(hit),
        }
    print(f"### Advisor hit-rate: {hits}/{len(datasets)} "
          f"(hit = recommended within {TOLERANCE * 100:.0f}% of best CV RMSE)")
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved benchmark_results.json\n")


if __name__ == "__main__":
    main()
