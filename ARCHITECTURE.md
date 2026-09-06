# Architecture: Imputer Advisor

## Problem

Choosing between `SimpleImputer`, `KNNImputer`, and `IterativeImputer` (MICE) is
usually done by habit. The wrong choice is silent: the pipeline still runs, just
worse. This module makes the choice **data-driven and explainable**.

## Design principles

1. **Heuristics, not black box.** Every recommendation ships with scores and a
   human-readable rationale, so a reviewer can disagree with it intelligently.
2. **Recommend from features alone.** The advisor never sees the target or the
   model score — it must generalize to any downstream estimator.
3. **Numerics decide; categoricals stay simple.** MICE/KNN on one-hot columns is
   discouraged in the literature, so categoricals always get `most_frequent`.
   The advisor recommends the *numeric* imputer.
4. **Fail safe.** Under extreme missingness or tiny data, recommend `Simple`.

## Components (`imputer_advisor.py`)

```text
describe_data(df)      -> characteristics dict
score_imputers(ch)     -> {simple, knn, mice} scores 0-100 + notes
recommend_imputer(df)  -> {recommended, confidence, scores, rationale, ...}
get_numeric_imputer()  -> sklearn imputer instance (factory)
ImputerAdvisor         -> sklearn-compatible fit/transform wrapper
```

### Characteristics measured

| Signal | Why it matters |
|---|---|
| `missing_rate_overall`, `max_col_missing` | >40% overall or >50% in one column: complex imputers go unstable |
| `n_samples`, `rows_per_feature` | MICE fits a regressor per column — needs rows |
| `mean_abs_corr` (numeric pairwise \|r\|) | MICE's core advantage: regress missing cols on correlated ones |
| `numeric_fraction` | Categorical-heavy data weakens the case for MICE |
| `mean_abs_skew` | High skew favors median-Simple robustness |
| `scale_disparity` (max-std/min-std) | KNN imputes **before** scaling: one huge-scale column hijacks Euclidean distance |

### Decision flow

```text
extreme missing? ──yes──> SIMPLE
tiny data (n<300)? ──yes──> SIMPLE/KNN (MICE penalized -20)
numeric-heavy + corr>=.25 + n>=500? ──yes──> MICE boosted (+12..+20)
weak corr (<.15)? ──yes──> MICE penalized (-12), KNN nudged (+5)
categorical-heavy? ──yes──> MICE penalized (-10)
scale disparity >50? ──yes──> KNN penalized (-15)
else ──> highest score wins; confidence = 0.5 + margin/60 (capped 0.95)
```

### Lesson learned from benchmarking (real iteration)

The first advisor version scored **3/4** (see git history / `BENCHMARK_RESULTS.md`).
The miss: mixed housing data with `size_sqft std≈365` vs `num_rooms std≈1.7`
(disparity ≈ 211). KNN's neighbor search was dominated by `size`, and KNN came
**last** (RMSE 27,181 vs MICE 26,208). The fix — the `scale_disparity` penalty
above — is textbook KNN behavior (distance-based, pre-scaling), not benchmark
overfitting. After the fix: **4/4**.

## Usage

```python
from imputer_advisor import recommend_imputer, ImputerAdvisor

rec = recommend_imputer(X)          # X: feature DataFrame
print(rec["recommended"], rec["confidence"])
print(rec["rationale"])

filled = ImputerAdvisor().fit_transform(X)   # drop-in cleaning step
```

CLI: `python imputer_advisor.py --csv data.csv [--target price]`

## Limitations (honest)

- Heuristics encode *common* cases; adversarial/missing-not-at-random (MNAR)
  data can still fool them. The rationale field exists so you can override.
- MICE cost: ~5–10x the fit time of Simple/KNN in our benchmark — the advisor
  does not currently weigh compute budget. Pass a time limit externally if needed.
- Correlation is measured linearly (Pearson); strongly non-linear structure is
  only partly captured. The clustered benchmark dataset covers this partially.
