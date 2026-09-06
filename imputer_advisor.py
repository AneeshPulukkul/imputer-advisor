"""
Imputer Advisor — recommend KNN vs MICE (vs Simple) from data characteristics.
============================================================================
Community solution: an *explainable heuristic* decision maker, not a black box.

    from imputer_advisor import recommend_imputer, ImputerAdvisor

    rec = recommend_imputer(df)   # df: features only (no target needed)
    print(rec["recommended"], rec["rationale"])

    advisor = ImputerAdvisor().fit(df)
    df_filled = advisor.transform(df)   # no NaNs left

Design (see ARCHITECTURE.md):
  1. describe_data()   -> measurable characteristics (missing rates, correlation,
     numeric/categorical mix, size, skew).
  2. score_imputers()  -> transparent 0-100 scores for simple/knn/mice.
  3. recommend_imputer()-> winner + confidence + human-readable rationale.
  4. ImputerAdvisor    -> sklearn-compatible fit/transform wrapper around the
     recommendation (numerics get knn/mice/simple, categoricals always
     most_frequent + unknown handling left to the caller's encoder).

Rules of thumb encoded:
  - Extreme missingness (>40% overall or any column >50%) -> Simple (robust).
  - Tiny data (<300 rows) -> Simple or KNN (MICE overfits / unstable).
  - Numeric-heavy + correlated (mean |corr| >= .25) + enough rows -> MICE.
  - Otherwise -> KNN as a fast local default.
  - Wildly different numeric scales (max-std/min-std > 10, since KNN imputes
    *before* scaling) -> demote KNN (distances dominated by one column).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

SIMPLE = "simple"
KNN = "knn"
MICE = "mice"

NUMERIC_KINDS = ("int16", "int32", "int64", "float16", "float32", "float64",
                 "Int64", "Float64")


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _numeric_cols(df)]


def describe_data(df: pd.DataFrame) -> Dict:
    """Compute human-meaningful characteristics used for the decision."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    n_samples, n_features = df.shape
    num = _numeric_cols(df)
    cat = _categorical_cols(df)

    missing_overall = float(df.isna().mean().mean()) if n_features else 0.0
    per_col = df.isna().mean()
    max_col_missing = float(per_col.max()) if len(per_col) else 0.0
    n_cols_with_missing = int((per_col > 0).sum())

    # Correlation on numerics: temp-median fill just for measurement (not modeling).
    mean_abs_corr = 0.0
    if len(num) >= 2:
        tmp = df[num].copy()
        tmp = tmp.fillna(tmp.median())
        # drop constant cols (corr undefined)
        tmp = tmp.loc[:, tmp.std() > 1e-12]
        if tmp.shape[1] >= 2:
            corr = tmp.corr(numeric_only=True).abs()
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            vals = corr.values[mask]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                mean_abs_corr = float(np.mean(vals))

    # Skew of numerics (high skew -> median/simple more attractive, KNN distances noisier)
    mean_abs_skew = 0.0
    if num:
        sk = df[num].skew(numeric_only=True).abs()
        sk = sk[~sk.isna()]
        if len(sk):
            mean_abs_skew = float(sk.mean())

    numeric_fraction = len(num) / max(n_features, 1)
    rows_per_feature = n_samples / max(n_features, 1)

    # Scale disparity: KNNImputer uses Euclidean distance BEFORE any scaler
    # in a typical pipeline, so features on wildly different scales (e.g.
    # size_sqft std~400 vs num_rooms std~1.7) let one column dominate the
    # neighbor search. MICE (per-column regressors) is scale-invariant.
    scale_disparity = 1.0
    if len(num) >= 2:
        tmp_sd = df[num].copy().fillna(df[num].median())
        stds = tmp_sd.std()
        stds = stds[stds > 1e-12]
        if len(stds) >= 2:
            scale_disparity = float(stds.max() / stds.min())

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_numeric": len(num),
        "n_categorical": len(cat),
        "numeric_cols": num,
        "categorical_cols": cat,
        "missing_rate_overall": round(missing_overall, 4),
        "max_col_missing": round(max_col_missing, 4),
        "n_cols_with_missing": n_cols_with_missing,
        "mean_abs_corr": round(mean_abs_corr, 4),
        "mean_abs_skew": round(mean_abs_skew, 4),
        "numeric_fraction": round(numeric_fraction, 4),
        "rows_per_feature": round(rows_per_feature, 2),
        "scale_disparity": round(scale_disparity, 2),
    }


def score_imputers(ch: Dict) -> Tuple[Dict[str, float], List[str]]:
    """Transparent scoring. Returns (scores, notes)."""
    notes: List[str] = []
    scores = {SIMPLE: 40.0, KNN: 50.0, MICE: 50.0}

    m_over, m_max = ch["missing_rate_overall"], ch["max_col_missing"]
    n, corr = ch["n_samples"], ch["mean_abs_corr"]
    num_frac = ch["numeric_fraction"]

    # --- Extreme missingness: complex imputers become unstable ---
    if m_max > 0.5 or m_over > 0.4:
        scores[SIMPLE] += 35
        scores[KNN] -= 20
        scores[MICE] -= 25
        notes.append(f"extreme missingness (overall={m_over}, max-col={m_max}): favor Simple")
    elif m_over > 0.2:
        scores[SIMPLE] += 10
        scores[KNN] -= 5
        scores[MICE] -= 5
        notes.append(f"high missingness ({m_over}): slight penalty on KNN/MICE")
    elif m_over == 0:
        notes.append("no missing values: any imputer works; recommendation is for pipeline design")

    # --- Sample size ---
    if n < 300:
        scores[MICE] -= 20
        scores[SIMPLE] += 12
        notes.append(f"tiny data (n={n}): MICE unstable, favor Simple/KNN")
    elif n >= 1000:
        scores[MICE] += 8
        scores[KNN] += 5
        notes.append(f"large data (n={n}): supports model-based MICE")
    elif n >= 500:
        scores[MICE] += 5

    # --- Correlation among numerics: MICE's main advantage ---
    if corr >= 0.4:
        scores[MICE] += 20
        notes.append(f"strong numeric correlation (mean|r|={corr}): MICE can regress missing cols well")
    elif corr >= 0.25:
        scores[MICE] += 12
        notes.append(f"moderate correlation (mean|r|={corr}): lean MICE")
    elif corr < 0.15 and ch["n_numeric"] >= 2:
        scores[MICE] -= 12
        scores[KNN] += 5
        notes.append(f"weak correlation (mean|r|={corr}): MICE has little linear signal")

    # --- Feature mix ---
    if num_frac < 0.5 and ch["n_categorical"] > 0:
        scores[MICE] -= 10
        notes.append(f"categorical-heavy ({ch['n_categorical']} cat cols): MICE on one-hots is poor; KNN/Simple safer")
    elif num_frac >= 0.8:
        scores[MICE] += 5
        notes.append("numeric-heavy: MICE applicable to most columns")

    # --- Skew ---
    if ch["mean_abs_skew"] > 2.0:
        scores[SIMPLE] += 5
        notes.append(f"high skew ({ch['mean_abs_skew']}): median-Simple robust")

    # --- Scale disparity: KNN distance is dominated by the largest-scale column
    # (imputation happens before scaling in a standard pipeline) ---
    disp = ch.get("scale_disparity", 1.0)
    if disp > 50:
        scores[KNN] -= 15
        notes.append(f"huge scale disparity (max-std/min-std={disp}): KNN neighbors "
                     "dominated by one column unless you scale first; demote KNN")
    elif disp > 10:
        scores[KNN] -= 10
        notes.append(f"large scale disparity ({disp}): KNN at a disadvantage pre-scaling")

    for k in scores:
        scores[k] = round(float(np.clip(scores[k], 0, 100)), 1)
    return scores, notes


def recommend_imputer(df: pd.DataFrame, random_state: int = 42) -> Dict:
    """Recommend one of 'simple' | 'knn' | 'mice' with rationale.

    Returns dict with keys: recommended, confidence, scores, rationale,
    characteristics. `random_state` is stored for reproducible imputer builds.
    """
    ch = describe_data(df)
    scores, notes = score_imputers(ch)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winner, runner = ranked[0], ranked[1]
    margin = winner[1] - runner[1]
    confidence = round(float(np.clip(0.5 + margin / 60.0, 0.5, 0.95)), 2)

    label = {SIMPLE: "SimpleImputer (median/most_frequent)",
             KNN: "KNNImputer (k=5)", MICE: "IterativeImputer / MICE"}[winner[0]]
    rationale = f"Recommend {label} for n={ch['n_samples']}, {ch['n_numeric']} numeric + " \
                f"{ch['n_categorical']} categorical, missing={ch['missing_rate_overall']} " \
                f"(max-col={ch['max_col_missing']}), mean|r|={ch['mean_abs_corr']}. " + " | ".join(notes)
    return {
        "recommended": winner[0],
        "confidence": confidence,
        "scores": scores,
        "rationale": rationale,
        "characteristics": ch,
        "random_state": random_state,
    }


def get_numeric_imputer(name: str, random_state: int = 42):
    """Factory for the numeric imputer behind a recommendation."""
    if name == KNN:
        return KNNImputer(n_neighbors=5)
    if name == MICE:
        return IterativeImputer(max_iter=10, random_state=random_state)
    if name == SIMPLE:
        return SimpleImputer(strategy="median")
    raise ValueError(f"unknown imputer '{name}'; choose from simple/knn/mice")


@dataclass
class ImputerAdvisor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible auto-imputer.

    fit() learns the recommendation; transform() applies it:
      numerics     -> recommended (median / knn / mice)
      categoricals -> most_frequent (always; MICE/KNN on one-hots is discouraged)

    Attributes after fit: recommendation_, characteristics_, scores_, rationale_
    """
    random_state: int = 42
    recommendation_: str = field(default="", init=False)
    characteristics_: Dict = field(default_factory=dict, init=False)
    scores_: Dict = field(default_factory=dict, init=False)
    rationale_: str = field(default="", init=False)
    _num_imp_: object = field(default=None, init=False)
    _cat_imp_: object = field(default=None, init=False)
    _num_cols_: List[str] = field(default_factory=list, init=False)
    _cat_cols_: List[str] = field(default_factory=list, init=False)

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X = X.replace({None: np.nan})  # normalize None -> NaN (esp. object cols)
        rec = recommend_imputer(X, random_state=self.random_state)
        self.recommendation_ = rec["recommended"]
        self.characteristics_ = rec["characteristics"]
        self.scores_ = rec["scores"]
        self.rationale_ = rec["rationale"]
        self._num_cols_ = self.characteristics_["numeric_cols"]
        self._cat_cols_ = self.characteristics_["categorical_cols"]
        if self._num_cols_:
            self._num_imp_ = get_numeric_imputer(self.recommendation_, self.random_state)
            self._num_imp_.fit(X[self._num_cols_])
        if self._cat_cols_:
            self._cat_imp_ = SimpleImputer(strategy="most_frequent")
            self._cat_imp_.fit(X[self._cat_cols_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X = X.replace({None: np.nan})
        out = X.copy()
        if self._num_cols_ and self._num_imp_ is not None:
            out[self._num_cols_] = self._num_imp_.transform(X[self._num_cols_])
        if self._cat_cols_ and self._cat_imp_ is not None:
            out[self._cat_cols_] = self._cat_imp_.transform(X[self._cat_cols_])
        return out


if __name__ == "__main__":  # tiny CLI: python imputer_advisor.py --csv data.csv
    import argparse, json
    ap = argparse.ArgumentParser(description="Recommend an imputer for a CSV file.")
    ap.add_argument("--csv", required=True, help="CSV file (features; target column optional)")
    ap.add_argument("--target", default=None, help="target column to exclude from analysis")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])
    print(json.dumps(recommend_imputer(df), indent=2, default=str))
