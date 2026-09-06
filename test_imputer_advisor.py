"""
Tests for imputer_advisor.py — run with:  python test_imputer_advisor.py
(Plain asserts, no pytest needed, so any community user can run it.)
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from imputer_advisor import (
    describe_data, score_imputers, recommend_imputer,
    get_numeric_imputer, ImputerAdvisor, SIMPLE, KNN, MICE,
)

passed = failed = 0

def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"PASS  {name}")
        passed += 1
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
        failed += 1


def t_describe_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", None, "y"]})
    ch = describe_data(df)
    assert ch["n_samples"] == 3 and ch["n_numeric"] == 1 and ch["n_categorical"] == 1
    assert ch["missing_rate_overall"] > 0 and ch["max_col_missing"] > 0


def t_correlated_recommends_mice():
    rng = np.random.RandomState(0)
    x1 = rng.normal(0, 1, 600)
    df = pd.DataFrame({"x1": x1, "x2": x1 * 0.9 + rng.normal(0, 0.2, 600),
                       "x3": x1 * 0.8 + rng.normal(0, 0.2, 600)})
    df.iloc[::10, 0] = np.nan
    rec = recommend_imputer(df)
    assert rec["recommended"] == MICE, f"got {rec}"


def t_extreme_missing_recommends_simple():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    df.iloc[:, 0] = np.where(rng.rand(500) < 0.6, np.nan, df.iloc[:, 0])
    rec = recommend_imputer(df)
    assert rec["recommended"] == SIMPLE, f"got {rec}"


def t_tiny_data_avoids_mice():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0] * 20, "b": range(60)})
    rec = recommend_imputer(df)
    assert rec["recommended"] != MICE, f"got {rec}"


def t_factory_types():
    assert isinstance(get_numeric_imputer(KNN), KNNImputer)
    assert isinstance(get_numeric_imputer(MICE), IterativeImputer)
    assert isinstance(get_numeric_imputer(SIMPLE), SimpleImputer)


def t_advisor_transform_no_nans():
    df = pd.DataFrame({"n1": [1.0, np.nan, 3.0, 4.0] * 100,
                       "n2": [2.0, 2.5, np.nan, 5.0] * 100,
                       "c": ["a", "b", None, "a"] * 100})
    adv = ImputerAdvisor().fit(df)
    out = adv.transform(df)
    assert not out.isna().any().any(), "NaNs remain after transform"
    assert adv.recommendation_ in (SIMPLE, KNN, MICE)


def t_scores_bounded():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    scores, _ = score_imputers(describe_data(df))
    assert all(0 <= v <= 100 for v in scores.values())


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("t_")}.items()):
    check(name, fn)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
