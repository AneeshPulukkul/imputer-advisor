"""Bridge between the API and the imputer_advisor engine at the repo root.

Resolves the import for both layouts:
  - local dev:  <repo>/imputer_advisor.py  (two levels above this file's backend/app)
  - container:  imputer_advisor.py copied next to /app (see backend/Dockerfile)
"""
import os
import sys
from typing import Any

import pandas as pd

try:
    from imputer_advisor import recommend_imputer
except ImportError:
    _CANDIDATES = [
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "/app",
    ]
    for _p in _CANDIDATES:
        if os.path.isfile(os.path.join(_p, "imputer_advisor.py")) and _p not in sys.path:
            sys.path.insert(0, _p)
    from imputer_advisor import recommend_imputer  # raises honestly if still missing


def dataframe_from_records(columns: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    return df.replace({None: float("nan")})


def advise(df: pd.DataFrame, target_column: str | None = None) -> dict:
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    if df.shape[1] == 0:
        raise ValueError("no feature columns left after excluding target_column")
    return recommend_imputer(df)
