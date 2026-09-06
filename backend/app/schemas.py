"""Pydantic request/response schemas for the advisor API."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    dataset_name: Optional[str] = None
    columns: list[str] = Field(min_length=1)
    rows: list[list[Any]] = Field(min_length=1)
    target_column: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "dataset_name": "housing-sample",
        "columns": ["size", "rooms", "style"],
        "rows": [[1500, 3, "apt"], [None, 4, "detached"], [1800, None, None]],
    }}}


class RecommendationResponse(BaseModel):
    record_id: int
    recommended: str
    confidence: float
    scores: dict[str, float]
    rationale: str
    characteristics: dict


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    dataset_name: Optional[str]
    recommended: str
    confidence: float
    n_samples: int
    n_features: int

    model_config = {"from_attributes": True}
