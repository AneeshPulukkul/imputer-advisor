"""Persisted recommendation history (Postgres in prod, SQLite for local dev)."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .database import Base, engine


class RecommendationRecord(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    dataset_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    n_samples: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    n_numeric: Mapped[int] = mapped_column(Integer)
    n_categorical: Mapped[int] = mapped_column(Integer)
    missing_rate: Mapped[float] = mapped_column(Float)
    max_col_missing: Mapped[float] = mapped_column(Float)
    mean_abs_corr: Mapped[float] = mapped_column(Float)
    scale_disparity: Mapped[float] = mapped_column(Float)
    recommended: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    # JSONB on Postgres, plain JSON elsewhere (SQLite)
    scores: Mapped[dict] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=False
    )
    rationale: Mapped[str] = mapped_column(Text)


Base.metadata.create_all(bind=engine)
