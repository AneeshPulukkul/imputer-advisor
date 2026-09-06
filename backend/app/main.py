"""Imputer Advisor API — recommend simple/knn/mice from data characteristics."""
import io
import os

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import models  # noqa: F401  (registers tables)
from .advisor_service import advise, dataframe_from_records
from .database import get_db
from .schemas import HistoryItem, RecommendationResponse, RecommendRequest

app = FastAPI(title="Imputer Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",") if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _persist(db: Session, dataset_name: str | None, rec: dict) -> int:
    ch = rec["characteristics"]
    row = models.RecommendationRecord(
        dataset_name=dataset_name,
        n_samples=ch["n_samples"], n_features=ch["n_features"],
        n_numeric=ch["n_numeric"], n_categorical=ch["n_categorical"],
        missing_rate=ch["missing_rate_overall"], max_col_missing=ch["max_col_missing"],
        mean_abs_corr=ch["mean_abs_corr"],
        scale_disparity=ch.get("scale_disparity", 1.0),
        recommended=rec["recommended"], confidence=rec["confidence"],
        scores=rec["scores"], rationale=rec["rationale"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendRequest, db: Session = Depends(get_db)):
    try:
        rec = advise(dataframe_from_records(req.columns, req.rows), req.target_column)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    record_id = _persist(db, req.dataset_name, rec)
    return RecommendationResponse(record_id=record_id, **{k: rec[k] for k in (
        "recommended", "confidence", "scores", "rationale", "characteristics")})


@app.post("/api/recommend-csv", response_model=RecommendationResponse)
async def recommend_csv(
    file: UploadFile = File(...),
    dataset_name: str | None = Query(default=None),
    target_column: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"cannot parse CSV: {e}")
    try:
        rec = advise(df, target_column)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    record_id = _persist(db, dataset_name or file.filename, rec)
    return RecommendationResponse(record_id=record_id, **{k: rec[k] for k in (
        "recommended", "confidence", "scores", "rationale", "characteristics")})


@app.get("/api/history", response_model=list[HistoryItem])
def history(limit: int = Query(default=50, le=200), db: Session = Depends(get_db)):
    return (db.query(models.RecommendationRecord)
            .order_by(models.RecommendationRecord.id.desc()).limit(limit).all())


@app.get("/api/history/{record_id}")
def history_detail(record_id: int, db: Session = Depends(get_db)):
    row = db.get(models.RecommendationRecord, record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="record not found")
    return {
        "id": row.id, "created_at": row.created_at, "dataset_name": row.dataset_name,
        "recommended": row.recommended, "confidence": row.confidence,
        "scores": row.scores, "rationale": row.rationale,
        "characteristics": {
            "n_samples": row.n_samples, "n_features": row.n_features,
            "n_numeric": row.n_numeric, "n_categorical": row.n_categorical,
            "missing_rate_overall": row.missing_rate,
            "max_col_missing": row.max_col_missing,
            "mean_abs_corr": row.mean_abs_corr,
            "scale_disparity": row.scale_disparity,
        },
    }
