from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.alignment import router as alignment_router
from app.api.export import router as export_router
from app.api.frames import router as frames_router
from app.api.hardware import router as hardware_router
from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router
from app.api.recommendation import router as recommendation_router
from app.api.training import router as training_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)
app.include_router(frames_router)
app.include_router(hardware_router)
app.include_router(recommendation_router)
app.include_router(alignment_router)
app.include_router(training_router)
app.include_router(export_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
