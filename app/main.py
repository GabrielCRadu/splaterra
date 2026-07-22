from fastapi import FastAPI

from app.api.frames import router as frames_router
from app.api.hardware import router as hardware_router
from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router
from app.api.recommendation import router as recommendation_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)
app.include_router(frames_router)
app.include_router(hardware_router)
app.include_router(recommendation_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
