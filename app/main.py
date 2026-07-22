from fastapi import FastAPI

from app.api.ingestion import router as ingestion_router
from app.api.jobs import router as jobs_router

app = FastAPI(title="Splat Prep")
app.include_router(jobs_router)
app.include_router(ingestion_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
