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

app = FastAPI(title="Splaterra")
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


class NoCacheStaticFiles(StaticFiles):
    # The UI shell (HTML/CSS/JS) gets edited and re-served constantly during
    # development, but the owner mostly tests from a phone over Tailscale and
    # won't reliably hard-refresh - a browser silently reusing a stale cached
    # copy without revalidating looks identical to a real bug (e.g. a
    # function "not defined" that has been on disk the whole time). Forcing
    # revalidation on every request keeps ETags doing their job (fast 304s)
    # while guaranteeing the served code is never silently out of date.
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", NoCacheStaticFiles(directory=_STATIC_DIR, html=True), name="static")
