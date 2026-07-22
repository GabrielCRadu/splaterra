from fastapi import FastAPI

app = FastAPI(title="Splat Prep")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
