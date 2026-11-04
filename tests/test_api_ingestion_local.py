def test_ingest_local_starts_background_task_and_returns_202(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake video bytes")

    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 202
    assert response.json() == {"status": "started"}

    job = client.get(f"/jobs/{job_id}").json()
    assert job["raw_video_path"] is not None
    assert job["raw_video_path"].endswith("video.mp4")


def test_ingest_local_unknown_job_returns_404(client, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"data")
    response = client.post(
        "/jobs/does-not-exist/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 404


def test_ingest_local_missing_source_records_failed_stage(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(tmp_path / "missing.mp4")}
    )
    assert response.status_code == 202

    stages = client.get(f"/jobs/{job_id}/stages").json()
    ingestion_stage = next(s for s in stages if s["stage"] == "ingestion")
    assert ingestion_stage["status"] == "failed"


def test_cancel_ingestion_endpoint_clears_orphaned_running_stage(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "ingestion", "running", started_at="2026-01-01T00:00:00Z")
    conn.close()

    response = client.post(f"/jobs/{job_id}/ingest/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}

    stages = client.get(f"/jobs/{job_id}/stages").json()
    ingestion_stage = next(s for s in stages if s["stage"] == "ingestion")
    assert ingestion_stage["status"] == "failed"


def test_cancel_ingestion_endpoint_returns_false_when_nothing_running(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/ingest/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def test_cancel_ingestion_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/ingest/cancel")
    assert response.status_code == 404
