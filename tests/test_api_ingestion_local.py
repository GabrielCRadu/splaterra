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
