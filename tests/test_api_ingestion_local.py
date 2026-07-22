def test_ingest_local_sets_raw_video_path(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake video bytes")

    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 200
    assert response.json()["raw_video_path"].endswith("video.mp4")

    job = client.get(f"/jobs/{job_id}").json()
    assert job["raw_video_path"] is not None


def test_ingest_local_unknown_job_returns_404(client, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"data")
    response = client.post(
        "/jobs/does-not-exist/ingest/local", json={"source_path": str(source)}
    )
    assert response.status_code == 404


def test_ingest_local_missing_source_returns_400(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(
        f"/jobs/{job_id}/ingest/local", json={"source_path": "/no/such/file.mp4"}
    )
    assert response.status_code == 400
