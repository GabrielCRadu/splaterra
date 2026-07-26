def test_list_raw_videos_returns_empty_for_job_with_no_clips(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/raw-videos")
    assert response.status_code == 200
    assert response.json() == {"videos": []}


def test_list_raw_videos_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/raw-videos")
    assert response.status_code == 404


def test_list_raw_videos_returns_ingested_clip_names(client, tmp_path):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    source_a = tmp_path / "a_clip.mp4"
    source_b = tmp_path / "b_clip.mp4"
    source_a.write_bytes(b"fake")
    source_b.write_bytes(b"fake")

    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(source_a)})
    client.post(f"/jobs/{job_id}/ingest/local", json={"source_path": str(source_b)})

    response = client.get(f"/jobs/{job_id}/raw-videos")
    assert response.status_code == 200
    assert response.json() == {"videos": ["a_clip.mp4", "b_clip.mp4"]}
