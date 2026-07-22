from app.api import frames as frames_api


def test_extract_frames_endpoint_returns_count(client, monkeypatch):
    monkeypatch.setattr(frames_api, "extract_frames", lambda store, job_id, **kw: 5)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/extract-frames", json={})
    assert response.status_code == 200
    assert response.json() == {"frame_count": 5}


def test_extract_frames_endpoint_returns_400_on_error(client, monkeypatch):
    def raise_error(store, job_id, **kw):
        raise frames_api.ExtractionError("no ingested video")

    monkeypatch.setattr(frames_api, "extract_frames", raise_error)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/extract-frames", json={})
    assert response.status_code == 400
