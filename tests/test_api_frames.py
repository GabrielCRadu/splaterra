def test_extract_frames_endpoint_returns_202_and_invokes_extraction(client, monkeypatch):
    from app.api import frames as frames_api

    calls = []

    def fake_extract_frames(store, job_id, *, selection_method, num_frames, fps):
        calls.append((job_id, selection_method, num_frames, fps))
        return 5

    monkeypatch.setattr(frames_api, "extract_frames", fake_extract_frames)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "num_frames": 100, "fps": 5},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, "batched", 100, 5)]


def test_extract_frames_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/extract-frames", json={})
    assert response.status_code == 404
