def test_recommend_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import recommendation as recommendation_api

    calls = []

    def fake_compute_recommendation(store, job_id, srt_path):
        calls.append((job_id, srt_path))

    monkeypatch.setattr(recommendation_api, "compute_recommendation", fake_compute_recommendation)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/recommend", json={})

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, None)]


def test_recommend_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/recommend", json={})
    assert response.status_code == 404


def test_get_recommendation_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/recommendation")
    assert response.status_code == 404


def test_get_recommendation_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/recommendation")
    assert response.status_code == 404
