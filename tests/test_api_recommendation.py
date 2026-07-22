from app.api import recommendation as recommendation_api
from app.recommendation import LichtFeldSettings, Recommendation, RealityScanSettings, Verdict

FAKE_RECOMMENDATION = Recommendation(
    realityscan=RealityScanSettings(downscale=1, rolling_shutter_correction=False),
    lichtfeld=LichtFeldSettings(max_gaussians=1_000_000, bf16=True),
    verdict=Verdict(use_cloud=False, reason="Local GPU meets LichtFeld's minimum requirements"),
)


def test_recommend_endpoint_returns_computed_recommendation(client, monkeypatch):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    monkeypatch.setattr(
        recommendation_api,
        "compute_recommendation",
        lambda store, job_id, srt_path=None: FAKE_RECOMMENDATION,
    )

    response = client.post(f"/jobs/{job_id}/recommend", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["realityscan"]["downscale"] == 1
    assert body["lichtfeld"]["max_gaussians"] == 1_000_000
    assert body["verdict"]["use_cloud"] is False


def test_recommend_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/recommend", json={})
    assert response.status_code == 404


def test_recommend_endpoint_returns_400_on_error(client, monkeypatch):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    def raise_error(store, job_id, srt_path=None):
        raise recommendation_api.RecommendationError("Job has no ingested video yet")

    monkeypatch.setattr(recommendation_api, "compute_recommendation", raise_error)

    response = client.post(f"/jobs/{job_id}/recommend", json={})
    assert response.status_code == 400


def test_get_recommendation_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/recommendation")
    assert response.status_code == 404


def test_get_recommendation_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/recommendation")
    assert response.status_code == 404
