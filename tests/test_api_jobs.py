def test_create_and_get_job(client):
    create_response = client.post("/jobs", json={"name": "Curte interioara"})
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["name"] == "Curte interioara"
    assert created["frame_count"] is None

    get_response = client.get(f"/jobs/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == created["id"]


def test_get_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_list_jobs_returns_created_jobs(client):
    client.post("/jobs", json={"name": "A"})
    client.post("/jobs", json={"name": "B"})

    response = client.get("/jobs")
    assert response.status_code == 200
    names = {job["name"] for job in response.json()}
    assert names == {"A", "B"}


def test_get_job_includes_recommendation_json_field(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}")
    body = response.json()
    assert "recommendation_json" in body
    assert body["recommendation_json"] is None


def test_get_job_stages_returns_recorded_stages(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    from app import db
    from app.job_store import JobStore

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "ingestion", "done", started_at="t0", finished_at="t1")
    conn.close()

    response = client.get(f"/jobs/{job_id}/stages")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["stage"] == "ingestion"
    assert body[0]["status"] == "done"


def test_get_job_stages_empty_list_for_new_job(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/stages")
    assert response.status_code == 200
    assert response.json() == []


def test_get_job_stages_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/stages")
    assert response.status_code == 404
