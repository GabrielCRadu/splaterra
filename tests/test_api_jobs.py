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


def test_get_job_includes_alignment_fields(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}")
    body = response.json()
    assert "registration_path" in body
    assert "point_cloud_path" in body
    assert body["registration_path"] is None


def test_delete_job_removes_job_and_its_files(client):
    from app.paths import job_dir

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "marker.txt").write_text("data")

    response = client.delete(f"/jobs/{job_id}")
    assert response.status_code == 204

    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert not directory.exists()


def test_delete_job_unknown_returns_404(client):
    response = client.delete("/jobs/does-not-exist")
    assert response.status_code == 404


def test_delete_job_blocked_while_stage_running(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "alignment", "running")
    conn.close()

    response = client.delete(f"/jobs/{job_id}")
    assert response.status_code == 409

    # job must still exist - blocked deletion is not a partial deletion
    assert client.get(f"/jobs/{job_id}").status_code == 200


def test_get_job_stages_includes_progress_percent(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "frame_extraction", "running", started_at="t0", progress_percent=33.0)
    conn.close()

    response = client.get(f"/jobs/{job_id}/stages")
    assert response.status_code == 200
    stage = response.json()[0]
    assert stage["progress_percent"] == 33.0


def test_get_job_stages_progress_percent_defaults_to_none(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    from app import db
    from app.job_store import JobStore

    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "ingestion", "done")
    conn.close()

    response = client.get(f"/jobs/{job_id}/stages")
    assert response.json()[0]["progress_percent"] is None
