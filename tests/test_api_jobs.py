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
