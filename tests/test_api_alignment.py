def test_align_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import alignment as alignment_api

    calls = []

    def fake_compute_alignment(store, job_id):
        calls.append(job_id)

    monkeypatch.setattr(alignment_api, "compute_alignment", fake_compute_alignment)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/align")

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [job_id]


def test_align_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/align")
    assert response.status_code == 404


def test_cancel_align_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/align/cancel")
    assert response.status_code == 404


def test_cancel_align_endpoint_reports_false_when_nothing_running(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/align/cancel")
    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def test_cancel_align_endpoint_kills_registered_process(client, monkeypatch):
    from unittest.mock import MagicMock

    from app import process_registry

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    process = MagicMock()
    process.poll.return_value = None
    process_registry.register(job_id, "alignment", process)

    response = client.post(f"/jobs/{job_id}/align/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    process.kill.assert_called_once()
    process_registry.unregister(job_id, "alignment")


def test_get_alignment_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/alignment")
    assert response.status_code == 404


def test_get_alignment_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/alignment")
    assert response.status_code == 404


def test_get_alignment_returns_paths_once_computed(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(
        job_id, colmap_sparse_dir="/data/sparse/0", undistorted_images_dir="/data/images"
    )
    conn.close()

    response = client.get(f"/jobs/{job_id}/alignment")
    assert response.status_code == 200
    assert response.json() == {
        "colmap_sparse_dir": "/data/sparse/0",
        "undistorted_images_dir": "/data/images",
    }
