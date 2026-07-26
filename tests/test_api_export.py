def test_export_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import export as export_api

    calls = []

    def fake_compute_export(store, job_id, *, format):
        calls.append((job_id, format))

    monkeypatch.setattr(export_api, "compute_export", fake_compute_export)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/export", json={"format": "spz"})

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, "spz")]


def test_export_endpoint_rejects_unknown_format(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/export", json={"format": "xyz"})
    assert response.status_code == 422


def test_export_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/export", json={"format": "spz"})
    assert response.status_code == 404


def test_get_export_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/export")
    assert response.status_code == 404


def test_get_export_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/export")
    assert response.status_code == 404


def test_get_export_returns_path_and_format_once_computed(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(job_id, export_path="/data/training/export.spz", export_format="spz")
    conn.close()

    response = client.get(f"/jobs/{job_id}/export")
    assert response.status_code == 200
    assert response.json() == {"export_path": "/data/training/export.spz", "export_format": "spz"}


def test_download_export_result_returns_file(client, tmp_path):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    export_file = tmp_path / "export.spz"
    export_file.write_bytes(b"fake spz data")

    conn = db.connect()
    JobStore(conn).update_job(job_id, export_path=str(export_file), export_format="spz")
    conn.close()

    response = client.get(f"/jobs/{job_id}/export/download")
    assert response.status_code == 200
    assert response.content == b"fake spz data"


def test_download_export_result_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/export/download")
    assert response.status_code == 404


def test_download_export_result_missing_on_disk_returns_404(client, tmp_path):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(
        job_id, export_path=str(tmp_path / "gone.spz"), export_format="spz"
    )
    conn.close()

    response = client.get(f"/jobs/{job_id}/export/download")
    assert response.status_code == 404
