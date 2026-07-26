def test_train_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(store, job_id, *, variable_exposure=False):
        calls.append((job_id, variable_exposure))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train")

    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, False)]


def test_train_endpoint_forwards_variable_exposure_flag(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(store, job_id, *, variable_exposure=False):
        calls.append((job_id, variable_exposure))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train", json={"variable_exposure": True})

    assert response.status_code == 202
    assert calls == [(job_id, True)]


def test_train_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/train")
    assert response.status_code == 404


def test_train_endpoint_rejects_second_request_while_already_running(client, monkeypatch):
    # Real bug this guards against: a stale page or fast double-tap firing a
    # second /train while one is already in flight silently orphans the
    # first process from process_registry (only one slot per job/stage),
    # leaving it uncancelable while it keeps running in the background.
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "training", "running", started_at="2026-01-01T00:00:00Z")
    conn.close()

    response = client.post(f"/jobs/{job_id}/train")

    assert response.status_code == 409


def test_cancel_train_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/train/cancel")
    assert response.status_code == 404


def test_cancel_train_endpoint_reports_false_when_nothing_running(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train/cancel")
    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def test_cancel_train_endpoint_kills_registered_process(client, monkeypatch):
    from unittest.mock import MagicMock

    from app import process_registry

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    process = MagicMock()
    process.poll.return_value = None
    process_registry.register(job_id, "training", process)

    response = client.post(f"/jobs/{job_id}/train/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    process.kill.assert_called_once()
    process_registry.unregister(job_id, "training")


def test_get_training_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/training")
    assert response.status_code == 404


def test_get_training_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/training")
    assert response.status_code == 404


def test_get_training_returns_splat_path_once_computed(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(job_id, splat_path="/data/training/result.ply")
    conn.close()

    response = client.get(f"/jobs/{job_id}/training")
    assert response.status_code == 200
    assert response.json() == {"splat_path": "/data/training/result.ply"}


def test_download_training_result_returns_file(client, tmp_path):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    splat_file = tmp_path / "result.ply"
    splat_file.write_bytes(b"fake splat data")

    conn = db.connect()
    JobStore(conn).update_job(job_id, splat_path=str(splat_file))
    conn.close()

    response = client.get(f"/jobs/{job_id}/training/download")
    assert response.status_code == 200
    assert response.content == b"fake splat data"


def test_download_training_result_not_yet_computed_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/training/download")
    assert response.status_code == 404


def test_download_training_result_missing_on_disk_returns_404(client, tmp_path):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    JobStore(conn).update_job(job_id, splat_path=str(tmp_path / "gone.ply"))
    conn.close()

    response = client.get(f"/jobs/{job_id}/training/download")
    assert response.status_code == 404
