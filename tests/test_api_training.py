def test_train_endpoint_returns_202_and_invokes_service(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
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

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append((job_id, variable_exposure))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train", json={"variable_exposure": True})

    assert response.status_code == 202
    assert calls == [(job_id, True)]


def test_train_endpoint_forwards_max_gaussians_and_iterations(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append((job_id, max_gaussians, iterations))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/train", json={"max_gaussians": 1_500_000, "iterations": 15000}
    )

    assert response.status_code == 202
    assert calls == [(job_id, 1_500_000, 15000)]


def test_train_endpoint_forwards_resize_factor(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append((job_id, resize_factor))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train", json={"resize_factor": 4})

    assert response.status_code == 202
    assert calls == [(job_id, 4)]


def test_train_endpoint_rejects_invalid_resize_factor(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train", json={"resize_factor": 3})
    assert response.status_code == 422


def test_train_endpoint_forwards_eval_and_test_every(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append((job_id, eval, test_every))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train", json={"eval": True, "test_every": 4})

    assert response.status_code == 202
    assert calls == [(job_id, True, 4)]


def test_train_endpoint_rejects_zero_test_every(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train", json={"test_every": 0})
    assert response.status_code == 422


def test_train_endpoint_forwards_cloud_fields(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append((job_id, use_cloud, cloud_provider, cloud_image, cloud_gpu_type))

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/train",
        json={
            "use_cloud": True,
            "cloud_provider": "vastai",
            "cloud_image": "ghcr.io/example/lichtfeld:cuda",
            "cloud_gpu_type": "RTX_4090",
        },
    )

    assert response.status_code == 202
    assert calls == [(job_id, True, "vastai", "ghcr.io/example/lichtfeld:cuda", "RTX_4090")]


def test_train_endpoint_defaults_cloud_provider_to_runpod(client, monkeypatch):
    from app.api import training as training_api

    calls = []

    def fake_compute_training(
        store, job_id, *, variable_exposure=False, max_gaussians=None, iterations=None, resize_factor=None, eval=False, test_every=None, use_cloud=False, cloud_provider="runpod", cloud_image=None, cloud_gpu_type=None
    ):
        calls.append(cloud_provider)

    monkeypatch.setattr(training_api, "compute_training", fake_compute_training)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(f"/jobs/{job_id}/train", json={})

    assert response.status_code == 202
    assert calls == ["runpod"]


def test_train_endpoint_rejects_invalid_cloud_provider(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train", json={"cloud_provider": "aws"})
    assert response.status_code == 422


def test_train_endpoint_rejects_zero_max_gaussians(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train", json={"max_gaussians": 0})
    assert response.status_code == 422


def test_train_endpoint_rejects_negative_iterations(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/train", json={"iterations": -100})
    assert response.status_code == 422


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

    kill_mock = MagicMock()
    monkeypatch.setattr(process_registry, "kill_process_tree", kill_mock)
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    process = MagicMock()
    process.poll.return_value = None
    process_registry.register(job_id, "training", process)

    response = client.post(f"/jobs/{job_id}/train/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    kill_mock.assert_called_once_with(process)
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
