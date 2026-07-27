def test_extract_frames_endpoint_returns_202_and_invokes_extraction(client, monkeypatch):
    from app.api import frames as frames_api

    calls = []

    def fake_extract_frames(store, job_id, *, selection_method, num_frames, fps, interval_seconds, ai_mask):
        calls.append((job_id, selection_method, num_frames, fps, interval_seconds, ai_mask))
        return 5

    monkeypatch.setattr(frames_api, "extract_frames", fake_extract_frames)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "num_frames": 100, "fps": 5},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, "batched", 100, 5, 1.0, False)]


def test_extract_frames_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/extract-frames", json={})
    assert response.status_code == 404


def test_extract_frames_endpoint_rejects_second_request_while_already_running(client):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "frame_extraction", "running", started_at="2026-01-01T00:00:00Z")
    conn.close()

    response = client.post(f"/jobs/{job_id}/extract-frames", json={})

    assert response.status_code == 409


def test_cancel_extract_frames_endpoint_unknown_job_returns_404(client):
    response = client.post("/jobs/does-not-exist/extract-frames/cancel")
    assert response.status_code == 404


def test_cancel_extract_frames_endpoint_reports_false_when_nothing_running(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/extract-frames/cancel")
    assert response.status_code == 200
    assert response.json() == {"cancelled": False}


def test_cancel_extract_frames_endpoint_kills_registered_process(client):
    from unittest.mock import MagicMock

    from app import process_registry

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    process = MagicMock()
    process.poll.return_value = None
    process_registry.register(job_id, "frame_extraction", process)

    response = client.post(f"/jobs/{job_id}/extract-frames/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    process.kill.assert_called_once()
    process_registry.unregister(job_id, "frame_extraction")


def test_import_frames_endpoint_returns_202_and_invokes_import(client, monkeypatch, tmp_path):
    from app.api import frames as frames_api

    calls = []

    def fake_import_local_frames(store, job_id, source_dir):
        calls.append((job_id, source_dir))
        return 5

    monkeypatch.setattr(frames_api, "import_local_frames", fake_import_local_frames)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/import-frames", json={"source_path": str(tmp_path / "frames")}
    )
    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, tmp_path / "frames")]


def test_import_frames_endpoint_unknown_job_returns_404(client, tmp_path):
    response = client.post(
        "/jobs/does-not-exist/import-frames", json={"source_path": str(tmp_path)}
    )
    assert response.status_code == 404


def test_list_frames_returns_empty_when_not_extracted_yet(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/frames")
    assert response.status_code == 200
    assert response.json() == {"frames": []}


def test_list_frames_unknown_job_returns_404(client):
    response = client.get("/jobs/does-not-exist/frames")
    assert response.status_code == 404


def test_list_frames_returns_filenames_from_frames_dir(client):
    from app import db
    from app.job_store import JobStore
    from app.paths import frames_dir

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    directory = frames_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "0001.jpg").write_bytes(b"fake")
    (directory / "0002.jpg").write_bytes(b"fake")
    JobStore(conn).update_job(job_id, frames_dir=str(directory))
    conn.close()

    response = client.get(f"/jobs/{job_id}/frames")
    assert response.status_code == 200
    assert response.json() == {"frames": ["0001.jpg", "0002.jpg"]}


def test_list_frames_excludes_non_image_files(client):
    # Regression test: sharp-frames writes a selected_metadata.json file into
    # the same output directory as the extracted frames. "selected_metadata.json"
    # sorts alphabetically after any "frame_XXXXX.jpg" name ('s' > 'f'), so an
    # unfiltered glob() returned it as the last "frame" - the frontend then
    # tried to render it as an <img>, showing a broken/missing image for what
    # looked like the last extracted frame, every single time.
    from app import db
    from app.job_store import JobStore
    from app.paths import frames_dir

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    directory = frames_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "frame_00001.jpg").write_bytes(b"fake")
    (directory / "frame_00002.jpg").write_bytes(b"fake")
    (directory / "selected_metadata.json").write_text("{}")
    JobStore(conn).update_job(job_id, frames_dir=str(directory))
    conn.close()

    response = client.get(f"/jobs/{job_id}/frames")
    assert response.status_code == 200
    assert response.json() == {"frames": ["frame_00001.jpg", "frame_00002.jpg"]}


def test_get_frame_returns_file_content(client):
    from app import db
    from app.job_store import JobStore
    from app.paths import frames_dir

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    directory = frames_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "0001.jpg").write_bytes(b"fake-jpeg-bytes")
    JobStore(conn).update_job(job_id, frames_dir=str(directory))
    conn.close()

    response = client.get(f"/jobs/{job_id}/frames/0001.jpg")
    assert response.status_code == 200
    assert response.content == b"fake-jpeg-bytes"


def test_get_frame_unknown_file_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/frames/does-not-exist.jpg")
    assert response.status_code == 404


def test_get_frame_rejects_path_traversal(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/frames/..%2F..%2Fapp%2Fmain.py")
    assert response.status_code in (400, 404)


def test_get_frame_rejects_bare_dotdot_filename(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.get(f"/jobs/{job_id}/frames/%2E%2E")
    assert response.status_code == 400
