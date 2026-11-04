def test_extract_frames_endpoint_returns_202_and_invokes_extraction(client, monkeypatch):
    from app.api import frames as frames_api

    calls = []

    def fake_extract_frames(
        store, job_id, *, selection_method, num_frames, fps, interval_seconds, ai_mask, skip_up_frames,
        layout, camera_count, resolution, nadir_mask,
    ):
        calls.append((job_id, selection_method, num_frames, fps, interval_seconds, ai_mask, skip_up_frames))
        return 5

    monkeypatch.setattr(frames_api, "extract_frames", fake_extract_frames)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"selection_method": "batched", "num_frames": 100, "fps": 5},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    assert calls == [(job_id, "batched", 100, 5, 1.0, False, False)]


def test_extract_frames_endpoint_forwards_360_settings(client, monkeypatch):
    from app.api import frames as frames_api

    calls = []

    def fake_extract_frames(
        store, job_id, *, selection_method, num_frames, fps, interval_seconds, ai_mask, skip_up_frames,
        layout, camera_count, resolution, nadir_mask,
    ):
        calls.append((layout, camera_count, resolution, nadir_mask))
        return 5

    monkeypatch.setattr(frames_api, "extract_frames", fake_extract_frames)

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    response = client.post(
        f"/jobs/{job_id}/extract-frames",
        json={"layout": "fibonacci", "camera_count": 12, "resolution": 3072, "nadir_mask": True},
    )
    assert response.status_code == 202
    assert calls == [("fibonacci", 12, 3072, True)]


def test_extract_frames_endpoint_rejects_invalid_layout(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/extract-frames", json={"layout": "sphere"})
    assert response.status_code == 422


def test_extract_frames_endpoint_rejects_camera_count_out_of_range(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/extract-frames", json={"camera_count": 1})
    assert response.status_code == 422


def test_extract_frames_endpoint_rejects_zero_num_frames(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/extract-frames", json={"num_frames": 0})
    assert response.status_code == 422


def test_extract_frames_endpoint_rejects_negative_interval(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.post(f"/jobs/{job_id}/extract-frames", json={"interval_seconds": -1.0})
    assert response.status_code == 422


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


def test_cancel_extract_frames_endpoint_kills_registered_process(client, monkeypatch):
    from unittest.mock import MagicMock

    from app import process_registry

    kill_mock = MagicMock()
    monkeypatch.setattr(process_registry, "kill_process_tree", kill_mock)
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    process = MagicMock()
    process.poll.return_value = None
    process_registry.register(job_id, "frame_extraction", process)

    response = client.post(f"/jobs/{job_id}/extract-frames/cancel")

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}
    kill_mock.assert_called_once_with(process)
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


def test_import_frames_endpoint_rejects_while_extraction_already_running(client, tmp_path):
    from app import db
    from app.job_store import JobStore

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    conn = db.connect()
    JobStore(conn).upsert_stage(job_id, "frame_extraction", "running", started_at="2026-01-01T00:00:00Z")
    conn.close()

    response = client.post(
        f"/jobs/{job_id}/import-frames", json={"source_path": str(tmp_path)}
    )

    assert response.status_code == 409


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


def test_delete_frame_removes_file_and_decrements_count(client):
    from app import db
    from app.job_store import JobStore
    from app.paths import frames_dir

    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    conn = db.connect()
    directory = frames_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "0001.jpg").write_bytes(b"fake")
    (directory / "0002.jpg").write_bytes(b"fake")
    JobStore(conn).update_job(job_id, frames_dir=str(directory), frame_count=2)
    conn.close()

    response = client.delete(f"/jobs/{job_id}/frames/0001.jpg")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}

    assert not (directory / "0001.jpg").exists()
    assert (directory / "0002.jpg").exists()

    listing = client.get(f"/jobs/{job_id}/frames")
    assert listing.json() == {"frames": ["0002.jpg"]}

    job = client.get(f"/jobs/{job_id}").json()
    assert job["frame_count"] == 1


def test_delete_frame_unknown_job_returns_404(client):
    response = client.delete("/jobs/does-not-exist/frames/0001.jpg")
    assert response.status_code == 404


def test_delete_frame_unknown_file_returns_404(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.delete(f"/jobs/{job_id}/frames/does-not-exist.jpg")
    assert response.status_code == 404


def test_delete_frame_rejects_path_traversal(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.delete(f"/jobs/{job_id}/frames/..%2F..%2Fapp%2Fmain.py")
    # %2F decodes to a real "/", so this no longer matches the single-segment
    # {filename} route at all - like the equivalent GET test, it falls through
    # to the StaticFiles catch-all mount, which only registers GET/HEAD
    # (hence 405 here instead of GET's 404) and blocks ".." itself either way.
    assert response.status_code in (400, 404, 405)


def test_delete_frame_rejects_bare_dotdot_filename(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    response = client.delete(f"/jobs/{job_id}/frames/%2E%2E")
    assert response.status_code == 400
