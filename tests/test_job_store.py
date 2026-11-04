def test_create_and_get_job(job_store):
    job = job_store.create_job("Curte interioara")
    fetched = job_store.get_job(job.id)
    assert fetched is not None
    assert fetched.name == "Curte interioara"
    assert fetched.raw_video_path is None
    assert fetched.device_type == "standard"


def test_create_job_with_device_type(job_store):
    job = job_store.create_job("Zbor drona", device_type="dji_avata_360")
    fetched = job_store.get_job(job.id)
    assert fetched.device_type == "dji_avata_360"


def test_list_jobs_returns_newest_first(job_store):
    first = job_store.create_job("Primul")
    second = job_store.create_job("Al doilea")
    jobs = job_store.list_jobs()
    assert [job.id for job in jobs] == [second.id, first.id]


def test_delete_job_removes_job_and_its_stages(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "ingestion", "done")

    job_store.delete_job(job.id)

    assert job_store.get_job(job.id) is None
    assert job_store.get_stage(job.id, "ingestion") is None


def test_update_job_sets_fields(job_store):
    job = job_store.create_job("Test")
    job_store.update_job(job.id, raw_video_path="/data/raw/video.mp4", frame_count=42)
    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == "/data/raw/video.mp4"
    assert fetched.frame_count == 42


def test_upsert_stage_creates_then_updates(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "ingestion", "running", started_at="t0")
    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "running"
    assert stage.started_at == "t0"

    job_store.upsert_stage(job.id, "ingestion", "done", finished_at="t1")
    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "done"
    assert stage.started_at == "t0"
    assert stage.finished_at == "t1"


def test_try_start_stage_claims_fresh_stage(job_store):
    job = job_store.create_job("Test")
    claimed = job_store.try_start_stage(job.id, "training", "t0")
    assert claimed is True
    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "running"
    assert stage.started_at == "t0"


def test_try_start_stage_rejects_when_already_running(job_store):
    job = job_store.create_job("Test")
    assert job_store.try_start_stage(job.id, "training", "t0") is True
    assert job_store.try_start_stage(job.id, "training", "t1") is False
    # the losing attempt must not have clobbered the winner's started_at
    stage = job_store.get_stage(job.id, "training")
    assert stage.started_at == "t0"


def test_try_start_stage_succeeds_again_after_done(job_store):
    job = job_store.create_job("Test")
    assert job_store.try_start_stage(job.id, "training", "t0") is True
    job_store.upsert_stage(job.id, "training", "done", finished_at="t1")

    assert job_store.try_start_stage(job.id, "training", "t2") is True
    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "running"
    assert stage.started_at == "t2"
    assert stage.finished_at is None
    assert stage.error is None


def test_try_start_stage_succeeds_again_after_failed(job_store):
    job = job_store.create_job("Test")
    assert job_store.try_start_stage(job.id, "training", "t0") is True
    job_store.upsert_stage(job.id, "training", "failed", finished_at="t1", error="boom")

    assert job_store.try_start_stage(job.id, "training", "t2") is True
    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "running"
    assert stage.error is None


def test_try_start_stage_returns_false_for_deleted_job(job_store):
    job = job_store.create_job("Test")
    job_store.delete_job(job.id)
    assert job_store.try_start_stage(job.id, "training", "t0") is False


def test_try_start_stage_is_race_free_under_concurrent_connections(tmp_path):
    # Empirically verified against the original race: without the atomic
    # claim, multiple concurrent callers could all read a non-"running"
    # status before any of them wrote "running", letting all of them win.
    import threading

    from app import db
    from app.job_store import JobStore

    db_path = tmp_path / "race.db"
    setup_conn = db.connect(db_path)
    job = JobStore(setup_conn).create_job("Test")
    setup_conn.close()

    results = []
    results_lock = threading.Lock()

    def worker(i):
        conn = db.connect(db_path)
        store = JobStore(conn)
        won = store.try_start_stage(job.id, "training", f"start-{i}")
        with results_lock:
            results.append(won)
        conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1


def test_upsert_stage_sets_and_preserves_settings_json(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(
        job.id, "training", "running", started_at="t0", settings_json='{"iterations": 30000}'
    )
    stage = job_store.get_stage(job.id, "training")
    assert stage.settings_json == '{"iterations": 30000}'

    # A progress update that doesn't pass settings_json (the normal case for
    # every report_progress()/report_log() call during a run) must not wipe
    # it back to NULL - same COALESCE behavior as started_at/progress_percent.
    job_store.upsert_stage(job.id, "training", "running", progress_percent=50.0)
    stage = job_store.get_stage(job.id, "training")
    assert stage.settings_json == '{"iterations": 30000}'
    assert stage.progress_percent == 50.0


def test_upsert_stage_settings_json_defaults_to_none(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "alignment", "running", started_at="t0")
    stage = job_store.get_stage(job.id, "alignment")
    assert stage.settings_json is None


def test_upsert_stage_sets_and_preserves_eval_metrics_json(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "training", "running", started_at="t0")
    job_store.upsert_stage(
        job.id, "training", "done", finished_at="t1",
        eval_metrics_json='[{"iteration": 7000, "psnr": 24.1}]',
    )
    stage = job_store.get_stage(job.id, "training")
    assert stage.eval_metrics_json == '[{"iteration": 7000, "psnr": 24.1}]'

    # A later update that doesn't pass eval_metrics_json must not wipe it -
    # same COALESCE behavior as settings_json/progress_percent.
    job_store.upsert_stage(job.id, "training", "done", log="some log")
    stage = job_store.get_stage(job.id, "training")
    assert stage.eval_metrics_json == '[{"iteration": 7000, "psnr": 24.1}]'


def test_upsert_stage_eval_metrics_json_defaults_to_none(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "training", "running", started_at="t0")
    stage = job_store.get_stage(job.id, "training")
    assert stage.eval_metrics_json is None


def test_delete_stages_removes_only_the_given_stages(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "frame_extraction", "done")
    job_store.upsert_stage(job.id, "alignment", "done")
    job_store.upsert_stage(job.id, "training", "done")

    job_store.delete_stages(job.id, ["alignment", "training"])

    assert job_store.get_stage(job.id, "frame_extraction") is not None
    assert job_store.get_stage(job.id, "alignment") is None
    assert job_store.get_stage(job.id, "training") is None


def test_delete_stages_with_empty_list_is_a_noop(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "alignment", "done")

    job_store.delete_stages(job.id, [])

    assert job_store.get_stage(job.id, "alignment") is not None


def test_update_job_sets_recommendation_json(job_store):
    job = job_store.create_job("Test")
    assert job.recommendation_json is None

    job_store.update_job(job.id, recommendation_json='{"verdict": {"use_cloud": false}}')
    fetched = job_store.get_job(job.id)
    assert fetched.recommendation_json == '{"verdict": {"use_cloud": false}}'


def test_list_stages_returns_all_stages_for_job(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "ingestion", "done", started_at="t0", finished_at="t1")
    job_store.upsert_stage(job.id, "frame_extraction", "running", started_at="t2")

    stages = job_store.list_stages(job.id)

    assert {s.stage for s in stages} == {"ingestion", "frame_extraction"}
    ingestion_stage = next(s for s in stages if s.stage == "ingestion")
    assert ingestion_stage.status == "done"


def test_list_stages_empty_for_job_with_no_stages(job_store):
    job = job_store.create_job("Test")
    assert job_store.list_stages(job.id) == []


def test_update_job_sets_colmap_sparse_and_undistorted_images_dirs(job_store):
    job = job_store.create_job("Test")
    job_store.update_job(
        job.id,
        colmap_sparse_dir="/data/alignment/sparse/0",
        undistorted_images_dir="/data/alignment/images",
    )
    fetched = job_store.get_job(job.id)
    assert fetched.colmap_sparse_dir == "/data/alignment/sparse/0"
    assert fetched.undistorted_images_dir == "/data/alignment/images"


def test_update_job_sets_splat_path(job_store):
    job = job_store.create_job("Test")
    assert job.splat_path is None

    job_store.update_job(job.id, splat_path="/data/training/output.ply")
    fetched = job_store.get_job(job.id)
    assert fetched.splat_path == "/data/training/output.ply"


def test_update_job_sets_export_path_and_format(job_store):
    job = job_store.create_job("Test")
    assert job.export_path is None
    assert job.export_format is None

    job_store.update_job(job.id, export_path="/data/training/export.spz", export_format="spz")
    fetched = job_store.get_job(job.id)
    assert fetched.export_path == "/data/training/export.spz"
    assert fetched.export_format == "spz"


def test_upsert_stage_sets_and_preserves_progress_percent(job_store):
    job = job_store.create_job("Test")

    job_store.upsert_stage(job.id, "frame_extraction", "running", started_at="t0", progress_percent=12.5)
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 12.5

    # A later call that doesn't pass progress_percent must not reset it to None
    job_store.upsert_stage(job.id, "frame_extraction", "running")
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 12.5

    # An explicit new value does update it
    job_store.upsert_stage(job.id, "frame_extraction", "running", progress_percent=47.0)
    stage = job_store.get_stage(job.id, "frame_extraction")
    assert stage.progress_percent == 47.0
