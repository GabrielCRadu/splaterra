def test_create_and_get_job(job_store):
    job = job_store.create_job("Curte interioara")
    fetched = job_store.get_job(job.id)
    assert fetched is not None
    assert fetched.name == "Curte interioara"
    assert fetched.raw_video_path is None


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
