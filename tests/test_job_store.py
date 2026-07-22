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
