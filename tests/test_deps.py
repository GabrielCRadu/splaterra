from unittest.mock import MagicMock

from app import deps, process_registry


def test_cancel_or_force_fail_kills_a_live_registered_process(job_store, monkeypatch):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "training", "running", started_at="t0")
    process = MagicMock()
    process.poll.return_value = None
    kill_mock = MagicMock()
    monkeypatch.setattr(process_registry, "kill_process_tree", kill_mock)
    process_registry.register(job.id, "training", process)

    result = deps.cancel_or_force_fail(job_store, job.id, "training")

    assert result is True
    kill_mock.assert_called_once_with(process)
    # A live kill defers to the service's own except-block to record
    # "failed" (see process_registry.was_cancelled()) - this path must not
    # also force-write the stage itself, or it'd race/duplicate that.
    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "running"
    process_registry.unregister(job.id, "training")


def test_cancel_or_force_fail_clears_an_orphaned_running_stage(job_store):
    # The real incident this defends against: process_registry is in-memory
    # and is wiped by a server restart, so a stage stuck on "running" from
    # before that restart has no registered process to find or kill - it
    # must still get cleared, or the job is stuck forever (can't retry,
    # can't even delete it).
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "ingestion", "running", started_at="t0")

    result = deps.cancel_or_force_fail(job_store, job.id, "ingestion")

    assert result is True
    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "failed"
    assert "restart" in stage.error


def test_cancel_or_force_fail_returns_false_when_stage_not_running(job_store):
    job = job_store.create_job("Test")
    job_store.upsert_stage(job.id, "training", "done", finished_at="t1")

    result = deps.cancel_or_force_fail(job_store, job.id, "training")

    assert result is False
    stage = job_store.get_stage(job.id, "training")
    assert stage.status == "done"  # untouched


def test_cancel_or_force_fail_returns_false_when_stage_never_existed(job_store):
    job = job_store.create_job("Test")

    result = deps.cancel_or_force_fail(job_store, job.id, "alignment")

    assert result is False
    assert job_store.get_stage(job.id, "alignment") is None


def test_run_with_new_store_calls_func_with_fresh_store(job_store):
    job = job_store.create_job("Test")
    calls = []

    def fake_service(store, job_id, extra):
        calls.append((store is not None, job_id, extra))

    deps.run_with_new_store(fake_service, job.id, "extra-arg")

    assert calls == [(True, job.id, "extra-arg")]


def test_run_with_new_store_supports_kwargs(job_store):
    job = job_store.create_job("Test")
    calls = []

    def fake_service(store, job_id, *, option=None):
        calls.append((job_id, option))

    deps.run_with_new_store(fake_service, job.id, option="value")

    assert calls == [(job.id, "value")]


def test_run_with_new_store_swallows_exceptions(job_store):
    def failing_service(store, job_id):
        raise RuntimeError("boom")

    deps.run_with_new_store(failing_service, "some-job-id")  # must not raise


def test_run_with_new_store_logs_swallowed_exception(job_store, caplog):
    def failing_service(store, job_id):
        raise RuntimeError("boom")

    with caplog.at_level("ERROR"):
        deps.run_with_new_store(failing_service, "some-job-id")

    assert "boom" in caplog.text
