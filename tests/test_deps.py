from app import deps


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
