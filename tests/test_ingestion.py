import pytest

from app.ingestion import IngestionError, import_local_file


def test_import_local_file_copies_and_records_job(job_store, tmp_path):
    job = job_store.create_job("Test")

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake video bytes")

    destination = import_local_file(job_store, job.id, source)

    assert destination.exists()
    assert destination.read_bytes() == b"fake video bytes"

    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == str(destination)

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage.status == "done"


def test_import_local_file_missing_source_raises(job_store, tmp_path):
    job = job_store.create_job("Test")

    with pytest.raises(IngestionError):
        import_local_file(job_store, job.id, tmp_path / "missing.mp4")
