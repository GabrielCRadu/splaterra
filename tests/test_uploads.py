import pytest

from app.uploads import (
    UploadError,
    complete_upload,
    received_chunks,
    save_chunk,
    start_upload,
)


def test_upload_round_trip(job_store):
    job = job_store.create_job("Test")

    upload_id = start_upload(job.id, "video.mp4", total_chunks=3)
    save_chunk(job.id, upload_id, 0, b"AAA")
    save_chunk(job.id, upload_id, 2, b"CCC")
    save_chunk(job.id, upload_id, 1, b"BBB")

    assert received_chunks(job.id, upload_id) == [0, 1, 2]

    destination = complete_upload(job_store, job.id, upload_id)
    assert destination.read_bytes() == b"AAABBBCCC"

    fetched = job_store.get_job(job.id)
    assert fetched.raw_video_path == str(destination)


def test_complete_upload_fails_when_chunks_missing(job_store):
    job = job_store.create_job("Test")

    upload_id = start_upload(job.id, "video.mp4", total_chunks=2)
    save_chunk(job.id, upload_id, 0, b"AAA")

    with pytest.raises(UploadError):
        complete_upload(job_store, job.id, upload_id)


def test_save_chunk_unknown_upload_raises(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(UploadError):
        save_chunk(job.id, "no-such-upload", 0, b"AAA")


def test_complete_upload_missing_chunks_records_failed_stage(job_store):
    job = job_store.create_job("Test")

    upload_id = start_upload(job.id, "video.mp4", total_chunks=2)
    save_chunk(job.id, upload_id, 0, b"AAA")

    with pytest.raises(UploadError):
        complete_upload(job_store, job.id, upload_id)

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


def test_complete_upload_unknown_upload_records_failed_stage(job_store):
    job = job_store.create_job("Test")

    with pytest.raises(UploadError):
        complete_upload(job_store, job.id, "no-such-upload")

    stage = job_store.get_stage(job.id, "ingestion")
    assert stage is not None
    assert stage.status == "failed"
    assert stage.error is not None


def test_start_upload_rejects_absolute_path_filename(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(UploadError):
        start_upload(job.id, "/etc/passwd", total_chunks=1)


def test_start_upload_rejects_path_traversal_filename(job_store):
    job = job_store.create_job("Test")
    with pytest.raises(UploadError):
        start_upload(job.id, "../../evil.txt", total_chunks=1)


def test_start_upload_accepts_normal_filename(job_store):
    job = job_store.create_job("Test")
    upload_id = start_upload(job.id, "video.mp4", total_chunks=1)
    save_chunk(job.id, upload_id, 0, b"AAA")

    destination = complete_upload(job_store, job.id, upload_id)
    assert destination.name == "video.mp4"
    assert destination.read_bytes() == b"AAA"
