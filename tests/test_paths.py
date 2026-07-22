from app import paths


def test_job_dir_uses_data_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    assert paths.job_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123"
    assert paths.raw_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "raw"
    assert paths.frames_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "frames"
    assert paths.upload_chunks_dir("abc123", "up1") == (
        tmp_path.resolve() / "jobs" / "abc123" / "uploads" / "up1" / "chunks"
    )


def test_data_root_defaults_to_relative_data_dir(monkeypatch):
    monkeypatch.delenv("SPLATPREP_DATA_DIR", raising=False)
    from pathlib import Path

    assert paths.data_root() == Path("./data").resolve()


def test_alignment_dir_uses_data_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLATPREP_DATA_DIR", str(tmp_path))
    assert paths.alignment_dir("abc123") == tmp_path.resolve() / "jobs" / "abc123" / "alignment"
