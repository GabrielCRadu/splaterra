from app import db


def test_connect_creates_schema(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "job_stages"} <= tables
    conn.close()
