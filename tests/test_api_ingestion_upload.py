def test_upload_round_trip_via_api(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]

    upload_id = client.post(
        f"/jobs/{job_id}/uploads", json={"filename": "video.mp4", "total_chunks": 2}
    ).json()["upload_id"]

    put_response = client.put(
        f"/jobs/{job_id}/uploads/{upload_id}/chunks/0", content=b"AAA"
    )
    assert put_response.status_code == 200
    client.put(f"/jobs/{job_id}/uploads/{upload_id}/chunks/1", content=b"BBB")

    chunks_response = client.get(f"/jobs/{job_id}/uploads/{upload_id}/chunks")
    assert chunks_response.json() == {"received_chunks": [0, 1]}

    complete_response = client.post(f"/jobs/{job_id}/uploads/{upload_id}/complete")
    assert complete_response.status_code == 200

    job = client.get(f"/jobs/{job_id}").json()
    assert job["raw_video_path"] is not None


def test_complete_upload_missing_chunks_returns_400(client):
    job_id = client.post("/jobs", json={"name": "Test"}).json()["id"]
    upload_id = client.post(
        f"/jobs/{job_id}/uploads", json={"filename": "video.mp4", "total_chunks": 2}
    ).json()["upload_id"]
    client.put(f"/jobs/{job_id}/uploads/{upload_id}/chunks/0", content=b"AAA")

    response = client.post(f"/jobs/{job_id}/uploads/{upload_id}/complete")
    assert response.status_code == 400
