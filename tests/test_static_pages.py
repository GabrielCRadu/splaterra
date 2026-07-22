def test_index_page_contains_job_list_and_new_job_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="job-list"' in response.text
    assert 'href="/new-job.html"' in response.text
    assert "/js/app.js" in response.text


def test_new_job_page_contains_local_and_upload_forms(client):
    response = client.get("/new-job.html")
    assert response.status_code == 200
    assert 'id="local-path"' in response.text
    assert 'id="file-input"' in response.text


def test_job_page_contains_stage_list_and_action_buttons(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="stage-list"' in response.text
    assert 'id="btn-extract"' in response.text
    assert 'id="btn-recommend"' in response.text


def test_frames_page_contains_frame_grid(client):
    response = client.get("/frames.html")
    assert response.status_code == 200
    assert 'id="frame-grid"' in response.text


def test_recommendation_page_served(client):
    response = client.get("/recommendation.html")
    assert response.status_code == 200
    assert 'id="content"' in response.text


def test_job_page_contains_align_button(client):
    response = client.get("/job.html")
    assert response.status_code == 200
    assert 'id="btn-align"' in response.text


def test_alignment_page_served(client):
    response = client.get("/alignment.html")
    assert response.status_code == 200
    assert 'id="content"' in response.text
