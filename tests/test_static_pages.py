from pathlib import Path


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


def test_job_page_restarts_polling_after_each_action():
    # Regression test: polling used to be a one-shot pollStages() call that
    # permanently stopped itself once no stage was "running". If a user
    # started a new stage (e.g. clicked "align") after polling had already
    # auto-stopped, the UI would show a single refresh() snapshot and then
    # never update again, even after the background job finished - it would
    # look stuck on "running" forever. The fix: each action button restarts
    # polling by reassigning stopPolling = startPolling() after its apiPost.
    content = (Path(__file__).resolve().parent.parent / "static" / "job.html").read_text(
        encoding="utf-8"
    )
    # 1 initial assignment (`let stopPolling = startPolling()`) + 3 button
    # handlers (extract/align/recommend) each reassigning it after apiPost.
    assert content.count("stopPolling = startPolling()") == 4
