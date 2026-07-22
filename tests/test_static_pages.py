def test_index_page_contains_job_list_and_new_job_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="job-list"' in response.text
    assert 'href="/new-job.html"' in response.text
    assert "/js/app.js" in response.text
