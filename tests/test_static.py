def test_health_endpoint_still_works_after_static_mount(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_jobs_api_still_works_after_static_mount(client):
    response = client.post("/jobs", json={"name": "Test"})
    assert response.status_code == 200


def test_static_index_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Splat Prep" in response.text


def test_static_css_served(client):
    response = client.get("/css/app.css")
    assert response.status_code == 200
    assert "--bg" in response.text


def test_static_js_served(client):
    response = client.get("/js/app.js")
    assert response.status_code == 200
    assert "apiGet" in response.text
