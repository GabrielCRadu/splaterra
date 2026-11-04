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
    assert "Splaterra" in response.text


def test_static_css_served(client):
    response = client.get("/css/app.css")
    assert response.status_code == 200
    assert "--bg" in response.text


def test_static_js_served(client):
    response = client.get("/js/app.js")
    assert response.status_code == 200
    assert "apiGet" in response.text


def test_static_assets_force_revalidation_instead_of_stale_cache(client):
    # Regression test: the UI shell changes on nearly every dev iteration,
    # but testing happens from a phone over Tailscale where a hard-refresh
    # is easy to forget - a stale cached JS file (e.g. missing a function
    # that has actually been on disk for a while) looks exactly like a real
    # bug. Cache-Control: no-cache forces ETag revalidation on every load.
    for path in ("/", "/job.html", "/css/app.css", "/js/app.js"):
        response = client.get(path)
        assert response.headers["cache-control"] == "no-cache", path
