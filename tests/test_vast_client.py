from unittest.mock import MagicMock

import pytest
import requests

from app import vast_client
from app.remote_exec import PodConnection
from app.vast_client import VastAIError


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_body


def test_search_offers_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(VastAIError, match="VAST_API_KEY"):
        vast_client.search_offers("RTX_4090")


def test_search_offers_returns_offers_from_response(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    offers = [{"id": 111, "gpu_name": "RTX_4090", "dph_total": 0.35}]
    monkeypatch.setattr(
        vast_client.requests, "post", MagicMock(return_value=_FakeResponse({"offers": offers}))
    )

    result = vast_client.search_offers("RTX_4090")

    assert result == offers


def test_search_offers_wraps_request_errors(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client.requests, "post", MagicMock(side_effect=requests.ConnectionError("no route"))
    )

    with pytest.raises(VastAIError, match="no route"):
        vast_client.search_offers("RTX_4090")


def test_create_gpu_pod_uses_numeric_gpu_type_id_as_offer_id_directly(monkeypatch):
    # A caller who already knows the exact offer id (e.g. picked it from the
    # vast.ai dashboard) can skip the auto-search entirely.
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    search_mock = MagicMock()
    monkeypatch.setattr(vast_client, "search_offers", search_mock)
    put_mock = MagicMock(return_value=_FakeResponse({"success": True, "new_contract": 999}))
    monkeypatch.setattr(vast_client.requests, "put", put_mock)

    pod_id = vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="777")

    search_mock.assert_not_called()
    assert put_mock.call_args.args[0].endswith("/api/v0/asks/777")
    assert pod_id == "999"


def test_create_gpu_pod_searches_and_uses_cheapest_offer_for_gpu_name(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client, "search_offers", MagicMock(return_value=[{"id": 555, "dph_total": 0.2}])
    )
    put_mock = MagicMock(return_value=_FakeResponse({"success": True, "new_contract": 999}))
    monkeypatch.setattr(vast_client.requests, "put", put_mock)

    pod_id = vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="RTX_4090")

    assert put_mock.call_args.args[0].endswith("/api/v0/asks/555")
    assert pod_id == "999"


def test_create_gpu_pod_raises_when_no_offers_found(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(vast_client, "search_offers", MagicMock(return_value=[]))

    with pytest.raises(VastAIError, match="No rentable vast.ai offers"):
        vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="RTX_4090")


def test_create_gpu_pod_raises_when_response_not_success(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client.requests, "put", MagicMock(return_value=_FakeResponse({"success": False, "error": "boom"}))
    )

    with pytest.raises(VastAIError, match="rejected instance creation"):
        vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="777")


def test_create_gpu_pod_raises_when_no_new_contract(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client.requests, "put", MagicMock(return_value=_FakeResponse({"success": True}))
    )

    with pytest.raises(VastAIError, match="did not return an instance id"):
        vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="777")


def test_create_gpu_pod_wraps_request_errors(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client.requests, "put", MagicMock(side_effect=requests.Timeout("slow"))
    )

    with pytest.raises(VastAIError, match="slow"):
        vast_client.create_gpu_pod("test", image_name="img", gpu_type_id="777")


def test_wait_for_pod_ready_returns_connection_when_running(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    ready = {"instances": {"actual_status": "running", "ssh_host": "1.2.3.4", "ssh_port": 40001}}
    monkeypatch.setattr(vast_client.requests, "get", MagicMock(return_value=_FakeResponse(ready)))
    monkeypatch.setattr(vast_client.time, "sleep", lambda _: None)

    connection = vast_client.wait_for_pod_ready("777", timeout_seconds=30)

    assert connection == PodConnection(pod_id="777", ssh_host="1.2.3.4", ssh_port=40001)


def test_wait_for_pod_ready_polls_until_running(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    loading = _FakeResponse({"instances": {"actual_status": "loading"}})
    ready = _FakeResponse(
        {"instances": {"actual_status": "running", "ssh_host": "1.2.3.4", "ssh_port": 40001}}
    )
    monkeypatch.setattr(
        vast_client.requests, "get", MagicMock(side_effect=[loading, loading, ready])
    )
    monkeypatch.setattr(vast_client.time, "sleep", lambda _: None)

    connection = vast_client.wait_for_pod_ready("777", timeout_seconds=30)

    assert connection.ssh_port == 40001


def test_wait_for_pod_ready_raises_immediately_on_terminal_status(monkeypatch):
    # exited/unknown/offline never transition to running (docs.vast.ai) - a
    # real poll loop would otherwise burn the full timeout waiting for a
    # status that's never coming.
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    exited = _FakeResponse({"instances": {"actual_status": "exited"}})
    monkeypatch.setattr(vast_client.requests, "get", MagicMock(return_value=exited))
    monkeypatch.setattr(vast_client.time, "sleep", lambda _: None)

    with pytest.raises(VastAIError, match="exited"):
        vast_client.wait_for_pod_ready("777", timeout_seconds=30)


def test_wait_for_pod_ready_raises_on_timeout(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    loading = _FakeResponse({"instances": {"actual_status": "loading"}})
    monkeypatch.setattr(vast_client.requests, "get", MagicMock(return_value=loading))
    monkeypatch.setattr(vast_client.time, "sleep", lambda _: None)

    fake_clock = [0.0]

    def fake_monotonic():
        fake_clock[0] += 100.0
        return fake_clock[0]

    monkeypatch.setattr(vast_client.time, "monotonic", fake_monotonic)

    with pytest.raises(VastAIError, match="did not become SSH-ready"):
        vast_client.wait_for_pod_ready("777", timeout_seconds=30, poll_interval_seconds=0)


def test_terminate_pod_calls_delete(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    delete_mock = MagicMock(return_value=_FakeResponse({"success": True}))
    monkeypatch.setattr(vast_client.requests, "delete", delete_mock)

    vast_client.terminate_pod("777")

    assert delete_mock.call_args.args[0].endswith("/api/v0/instances/777")


def test_terminate_pod_wraps_request_errors(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "fake-key")
    monkeypatch.setattr(
        vast_client.requests, "delete", MagicMock(side_effect=requests.ConnectionError("no route"))
    )

    with pytest.raises(VastAIError, match="no route"):
        vast_client.terminate_pod("777")
