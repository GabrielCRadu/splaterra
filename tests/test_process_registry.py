from unittest.mock import MagicMock

from app import process_registry


def _fake_process(running: bool = True) -> MagicMock:
    process = MagicMock()
    process.poll.return_value = None if running else 0
    return process


def test_cancel_kills_registered_running_process(monkeypatch):
    kill_mock = MagicMock()
    monkeypatch.setattr(process_registry, "kill_process_tree", kill_mock)
    process = _fake_process(running=True)
    process_registry.register("job-1", "alignment", process)

    result = process_registry.cancel("job-1", "alignment")

    assert result is True
    kill_mock.assert_called_once_with(process)
    assert process_registry.was_cancelled("job-1", "alignment") is True


def test_cancel_returns_false_when_nothing_registered():
    assert process_registry.cancel("no-such-job", "alignment") is False


def test_cancel_returns_false_when_process_already_finished(monkeypatch):
    kill_mock = MagicMock()
    monkeypatch.setattr(process_registry, "kill_process_tree", kill_mock)
    process = _fake_process(running=False)
    process_registry.register("job-2", "training", process)

    result = process_registry.cancel("job-2", "training")

    assert result is False
    kill_mock.assert_not_called()


def test_unregister_clears_process_and_cancelled_flag(monkeypatch):
    monkeypatch.setattr(process_registry, "kill_process_tree", MagicMock())
    process = _fake_process(running=True)
    process_registry.register("job-3", "alignment", process)
    process_registry.cancel("job-3", "alignment")

    process_registry.unregister("job-3", "alignment")

    assert process_registry.cancel("job-3", "alignment") is False
    assert process_registry.was_cancelled("job-3", "alignment") is False


def test_registering_again_clears_stale_cancelled_flag(monkeypatch):
    # A job re-running the same stage after a previous cancel shouldn't be
    # reported as "cancelled" for the new run just because the old flag stuck.
    monkeypatch.setattr(process_registry, "kill_process_tree", MagicMock())
    first_process = _fake_process(running=True)
    process_registry.register("job-4", "training", first_process)
    process_registry.cancel("job-4", "training")
    assert process_registry.was_cancelled("job-4", "training") is True

    second_process = _fake_process(running=True)
    process_registry.register("job-4", "training", second_process)

    assert process_registry.was_cancelled("job-4", "training") is False
