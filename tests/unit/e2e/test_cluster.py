"""Unit tests for e2e cluster utilities."""

import pytest

from tests.e2e.utils import cluster


def test_wait_for_running_pod_raises_runtime_error_when_containers_stay_unready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make app-pod readiness timeouts eligible for the setup retry."""
    attempts = iter((True, True, False))
    monkeypatch.setattr(
        cluster,
        "retry_until_timeout_or_success",
        lambda *_args, **_kwargs: next(attempts),
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for containers"):
        cluster.wait_for_running_pod()
