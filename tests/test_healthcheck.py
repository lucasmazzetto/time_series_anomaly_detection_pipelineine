from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app


client = TestClient(app)


class _CountQuery:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar(self) -> int:
        return self._value


class _DummySession:
    def __init__(self, series_count: int) -> None:
        self._series_count = series_count

    def query(self, *_args, **_kwargs) -> _CountQuery:
        return _CountQuery(self._series_count)


@pytest.fixture(autouse=True)
def _test_overrides():
    def _override_session():
        yield _DummySession(series_count=2)

    app.dependency_overrides[get_session] = _override_session
    yield
    app.dependency_overrides.pop(get_session, None)


def test_healthcheck_returns_series_count_and_latency_metrics():
    with patch(
        "app.services.healthcheck.LatencyRecord.get_latencies",
        side_effect=[[10.0, 20.0], [30.0, 40.0]],
    ):
        response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 2
    assert payload["training_latency_ms"] == {"avg": 15.0, "p95": 20.0}
    assert payload["inference_latency_ms"] == {"avg": 35.0, "p95": 40.0}


def test_healthcheck_returns_zero_metrics_when_no_requests():
    with patch(
        "app.services.healthcheck.LatencyRecord.get_latencies",
        side_effect=[[], []],
    ):
        response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 2
    assert payload["training_latency_ms"] == {"avg": 0.0, "p95": 0.0}
    assert payload["inference_latency_ms"] == {"avg": 0.0, "p95": 0.0}


def test_healthcheck_returns_503_when_redis_read_fails():
    with patch(
        "app.services.healthcheck.LatencyRecord.get_latencies",
        side_effect=RuntimeError("redis unavailable"),
    ):
        response = client.get("/healthcheck")

    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "Telemetry backend unavailable for healthcheck."
    )
