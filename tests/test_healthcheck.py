import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app
from app.middleware.latency import _update_latency_cache, reset_latency_cache


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

    reset_latency_cache()
    app.dependency_overrides[get_session] = _override_session
    yield
    app.dependency_overrides.pop(get_session, None)
    reset_latency_cache()


def test_healthcheck_returns_series_count_and_latency_metrics():
    _update_latency_cache("/fit/series_a", 10.0)
    _update_latency_cache("/fit/series_b", 20.0)
    _update_latency_cache("/predict/series_a", 30.0)
    _update_latency_cache("/predict/series_a", 40.0)

    response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 2
    assert payload["training_latency_ms"] == {"avg": 15.0, "p95": 20.0}
    assert payload["inference_latency_ms"] == {"avg": 35.0, "p95": 40.0}


def test_healthcheck_returns_zero_metrics_when_no_requests():
    response = client.get("/healthcheck")

    assert response.status_code == 200
    payload = response.json()
    assert payload["series_trained"] == 2
    assert payload["training_latency_ms"] == {"avg": 0.0, "p95": 0.0}
    assert payload["inference_latency_ms"] == {"avg": 0.0, "p95": 0.0}
