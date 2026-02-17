from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app
from app.middleware.latency import (
    _update_latency_cache,
    get_latency_cache,
    reset_latency_cache,
)
from app.schemas.predict_response import PredictResponse
from app.schemas.train_response import TrainResponse


client = TestClient(app)


@pytest.fixture(autouse=True)
def _test_overrides():
    class DummySession:
        def rollback(self) -> None:
            pass

    def _override_session():
        yield DummySession()

    reset_latency_cache()
    app.dependency_overrides[get_session] = _override_session
    yield
    app.dependency_overrides.pop(get_session, None)
    reset_latency_cache()


def test_latency_cache_is_separated_by_endpoint_group():
    series_id = "latency_series"
    fit_payload = {
        "timestamps": [1700000000, 1700000001, 1700000002],
        "values": [1.0, 1.1, 1.2],
    }
    predict_payload = {"timestamp": "1700000010", "value": 1.4}

    with patch(
        "app.api.train.TrainService.train",
        return_value=TrainResponse(
            series_id=series_id,
            message="Training successfully started.",
            success=True,
        ),
    ), patch(
        "app.api.predict.PredictService.predict",
        return_value=PredictResponse(anomaly=False, model_version="1"),
    ):
        fit_response = client.post(f"/fit/{series_id}", json=fit_payload)
        predict_response = client.post(f"/predict/{series_id}", json=predict_payload)

    assert fit_response.status_code == 200
    assert predict_response.status_code == 200

    cache = get_latency_cache()
    assert cache["train"]["count"] == 1
    assert cache["predict"]["count"] == 1
    assert len(cache["train"]["latencies_ms"]) == 1
    assert len(cache["predict"]["latencies_ms"]) == 1
    assert cache["train"]["avg_ms"] > 0.0
    assert cache["predict"]["avg_ms"] > 0.0
    assert cache["train"]["p95_ms"] > 0.0
    assert cache["predict"]["p95_ms"] > 0.0


def test_latency_cache_updates_average_for_multiple_requests():
    series_id = "latency_average"
    payload = {
        "timestamps": [1700000100, 1700000101, 1700000102],
        "values": [2.0, 2.1, 2.2],
    }

    with patch(
        "app.api.train.TrainService.train",
        return_value=TrainResponse(
            series_id=series_id,
            message="Training successfully started.",
            success=True,
        ),
    ):
        first = client.post(f"/fit/{series_id}", json=payload)
        second = client.post(f"/fit/{series_id}", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    cache = get_latency_cache()
    assert cache["train"]["count"] == 2
    assert len(cache["train"]["latencies_ms"]) == 2
    assert cache["train"]["total_ms"] == pytest.approx(
        sum(cache["train"]["latencies_ms"])
    )
    assert cache["train"]["avg_ms"] == pytest.approx(
        cache["train"]["total_ms"] / cache["train"]["count"]
    )
    assert cache["train"]["p95_ms"] == max(cache["train"]["latencies_ms"])
    assert cache["predict"]["count"] == 0


def test_latency_cache_updates_p95_with_nearest_rank():
    _update_latency_cache("/fit/latency_p95", 10.0)
    _update_latency_cache("/fit/latency_p95", 20.0)
    _update_latency_cache("/fit/latency_p95", 30.0)
    _update_latency_cache("/fit/latency_p95", 40.0)
    _update_latency_cache("/fit/latency_p95", 50.0)
    _update_latency_cache("/fit/latency_p95", 60.0)
    _update_latency_cache("/fit/latency_p95", 70.0)
    _update_latency_cache("/fit/latency_p95", 80.0)
    _update_latency_cache("/fit/latency_p95", 90.0)
    _update_latency_cache("/fit/latency_p95", 100.0)

    cache = get_latency_cache()
    assert cache["train"]["count"] == 10
    assert cache["train"]["p95_ms"] == pytest.approx(100.0)
