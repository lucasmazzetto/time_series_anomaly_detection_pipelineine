from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app
from app.middleware.latency import _target_from_path
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

    app.dependency_overrides[get_session] = _override_session
    yield
    app.dependency_overrides.pop(get_session, None)


def test_target_from_path_maps_known_groups():
    assert _target_from_path("/fit/series_01") == "train"
    assert _target_from_path("/predict/series_01") == "predict"
    assert _target_from_path("/healthcheck") is None


def test_latency_middleware_pushes_train_and_predict_for_2xx():
    series_id = "latency_series"
    fit_payload = {
        "timestamps": [1700000000, 1700000001, 1700000002],
        "values": [1.0, 1.1, 1.2],
    }
    predict_payload = {"timestamp": "1700000010", "value": 1.4}

    with patch("app.middleware.latency.LatencyRecord.push_latency") as push_mock, patch(
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

    assert push_mock.call_count == 2
    first_target = push_mock.call_args_list[0].args[0]
    first_latency = push_mock.call_args_list[0].args[1]
    second_target = push_mock.call_args_list[1].args[0]
    second_latency = push_mock.call_args_list[1].args[1]

    assert first_target == "train"
    assert second_target == "predict"
    assert isinstance(first_latency, float) and first_latency > 0.0
    assert isinstance(second_latency, float) and second_latency > 0.0


def test_latency_middleware_ignores_non_2xx_responses():
    series_id = "latency_average"
    valid_payload = {
        "timestamps": [1700000100, 1700000101, 1700000102],
        "values": [2.0, 2.1, 2.2],
    }
    invalid_payload = {
        "timestamps": [1700000300, 1700000301],
        "values": [4.0, 4.1],
    }

    with patch("app.middleware.latency.LatencyRecord.push_latency") as push_mock, patch(
        "app.api.train.TrainService.train",
        return_value=TrainResponse(
            series_id=series_id,
            message="Training successfully started.",
            success=True,
        ),
    ):
        success_response = client.post(f"/fit/{series_id}", json=valid_payload)
    error_response = client.post(f"/fit/{series_id}", json=invalid_payload)

    assert success_response.status_code == 200
    assert error_response.status_code == 422
    assert push_mock.call_count == 1
    assert push_mock.call_args.args[0] == "train"


def test_latency_middleware_swallow_redis_errors():
    series_id = "latency_redis_error"
    payload = {
        "timestamps": [1700000400, 1700000401, 1700000402],
        "values": [5.0, 5.1, 5.2],
    }

    with patch(
        "app.middleware.latency.LatencyRecord.push_latency",
        side_effect=RuntimeError("redis unavailable"),
    ), patch(
        "app.api.train.TrainService.train",
        return_value=TrainResponse(
            series_id=series_id,
            message="Training successfully started.",
            success=True,
        ),
    ):
        response = client.post(f"/fit/{series_id}", json=payload)

    assert response.status_code == 200
