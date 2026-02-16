from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app
from app.schemas import PredictResponse


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


def test_predict_endpoint_accepts_prefixed_version_and_sanitizes():
    """@brief Verify prediction endpoint sanitizes querystring version.

    @details Ensures versions like `v12` are converted to integer `12`
    before being passed to the prediction service.
    """
    series_id = "series_predict_01"
    payload = {"timestamp": "1700000000", "value": 10.5}

    with patch(
        "app.api.predict.AnomalyDetectionPredictionService.predict",
        return_value=PredictResponse(anomaly=True, model_version="12"),
    ) as predict_mock:
        response = client.post(f"/predict/{series_id}?version=v12", json=payload)

    assert response.status_code == 200
    assert response.json() == {"anomaly": True, "model_version": "12"}

    called_series_id, called_version, called_payload = predict_mock.call_args[0]
    assert called_series_id == series_id
    assert called_version == 12
    assert called_payload.timestamp == "1700000000"
    assert called_payload.value == 10.5


def test_predict_endpoint_defaults_version_to_zero():
    """@brief Verify prediction endpoint defaults missing version to zero.

    @details Ensures omitted querystring version resolves to `0`, allowing
    the service layer to select the latest model.
    """
    series_id = "series_predict_default"
    payload = {"timestamp": "1700000001", "value": 8.0}

    with patch(
        "app.api.predict.AnomalyDetectionPredictionService.predict",
        return_value=PredictResponse(anomaly=False, model_version="5"),
    ) as predict_mock:
        response = client.post(f"/predict/{series_id}", json=payload)

    assert response.status_code == 200
    assert response.json() == {"anomaly": False, "model_version": "5"}

    _, called_version, _ = predict_mock.call_args[0]
    assert called_version == 0


def test_predict_endpoint_rejects_non_numeric_version():
    """@brief Verify invalid version querystrings return 422.

    @details Ensures non-numeric values (after sanitization) fail validation
    and are rejected before service execution.
    """
    series_id = "series_predict_invalid_version"
    payload = {"timestamp": "1700000002", "value": 7.5}

    with patch(
        "app.api.predict.AnomalyDetectionPredictionService.predict"
    ) as predict_mock:
        response = client.post(f"/predict/{series_id}?version=batata", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(
        "Version must contain at least one digit." in error["msg"]
        for error in errors
    )
    predict_mock.assert_not_called()
