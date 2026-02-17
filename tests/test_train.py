from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app
from app.schemas.train_response import TrainResponse


client = TestClient(app)

def make_timestamps(count: int, *, start: int = 1_700_000_000) -> list[int]:
    return list(range(start, start + count))


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


def test_fit_endpoint():
    """@brief Submit a valid time series to the fit endpoint.

    @details Expects a 200 response with the returned series_id matching the request and a success flag
    alongside a descriptive message.
    """
    series_id = "test_series_001"
    values = [1.0, 1.2, 1.1, 0.9, 1.0, 1.3]

    payload = {
        "timestamps": make_timestamps(6),
        "values": values
    }

    with patch(
        "app.api.train.TrainService.train",
        return_value=TrainResponse(
            series_id=series_id,
            message="Training successfully started.",
            success=True,
        ),
    ):
        response = client.post(f"/fit/{series_id}", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["series_id"] == series_id
    assert data["success"] is True
    assert data["message"] == "Training successfully started."


def test_fit_endpoint_validation_failure():
    """@brief Provide an undersized series to the fit endpoint to trigger validation.

    @details Sending too few points should cause the API to reject the payload with a 422 Unprocessable Entity
    status because the minimum length requirement is not satisfied.
    """
    series_id = "test_series_fail"
    values = [1.0, 1.2]

    payload = {
        "timestamps": make_timestamps(2),
        "values": values  # Too few values, should trigger validation error
    }
    
    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422


def test_fit_endpoint_rejects_invalid_values():
    """@brief Verify the endpoint rejects invalid numeric entries like None or Infinity.

    @details Each malformed payload is expected to return 422 with an explicit error message
    describing why the value (None or infinite) cannot be converted for training.
    """
    series_id = "test_series_invalid"

    # Test with None
    payload_none = {
        "timestamps": make_timestamps(4),
        "values": [1.0, 2.0, None, 4.0]
    }

    response_none = client.post(f"/fit/{series_id}", json=payload_none)

    assert response_none.status_code == 422
    msg_none = response_none.json()["detail"][0]["msg"].lower()
    assert any(token in msg_none for token in ("none", "number", "float"))

    # Test with Infinity (JSON string representation)
    payload_inf = {
        "timestamps": make_timestamps(4),
        "values": [1.0, 2.0, "Infinity", 4.0]
    }

    response_inf = client.post(f"/fit/{series_id}", json=payload_inf)

    assert response_inf.status_code == 422
    assert "infinite" in response_inf.json()["detail"][0]["msg"].lower()


def test_fit_endpoint_rejects_constant_values():
    """@brief Assert the fit endpoint rejects constant sequences with no variance.

    @details A constant data series lacks the variability needed for anomaly detection, so the API
    should respond with 422 and mention the constant-valued issue in the validation message.
    """
    series_id = "test_series_constant"

    payload = {
        "timestamps": make_timestamps(4),
        "values": [5.0, 5.0, 5.0, 5.0]
    }

    response = client.post(f"/fit/{series_id}", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert "constant" in detail.lower()


def test_fit_endpoint_rejects_non_numeric_values():
    """@brief Ensure payloads containing non-numeric tokens are rejected.
    
    @details Including a string among numbers should cause the validation to fail with a 422 error,
    preventing training on invalid range data.
    """
    series_id = "test_series_non_numeric"
    
    payload = {
        "timestamps": make_timestamps(4),
        "values": [1.0, 2.0, "invalid", 4.0]
    }

    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422
    msg = response.json()["detail"][0]["msg"].lower()
    assert any(token in msg for token in ("number", "float"))


def test_fit_endpoint_rejects_blank_series_id():
    """@brief Ensure blank/whitespace series_id is rejected at API validation.

    @details Path values that decode to whitespace (e.g. `%20%20`) must fail
    before training service execution.
    """
    payload = {
        "timestamps": make_timestamps(4),
        "values": [1.0, 2.0, 3.0, 4.0],
    }

    with patch("app.api.train.TrainService.train") as train_mock:
        response = client.post("/fit/%20%20", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("series_id must be a non-empty string." in error["msg"] for error in errors)
    train_mock.assert_not_called()
