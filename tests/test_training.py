from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_fit_endpoint():
    """@brief Submit a valid time series to the fit endpoint.

    @details Expects a 200 response with the returned series_id matching the request and a success flag
    alongside a descriptive message.
    """
    series_id = "test_series_001"
    payload = {
        "values": [1.0, 1.2, 1.1, 0.9, 1.0, 1.3]
    }

    response = client.post(f"/fit/{series_id}", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["series_id"] == series_id
    assert data["success"] is True
    assert "message" in data


def test_fit_endpoint_validation_failure():
    """@brief Provide an undersized series to the fit endpoint to trigger validation.

    @details Sending too few points should cause the API to reject the payload with a 422 Unprocessable Entity
    status because the minimum length requirement is not satisfied.
    """
    series_id = "test_series_fail"
    payload = {
        "values": [1.0, 1.2]  # Too few values, should trigger validation error
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
    payload_none = {"values": [1.0, 2.0, None, 4.0]}
    response_none = client.post(f"/fit/{series_id}", json=payload_none)
    assert response_none.status_code == 422
    assert "None" in response_none.json()["detail"][0]["msg"]

    # Test with Infinity (JSON string representation)
    payload_inf = {"values": [1.0, 2.0, "Infinity", 4.0]}
    response_inf = client.post(f"/fit/{series_id}", json=payload_inf)
    assert response_inf.status_code == 422
    assert "infinite" in response_inf.json()["detail"][0]["msg"]


def test_fit_endpoint_rejects_constant_values():
    """@brief Assert the fit endpoint rejects constant sequences with no variance.

    @details A constant data series lacks the variability needed for anomaly detection, so the API
    should respond with 422 and mention the constant-valued issue in the validation message.
    """
    series_id = "test_series_constant"
    payload = {"values": [5.0, 5.0, 5.0, 5.0]}
    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422
    assert "constant" in response.json()["detail"][0]["msg"]


def test_fit_endpoint_rejects_non_numeric_values():
    """@brief Ensure payloads containing non-numeric tokens are rejected.
    
    @details Including a string among numbers should cause the validation to fail with a 422 error,
    preventing training on invalid range data.
    """
    series_id = "test_series_non_numeric"
    payload = {"values": [1.0, 2.0, "invalid", 4.0]}
    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422


if __name__ == "__main__":
    test_fit_endpoint()
    test_fit_endpoint_validation_failure()
    test_fit_endpoint_rejects_invalid_values()
    test_fit_endpoint_rejects_constant_values()
    test_fit_endpoint_rejects_non_numeric_values()
