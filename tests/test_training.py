from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_fit_endpoint():
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
    series_id = "test_series_fail"
    payload = {
        "values": [1.0, 1.2]  # Too few values, should trigger validation error
    }
    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422


def test_fit_endpoint_rejects_invalid_values():
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
    series_id = "test_series_constant"
    payload = {"values": [5.0, 5.0, 5.0, 5.0]}
    response = client.post(f"/fit/{series_id}", json=payload)
    assert response.status_code == 422
    assert "constant" in response.json()["detail"][0]["msg"]


def test_fit_endpoint_rejects_non_numeric_values():
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
