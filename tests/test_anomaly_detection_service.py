from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.schemas.data_point import DataPoint
from app.schemas.model_state import ModelState
from app.schemas.predict_response import PredictResponse
from app.schemas.time_series import TimeSeries
from app.schemas.train_response import TrainResponse
from app.services.predict import PredictService
from app.services.train import TrainService


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=1.1),
            DataPoint(timestamp=1_700_000_002, value=0.9),
        ]
    )


def _sample_point() -> DataPoint:
    return DataPoint(timestamp=1_700_000_999, value=2.5)


def test_train_success_saves_state_and_data():
    """@brief Validate training success persists state and data.

    @details Ensures the service calls the trainer, stores model state/data,
    updates the record, and avoids rollback on success.
    """
    session = MagicMock()
    trainer = MagicMock()
    storage = MagicMock()
    series_id = "series_ok"
    payload = _sample_series()

    state = ModelState(model="mock", parameters={"foo": "bar"})
    trainer.train.return_value = state
    storage.save_state.return_value = "/tmp/model.pkl"
    storage.save_data.return_value = "/tmp/data.json"

    service = TrainService(
        session=session, trainer=trainer, storage=storage
    )

    with patch(
        "app.services.train.AnomalyDetectionRecord.build"
    ) as build_mock, patch(
        "app.services.train.AnomalyDetectionRecord.save",
        return_value=7,
    ) as save_mock:
        model = MagicMock()
        build_mock.return_value = model

        result = service.train(series_id, payload)

    assert result == TrainResponse(
        series_id=series_id,
        version="7",
        points_used=3,
    )
    trainer.train.assert_called_once_with(payload)
    build_mock.assert_called_once_with(
        series_id=series_id, version=None, model_path=None, data_path=None
    )
    save_mock.assert_called_once_with(session, model)
    storage.save_state.assert_called_once_with(series_id, 7, state)
    storage.save_data.assert_called_once_with(series_id, 7, payload)
    model.update.assert_called_once_with(
        model_path="/tmp/model.pkl", data_path="/tmp/data.json"
    )
    model.commit.assert_called_once_with()
    session.rollback.assert_not_called()


def test_train_failure_rolls_back_and_raises_500():
    """@brief Validate training failures are mapped to server errors.

    @details Ensures exceptions short-circuit persistence, trigger rollback,
    and return a stable HTTP 500 error contract.
    """
    session = MagicMock()
    trainer = MagicMock()
    storage = MagicMock()
    series_id = "series_fail"
    payload = _sample_series()

    trainer.train.side_effect = RuntimeError("boom")

    service = TrainService(
        session=session, trainer=trainer, storage=storage
    )

    with patch(
        "app.services.train.AnomalyDetectionRecord.build"
    ) as build_mock, patch(
        "app.services.train.AnomalyDetectionRecord.save"
    ) as save_mock:
        with pytest.raises(HTTPException) as exc:
            service.train(series_id, payload)

    assert exc.value.status_code == 500
    assert "unexpected error while training model" in exc.value.detail.lower()
    trainer.train.assert_called_once_with(payload)
    build_mock.assert_not_called()
    save_mock.assert_not_called()
    storage.save_state.assert_not_called()
    storage.save_data.assert_not_called()
    session.rollback.assert_called_once()


def test_predict_latest_version_returns_predict_response():
    """@brief Validate prediction uses latest model when version is zero.

    @details Ensures the service resolves latest metadata, loads persisted
    state, runs prediction, and returns the expected response schema.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()
    state = ModelState(model="mock", parameters={"mean": 1.0, "std": 0.2})

    storage.load_state.return_value = state
    model.predict.return_value = True

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_last_model",
        return_value={"version": 6, "model_path": "/tmp/model_v6.pkl"},
    ) as get_last_mock, patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version"
    ) as get_version_mock:
        response = service.predict("series_predict", 0, payload)

    assert isinstance(response, PredictResponse)
    assert response.anomaly is True
    assert response.model_version == "6"
    get_last_mock.assert_called_once_with(session, "series_predict")
    get_version_mock.assert_not_called()
    storage.load_state.assert_called_once_with("/tmp/model_v6.pkl")
    model.load.assert_called_once_with(state)
    model.predict.assert_called_once_with(payload)


def test_predict_specific_version_uses_requested_model():
    """@brief Validate prediction uses requested non-zero model version.

    @details Ensures the service resolves metadata by explicit version and
    returns a prediction response with that version.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()
    state = ModelState(model="mock", parameters={"mean": 3.0, "std": 0.4})

    storage.load_state.return_value = state
    model.predict.return_value = False

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version",
        return_value={"version": 2, "model_path": "/tmp/model_v2.pkl"},
    ) as get_version_mock:
        response = service.predict("series_predict", 2, payload)

    assert response == PredictResponse(anomaly=False, model_version="2")
    get_version_mock.assert_called_once_with(session, "series_predict", 2)
    storage.load_state.assert_called_once_with("/tmp/model_v2.pkl")


def test_predict_raises_400_for_invalid_inputs():
    """@brief Validate input checks reject invalid identifiers and versions.

    @details Ensures blank series ids and negative versions are reported as
    400 Bad Request before any downstream work.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with pytest.raises(HTTPException) as blank_exc:
        service.predict("   ", 0, payload)
    assert blank_exc.value.status_code == 400

    with pytest.raises(HTTPException) as negative_exc:
        service.predict("series_predict", -1, payload)
    assert negative_exc.value.status_code == 400


def test_predict_raises_404_when_model_metadata_not_found():
    """@brief Validate missing database metadata maps to 404.

    @details Ensures ValueError from record lookup is translated into an
    HTTP 404 Not Found response.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version",
        side_effect=ValueError("Model version '3' not found for series_id 'x'."),
    ):
        with pytest.raises(HTTPException) as exc:
            service.predict("x", 3, payload)

    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail.lower()


def test_predict_raises_500_when_model_path_missing():
    """@brief Validate missing model_path in metadata maps to 500.

    @details Ensures incomplete database metadata fails fast with server
    error to avoid invalid storage access.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version",
        return_value={"version": 4, "model_path": None},
    ):
        with pytest.raises(HTTPException) as exc:
            service.predict("series_predict", 4, payload)

    assert exc.value.status_code == 500
    assert "model path is missing" in exc.value.detail.lower()


def test_predict_raises_404_when_model_artifact_missing():
    """@brief Validate missing artifact file maps to 404.

    @details Ensures file-system lookup failures return a not-found response
    instead of a generic server error.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()

    storage.load_state.side_effect = FileNotFoundError("missing")

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version",
        return_value={"version": 9, "model_path": "/tmp/missing.pkl"},
    ):
        with pytest.raises(HTTPException) as exc:
            service.predict("series_predict", 9, payload)

    assert exc.value.status_code == 404
    assert "artifact was not found" in exc.value.detail.lower()


def test_predict_raises_500_for_unexpected_runtime_errors():
    """@brief Validate unexpected prediction failures map to 500.

    @details Ensures generic runtime exceptions are wrapped into a stable
    internal server error contract.
    """
    session = MagicMock()
    model = MagicMock()
    storage = MagicMock()
    payload = _sample_point()
    state = ModelState(model="mock", parameters={"mean": 1.0, "std": 0.1})

    storage.load_state.return_value = state
    model.load.side_effect = RuntimeError("unexpected")

    service = PredictService(
        session=session, model=model, storage=storage
    )

    with patch(
        "app.services.predict.AnomalyDetectionRecord.get_model_version",
        return_value={"version": 8, "model_path": "/tmp/model_v8.pkl"},
    ):
        with pytest.raises(HTTPException) as exc:
            service.predict("series_predict", 8, payload)

    assert exc.value.status_code == 500
    assert "unexpected error" in exc.value.detail.lower()
