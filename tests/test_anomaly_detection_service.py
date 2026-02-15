from unittest.mock import MagicMock, patch

from app.core.schema import DataPoint, ModelState, TimeSeries
from app.services.anomaly_detection_service import AnomalyDetectionTrainingService


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=1.1),
            DataPoint(timestamp=1_700_000_002, value=0.9),
        ]
    )


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

    service = AnomalyDetectionTrainingService(
        session=session, trainer=trainer, storage=storage
    )

    with patch(
        "app.services.anomaly_detection_service.AnomalyDetectionRecord.build"
    ) as build_mock, patch(
        "app.services.anomaly_detection_service.AnomalyDetectionRecord.save",
        return_value=7,
    ) as save_mock:
        model = MagicMock()
        build_mock.return_value = model

        result = service.train(series_id, payload)

    assert result is True
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
    session.rollback.assert_not_called()


def test_train_failure_rolls_back_and_returns_false():
    """@brief Validate training failures are handled safely.

    @details Ensures exceptions short-circuit persistence and trigger a rollback,
    returning False to the caller.
    """
    session = MagicMock()
    trainer = MagicMock()
    storage = MagicMock()
    series_id = "series_fail"
    payload = _sample_series()

    trainer.train.side_effect = RuntimeError("boom")

    service = AnomalyDetectionTrainingService(
        session=session, trainer=trainer, storage=storage
    )

    with patch(
        "app.services.anomaly_detection_service.AnomalyDetectionRecord.build"
    ) as build_mock, patch(
        "app.services.anomaly_detection_service.AnomalyDetectionRecord.save"
    ) as save_mock:
        result = service.train(series_id, payload)

    assert result is False
    trainer.train.assert_called_once_with(payload)
    build_mock.assert_not_called()
    save_mock.assert_not_called()
    storage.save_state.assert_not_called()
    storage.save_data.assert_not_called()
    session.rollback.assert_called_once()
