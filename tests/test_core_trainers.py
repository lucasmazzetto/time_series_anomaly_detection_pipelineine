from unittest.mock import MagicMock

import pytest

from app.core.simple_model import SimpleModel
from app.schemas import DataPoint, ModelState, TimeSeries
from app.core.trainer import AnomalyDetectionTrainer


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=2.0),
            DataPoint(timestamp=1_700_000_002, value=3.0),
        ]
    )


def test_anomaly_detection_trainer_returns_state_and_invokes_callback():
    """@brief Validate trainer returns state and invokes callback.

    @details Ensures training yields a ModelState and calls the provided callback.
    """
    model = SimpleModel()
    callback = MagicMock()
    trainer = AnomalyDetectionTrainer(model=model, callback=callback)

    data = _sample_series()
    state = trainer.train(data)

    assert isinstance(state, ModelState)
    assert state.parameters["mean"] == pytest.approx(2.0)
    assert callback.call_count == 1
    callback_state = callback.call_args[0][0]
    assert isinstance(callback_state, ModelState)
