import pytest

from app.core.model import SimpleModel
from app.schemas import DataPoint, ModelState, TimeSeries


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=2.0),
            DataPoint(timestamp=1_700_000_002, value=3.0),
        ]
    )


def test_simple_model_fit_save_and_predict():
    """@brief Validate fit/save/predict flow for SimpleModel.

    @details Ensures training computes expected stats and prediction
    flags anomalous points.
    """
    model = SimpleModel()
    data = _sample_series()

    model.fit(data)
    state = model.save()

    assert isinstance(state, ModelState)
    assert state.parameters["mean"] == pytest.approx(2.0)
    assert state.parameters["std"] == pytest.approx(0.81649658, rel=1e-6)

    assert model.predict(DataPoint(timestamp=1, value=5.0)) is True
    assert model.predict(DataPoint(timestamp=1, value=2.1)) is False


def test_simple_model_predict_requires_training():
    """@brief Validate predict requires prior training.

    @details Ensures calling predict on an untrained model raises.
    """
    model = SimpleModel()

    with pytest.raises(ValueError, match="trained"):
        model.predict(DataPoint(timestamp=1, value=1.0))


def test_simple_model_load_restores_state():
    """@brief Validate load restores model parameters.

    @details Ensures loaded stats drive prediction results as expected.
    """
    model = SimpleModel()
    state = ModelState(model="anomaly_detection_model", parameters={"mean": 10.0, "std": 2.0})

    model.load(state)

    assert model.predict(DataPoint(timestamp=1, value=17.0)) is True
    assert model.predict(DataPoint(timestamp=1, value=12.0)) is False
