import json

import pytest
from pydantic import ValidationError

from app.storage.local_storage import LocalStorage
from app.schemas.data_point import DataPoint
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=1.1),
        ]
    )


def test_local_storage_saves_state_to_disk(tmp_path, monkeypatch):
    """@brief Verify model state is persisted to disk as JSON.

    @details Ensures the storage path respects environment overrides,
    creates the expected file, and serializes the model state payload.
    """
    series_id = "series_a"
    version = 3
    state = ModelState(model="mock", parameters={"alpha": 0.1})

    model_dir = tmp_path / "models"
    monkeypatch.setenv("MODEL_STATE_FOLDER", str(model_dir))

    storage = LocalStorage()
    file_path = storage.save_state(series_id, version, state)

    saved_path = model_dir / series_id / f"{series_id}_model_v{version}.json"
    assert file_path == str(saved_path)
    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    assert payload == state.model_dump(mode="json")


def test_local_storage_saves_data_to_disk(tmp_path, monkeypatch):
    """@brief Verify training data is persisted to disk as JSON.

    @details Ensures the storage path respects environment overrides,
    creates the expected file, and serializes the time-series payload.
    """
    series_id = "series_b"
    version = 2
    payload = _sample_series()

    data_dir = tmp_path / "data"
    monkeypatch.setenv("TRAINING_DATA_FOLDER", str(data_dir))

    storage = LocalStorage()
    file_path = storage.save_data(series_id, version, payload)

    saved_path = data_dir / series_id / f"{series_id}_data_v{version}.json"
    assert file_path == str(saved_path)
    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file_obj:
        saved = json.load(file_obj)

    assert saved == payload.model_dump(mode="json")


def test_local_storage_loads_state_from_disk(tmp_path):
    """@brief Verify load_state restores a serialized model state payload.

    @details Ensures the method reads JSON content and validates it as
    `ModelState` before returning.
    """
    state = ModelState(
        model="anomaly_detection_model",
        parameters={"mean": 2.0, "std": 0.5},
        metrics={"samples": 10},
    )
    saved_path = tmp_path / "model_state.json"

    with saved_path.open("w", encoding="utf-8") as file_obj:
        json.dump(state.model_dump(mode="json"), file_obj)

    storage = LocalStorage()
    loaded = storage.load_state(str(saved_path))

    assert isinstance(loaded, ModelState)
    assert loaded == state


def test_local_storage_load_state_rejects_invalid_payload(tmp_path):
    """@brief Verify load_state validates payload shape and types.

    @details Ensures malformed JSON content fails with Pydantic validation
    instead of returning a partially invalid state object.
    """
    saved_path = tmp_path / "invalid_model_state.json"
    invalid_payload = {"model": "anomaly_detection_model"}  # Missing parameters

    with saved_path.open("w", encoding="utf-8") as file_obj:
        json.dump(invalid_payload, file_obj)

    storage = LocalStorage()

    with pytest.raises(ValidationError):
        storage.load_state(str(saved_path))
