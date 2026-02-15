import json
import pickle

from app.core.schema import DataPoint, ModelState, TimeSeries
from app.repositories.storage import LocalStorage


def _sample_series() -> TimeSeries:
    return TimeSeries(
        data=[
            DataPoint(timestamp=1_700_000_000, value=1.0),
            DataPoint(timestamp=1_700_000_001, value=1.1),
        ]
    )


def test_local_storage_saves_state_to_disk(tmp_path, monkeypatch):
    """@brief Verify model state is persisted to disk as a pickle.

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

    saved_path = model_dir / series_id / f"{series_id}_model_v{version}.pkl"
    assert file_path == str(saved_path)
    assert saved_path.exists()

    with saved_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

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
