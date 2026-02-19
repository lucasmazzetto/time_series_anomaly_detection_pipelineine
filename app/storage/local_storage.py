from __future__ import annotations

import json
from pathlib import Path

from app.storage.storage import Storage
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries
from app.utils.env import get_model_state_folder, get_training_data_folder


class LocalStorage(Storage):
    def save_state(self, series_id: str, version: int, state: ModelState) -> str:
        """@brief Save model state locally as a JSON file.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param state Serialized model state payload.
        @return Filesystem path where the state was stored.
        """
        folder = Path(get_model_state_folder())
        series_folder = folder / series_id
        series_folder.mkdir(parents=True, exist_ok=True)
        file_path = series_folder / f"{series_id}_model_v{version}.json"

        with file_path.open("w", encoding="utf-8") as file_obj:
            json.dump(state.model_dump(mode="json"), file_obj)

        return str(file_path)

    def save_data(self, series_id: str, version: int, payload: TimeSeries) -> str:
        """@brief Save training data locally as a JSON file.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param payload Training data payload.
        @return Filesystem path where the data was stored.
        """
        folder = Path(get_training_data_folder())
        series_folder = folder / series_id
        series_folder.mkdir(parents=True, exist_ok=True)
        file_path = series_folder / f"{series_id}_data_v{version}.json"

        with file_path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload.model_dump(mode="json"), file_obj)

        return str(file_path)

    def load_state(self, model_path: str) -> ModelState:
        """@brief Load model state from a JSON file.

        @param model_path Filesystem path to the persisted model state.
        @return Deserialized model state payload.
        """
        file_path = Path(model_path)
        with file_path.open("r", encoding="utf-8") as file_obj:
            raw_state = json.load(file_obj)

        return ModelState.model_validate(raw_state)

    def load_data(self, data_path: str) -> TimeSeries:
        """@brief Load training data from a JSON file.

        @param data_path Filesystem path to the persisted training data.
        @return Deserialized training data payload.
        @throws FileNotFoundError If the target file does not exist.
        @throws ValidationError If file contents do not match `TimeSeries`.
        """
        file_path = Path(data_path)
        with file_path.open("r", encoding="utf-8") as file_obj:
            raw_data = json.load(file_obj)

        return TimeSeries.model_validate(raw_data)
