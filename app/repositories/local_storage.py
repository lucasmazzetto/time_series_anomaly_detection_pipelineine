import json
import os
import pickle
from pathlib import Path
from typing import Any

from app.core.schema import TimeSeries, ModelState
from app.utils.params import load_params as get_params


class LocalStorage:
    @staticmethod
    def _resolve_folder(
        params: dict[str, Any],
        param_keys: tuple[str, ...],
        env_keys: tuple[str, ...],
        fallback: str,
    ) -> Path:
        for key in param_keys:
            value = params.get(key)
            if value:
                return Path(str(value))

        for key in env_keys:
            value = os.getenv(key)
            if value:
                return Path(value)

        return Path(fallback)

    def save_state(self, series_id: str, version: int, state: ModelState) -> str:
        params = get_params()
        folder = self._resolve_folder(
            params=params,
            param_keys=("model_state_folder", "model_folder"),
            env_keys=("MODEL_STATE_FOLDER", "MODEL_FOLDER"),
            fallback="./data/models",
        )
        series_folder = folder / series_id
        series_folder.mkdir(parents=True, exist_ok=True)
        file_path = series_folder / f"{series_id}_model_v{version}.pkl"

        with file_path.open("wb") as file_obj:
            pickle.dump(state.model_dump(mode="json"), file_obj, protocol=pickle.HIGHEST_PROTOCOL)

        return str(file_path)

    def save_data(self, series_id: str, version: int, payload: TimeSeries) -> str:
        params = get_params()
        folder = self._resolve_folder(
            params=params,
            param_keys=("training_data_folder", "data_folder"),
            env_keys=("TRAINING_DATA_FOLDER", "DATA_FOLDER"),
            fallback="./data/data",
        )
        series_folder = folder / series_id
        series_folder.mkdir(parents=True, exist_ok=True)
        file_path = series_folder / f"{series_id}_data_v{version}.json"

        with file_path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload.model_dump(mode="json"), file_obj)

        return str(file_path)
