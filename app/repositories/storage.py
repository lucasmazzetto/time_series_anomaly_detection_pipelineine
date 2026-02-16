from __future__ import annotations

from abc import ABC, abstractmethod

import json
import os
import pickle
from pathlib import Path
from typing import Any

from app.utils.params import load_params as get_params
from app.core.schema import ModelState, TimeSeries


class Storage(ABC):
    @abstractmethod
    def save_state(self, series_id: str, version: int, state: ModelState) -> str:
        """@brief Persist a serialized model state.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param state Serialized model state payload.
        @return Filesystem path where the state was stored.
        """
        raise NotImplementedError

    @abstractmethod
    def save_data(self, series_id: str, version: int, payload: TimeSeries) -> str:
        """@brief Persist the training data used for the model.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param payload Training data payload.
        @return Filesystem path where the data was stored.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, model_path: str) -> ModelState:
        """@brief Load a serialized model state from disk.

        @param model_path Filesystem path to the persisted model state.
        @return Deserialized model state payload.
        """
        raise NotImplementedError


class LocalStorage(Storage):
    @staticmethod
    def _resolve_folder(
        params: dict[str, Any],
        param_keys: tuple[str, ...],
        env_keys: tuple[str, ...],
        fallback: str,
    ) -> Path:
        """@brief Resolve a storage folder from params, env vars, or fallback.

        @param params Parameters dictionary to check first.
        @param param_keys Keys to search in params.
        @param env_keys Environment variables to search next.
        @param fallback Default folder if nothing else is set.
        @return Resolved folder path.
        """
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
        """@brief Save model state locally as a pickle file.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param state Serialized model state payload.
        @return Filesystem path where the state was stored.
        """
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
        """@brief Save training data locally as a JSON file.

        @param series_id Identifier for the time series.
        @param version Model version to persist.
        @param payload Training data payload.
        @return Filesystem path where the data was stored.
        """
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

    def load_state(self, model_path: str) -> ModelState:
        """@brief Load model state from a pickle file.

        @param model_path Filesystem path to the persisted model state.
        @return Deserialized model state payload.
        """
        file_path = Path(model_path)
        with file_path.open("rb") as file_obj:
            raw_state = pickle.load(file_obj)

        return ModelState.model_validate(raw_state)
