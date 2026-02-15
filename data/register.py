from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data.database import Database, database_from_params
from data.repository import Storage, FileStorage
from utils.params import load_params


class Register:

    def __init__(self, 
                 storage: Storage | None = None,
                 database: Database | None = None,
                 artifacts_path: str | Path | None = None) -> None:
        """@brief Initialize the register with storage/database backends.

        @param storage Storage backend for model artifacts; defaults to FileStorage
            rooted at artifacts_path or params.yaml's model_artifacts_path.
        @param database Database backend for training metadata.
        @param artifacts_path Optional override for the artifact root directory.
        """
        params = load_params() if storage is None or database is None else {}

        if storage is None:
            root = artifacts_path or params.get("model_artifacts_path")
            if not root:
                raise ValueError("model_artifacts_path must be set in params.yaml.")
            storage = FileStorage(root)

        if database is None:
            database = database_from_params(params)

        self.storage = storage
        self.database = database
        self._training_versions: Dict[str, int] = {}

    def _require_database(self) -> Database:
        if self.database is None:
            raise ValueError("Database backend must be provided.")
        
        return self.database

    def add_training(self, series_id: str) -> None:
        """@brief Create a training entry in the database.

        @param series_id Identifier of the time series.
        """
        database = self._require_database()
        version = database.add_training(series_id)
        self._training_versions[series_id] = version

    def update_metrics(self, series_id: str, metrics: Dict[str, Any]) -> None:
        """@brief Store model metrics in storage and database.

        @param series_id Identifier of the time series.
        @param metrics Metrics to store.
        """
        self.save_model(series_id, metrics)

    def save_model(self, series_id: str, metrics: Dict[str, Any]) -> None:
        """@brief Persist model artifacts.

        @param series_id Identifier of the time series.
        @param metrics Metrics to serialize as model artifacts.
        """
        database = self._require_database()
        version = self._training_versions.get(series_id)
        if version is None:
            version = database.add_training(series_id)
            self._training_versions[series_id] = version
        model_path = self.storage.save_metrics(series_id, metrics, version)
        database.save_model_metrics(series_id, version, str(model_path), metrics)

    def save_training_data(self, series_id: str, data: Any) -> None:
        """@brief Persist training data.

        @param series_id Identifier of the time series.
        @param data Training data to persist in storage.
        """
        database = self._require_database()
        version = self._training_versions.get(series_id)
        if version is None:
            version = database.add_training(series_id)
            self._training_versions[series_id] = version
        data_path = self.storage.save_training_data(series_id, data, version)
        database.save_training_data(series_id, version, str(data_path))

    def load_model(self, series_id: str) -> Dict[str, Any] | None:
        """@brief Load model artifacts from storage.

        @param series_id Identifier of the time series.
        @return Serialized metrics used to reconstruct the model, if any.
        """
        return self.storage.load_metrics(series_id)
