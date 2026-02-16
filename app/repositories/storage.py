from abc import ABC, abstractmethod
from app.schemas import ModelState, TimeSeries


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
