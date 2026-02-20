from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from app.schemas.data_point import DataPoint
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries


class Model(ABC):
    @abstractmethod
    def fit(self, data: TimeSeries,
            callback: Optional[Callable[[Any], None]] = None) -> None:
        """@brief Fit the model on training data.

        @param data Training dataset.
        @param callback Optional callable invoked during training with partial model updates.
        @return None.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_point: DataPoint) -> bool:
        """@brief Predict on a single data point.

        @param data_point Input data point.
        @return Prediction result.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> ModelState:
        """@brief Return serializable model state (params + metrics).

        @return ModelState instance with model parameters and metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, state: ModelState) -> None:
        """@brief Restore model from serialized state.
        
        @param state Dictionary with model parameters and metrics.
        @return None.
        """
        raise NotImplementedError
