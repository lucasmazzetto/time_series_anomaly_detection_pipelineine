from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np
from app.core.schema import TimeSeries, DataPoint, ModelState


class BaseModel(ABC):
    @abstractmethod
    def fit(self, data, callback: Optional[Callable[[Any], None]] = None):
        """@brief Fit the model on training data.

        @param data Training dataset.
        @param callback Optional callable invoked during training with partial model updates.
        @return None.
        """
        pass

    @abstractmethod
    def predict(self, data_point):
        """@brief Predict on a single data point.

        @param data_point Input data point.
        @return Prediction result.
        """
        pass
    
    @abstractmethod
    def save(self) -> ModelState:
        """@brief Return serializable model state (params + metrics).

        @return ModelState instance with model parameters and metrics.
        """
        pass

    @abstractmethod
    def load(self, state: ModelState) -> None:
        """@brief Restore model from serialized state.
        
        @param state Dictionary with model parameters and metrics.
        @return None.
        """
        pass


class SimpleModel(BaseModel):
    def fit(self, data: TimeSeries, callback: Optional[Callable[[Any], None]] = None) -> None:
        """@brief Fit the model on training data.

        @param data Training data containing the values stream.
        @param callback Optional callable invoked with the saved model state.
        @return None.
        """
        values_stream = np.fromiter(
            (point.value for point in data.data),
            dtype=float,
            count=len(data.data),
        )

        self.mean = float(np.mean(values_stream))
        self.std = float(np.std(values_stream))

        state = self.save()

        if callback:
            callback(state)

    def predict(self, data_point: DataPoint) -> bool:
        """@brief Predict whether a data point is an anomaly.

        @param data_point The data point to evaluate.
        @return True if the point is an anomaly, otherwise False.
        @throws ValueError If the model has not been trained.
        """
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise ValueError("Model must be trained before prediction.")
        
        return data_point.value > self.mean + 3 * self.std

    def save(self) -> ModelState:
        """@brief Serialize the model state.

        @return ModelState instance with model type and parameters.
        """
        return ModelState(
            model="anomaly_detection_model",
            parameters={
                "mean": self.mean,
                "std": self.std,
            },
        )

    def load(self, state: ModelState) -> None:
        """@brief Load a model instance from serialized state.

        @param state Serialized model state.
        @return None.
        """
        self.mean = state.parameters["mean"]
        self.std = state.parameters["std"]
