import numpy as np
from typing import Dict, Any
from schemas.univariate_time_series import TimeSeries, DataPoint
from core.models.base_model import BaseModel


class AnomalyDetectionModel(BaseModel):

    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        """@brief Fit the model on training data.

        @param data Training data containing the values stream.
        @return The fitted model instance.
        """
        values_stream = np.fromiter(
            (point.value for point in data.data),
            dtype=float,
            count=len(data.data),
        )
        self.mean = float(np.mean(values_stream))
        self.std = float(np.std(values_stream))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        """@brief Predict whether a data point is an anomaly.

        @param data_point The data point to evaluate.
        @return True if the point is an anomaly, otherwise False.
        @throws ValueError If the model has not been trained.
        """
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise ValueError("Model must be trained before prediction.")
        return data_point.value > self.mean + 3 * self.std

    def save(self) -> Dict[str, Any]:
        """@brief Serialize the model state.

        @return Dictionary with model type and parameters.
        """
        return {
            "model": "anomaly_detection_model",
            "parameters": {
                "mean": self.mean,
                "std": self.std,
            }
        }

    @classmethod
    def load(cls, state: Dict[str, Any]) -> "AnomalyDetectionModel":
        """@brief Load a model instance from serialized state.

        @param state Serialized model state.
        @return A reconstructed model instance.
        """
        model = cls()
        model.mean = state["parameters"]["mean"]
        model.std = state["parameters"]["std"]
        return model
