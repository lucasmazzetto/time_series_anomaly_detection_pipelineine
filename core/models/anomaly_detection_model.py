from __future__ import annotations

import numpy as np
from schemas.model import DataPoint, TrainData


class AnomalyDetectionModel:
    def fit(self, data: TrainData) -> "AnomalyDetectionModel":
        values_stream = data.values
        self.mean = np.mean(values_stream)
        self.std = np.std(values_stream)
        return self

    def predict(self, data_point: DataPoint) -> bool:
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise ValueError("Model must be trained before prediction.")
        
        return data_point.value > self.mean + 3 * self.std
