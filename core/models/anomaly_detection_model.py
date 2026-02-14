from __future__ import annotations

import numpy as np
from time_series_anomaly_detection.schemas.time_series import DataPoint, TimeSeries


class AnomalyDetectionModel:
    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        values_stream = [d.value for d in data.data]
        self.mean = np.mean(values_stream)
        self.std = np.std(values_stream)
        return self

    def predict(self, data_point: DataPoint) -> bool:
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise ValueError("Model must be trained before prediction.")
        return data_point.value > self.mean + 3 * self.std
