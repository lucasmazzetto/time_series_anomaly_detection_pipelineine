from typing import List
from schemas.train import TrainData
from core.monitor import TrainingObserver
from core.models.anomaly_detection_model import AnomalyDetectionModel


class Trainer:
    def __init__(self, series_id: str, data: TrainData):
        """@brief Initialize a trainer for a single time series.

        @param series_id Identifier of the time series to train on.
        @param data Training dataset for the series.
        """
        self.series_id = series_id
        self.data = data
        self.model = AnomalyDetectionModel()
        self._observers: List[TrainingObserver] = []

    def add_observer(self, observer: TrainingObserver) -> None:
        """@brief Register an observer to receive training updates.

        @param observer Observer instance to notify after training completes.
        """
        self._observers.append(observer)

    def _notify(self, metrics: dict) -> None:
        """@brief Notify all observers with training metrics.

        @param metrics Training metrics produced by the model.
        """
        for observer in self._observers:
            observer.update(self.series_id, self.data, metrics)

    def train(self):
        """@brief Train the model and notify observers with saved metrics."""
        print(f"Training model for series {self.series_id}...")
        self.model.fit(self.data)
        metrics = self.model.save()
        self._notify(metrics)
