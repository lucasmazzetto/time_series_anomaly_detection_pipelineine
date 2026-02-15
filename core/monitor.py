from abc import ABC, abstractmethod
from typing import Any, Dict
from schemas.univariate_time_series import TimeSeries


class TrainingObserver(ABC):
    @abstractmethod
    def update(
        self, series_id: str, data: TimeSeries, metrics: Dict[str, Any]
    ) -> None:
        pass


class TrainingMonitor(TrainingObserver):
    def __init__(self, register: Any) -> None:
        self.register = register

    def update(self, series_id: str, data: TimeSeries, metrics: Dict[str, Any]) -> None:
        self.register.save_model(series_id, metrics)
        self.register.save_training_data(series_id, data)
