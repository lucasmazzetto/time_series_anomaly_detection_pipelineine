from abc import ABC, abstractmethod
import sys
from typing import Any, Dict
from schemas.train import TrainData


class TrainingObserver(ABC):
    @abstractmethod
    def update(self, series_id: str, data: TrainData, metrics: Dict[str, Any]) -> None:
        pass


class TrainingMonitor(TrainingObserver):
    def update(self, series_id: str, data: TrainData, metrics: Dict[str, Any]) -> None:
        # TODO: store metrics in a database for later analysis.

        print(f"{series_id} {metrics}\n")
