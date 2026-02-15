from abc import ABC, abstractmethod
import sys
from typing import Any, Dict
from schemas.univariate_time_series import TimeSeries


class TrainingObserver(ABC):
    @abstractmethod
    def update(
        self, series_id: str, data: TimeSeries, metrics: Dict[str, Any]
    ) -> None:
        pass


class TrainingMonitor(TrainingObserver):
    def update(self, series_id: str, data: TimeSeries, metrics: Dict[str, Any]) -> None:
        # TODO: store metrics in a database for later analysis.
        sys.stdout.write(f"{series_id} {metrics}\n")
