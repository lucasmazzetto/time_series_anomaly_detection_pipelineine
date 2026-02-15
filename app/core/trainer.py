from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from app.core.schema import TimeSeries, ModelState
from app.core.model import BaseModel


class Trainer(ABC):

    def __init__(self, model: BaseModel, callback: Optional[Callable[[Any], None]] = None):
        """@brief Initialize the trainer with a model and optional callback.

        @param model Model class used to instantiate the trainer's model.
        @param callback Optional callback invoked during training for progress updates or metrics.
        """
        self.model = model
        self.callback = callback

    @abstractmethod
    def train(self, data: TimeSeries) -> ModelState:
        """@brief Train the model on provided data.

        @param data Training dataset for the series.
        @return Persisted model state after training.
        """
        raise NotImplementedError


class AnomalyDetectionTrainer(Trainer):

    def __init__(self, model: BaseModel, callback: Optional[Callable[[Any], None]] = None):
        """@brief Initialize a trainer for a single time series.

        @param model Model class or factory used to instantiate the trainer's model.
        @param callback Optional callback invoked during training.
        """
        super().__init__(model, callback)

    def train(self, data: TimeSeries) -> ModelState:
        """@brief Train the model and notify observers with saved metrics.

        @param data Training dataset for the series.
        @return Persisted model state after training.
        """
        self.data = data
        self.model.fit(data, self.callback)

        return self.model.save()
