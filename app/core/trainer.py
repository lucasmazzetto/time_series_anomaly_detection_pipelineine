from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from app.core.model import Model
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries


class Trainer(ABC):

    def __init__(self, model: Model, callback: Optional[Callable[[Any], None]] = None):
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
