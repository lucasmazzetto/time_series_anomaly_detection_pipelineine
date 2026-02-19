from typing import Any, Callable, Optional

from app.core.model import Model
from app.core.trainer import Trainer
from app.schemas.model_state import ModelState
from app.schemas.time_series import TimeSeries


class AnomalyDetectionTrainer(Trainer):

    def __init__(self, model: Model, callback: Optional[Callable[[Any], None]] = None):
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
