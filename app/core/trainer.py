from typing import Any, Callable, Optional
from app.core.schema import TimeSeries, ModelState
from app.core.model import BaseModel


class AnomalyDetectionTrainer:
    def __init__(self, model: BaseModel, callback: Optional[Callable[[Any], None]] = None):
        """@brief Initialize a trainer for a single time series.

        @param series_id Identifier of the time series to train on.
        @param data Training dataset for the series.
        """
        self.model = model()
        self.callback = callback

    def train(self, data: TimeSeries) -> ModelState:
        """@brief Train the model and notify observers with saved metrics."""
        
        self.data = data
        self.model.fit(data, self.callback)

        return self.model.save()
