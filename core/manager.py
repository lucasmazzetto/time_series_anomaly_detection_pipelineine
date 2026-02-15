from utils.params import load_params
from schemas.univariate_time_series import TimeSeries
from core.trainer import Trainer
from core.monitor import TrainingMonitor


class TrainingManager:
    def __init__(self, series_id: str):
        """
        @brief Initialize a training manager for the given series and load async settings.

        @param series_id Unique identifier for the time series whose training is being managed.
        """
        self.series_id = series_id
        params = load_params()
        self.assync_training = params.get("assync_training")

    def start_training(self, data: TimeSeries):
        """
        @brief Starts training for the provided series data.

        @param data The time series observations used to train the model.
        """
        if self.assync_training:
            # TODO: implement it using a queue
            raise NotImplementedError("Async training is not implemented.")
        else:
            trainer = Trainer(self.series_id, data)
            trainer.add_observer(TrainingMonitor())
            trainer.train()
