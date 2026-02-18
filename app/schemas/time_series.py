from typing import Sequence

from pydantic import BaseModel, Field, model_validator

from app.schemas.data_point import DataPoint
from app.utils.env import get_min_training_data_points


class TimeSeries(BaseModel):
    """@brief Ordered collection of `DataPoint` samples for a single series.

    @details Used by training request conversion (`app/schemas/train_data.py`),
    model/trainer contracts (`app/core/model.py`, `app/core/trainer.py`),
    storage (`app/storage/storage.py`), and training service flows
    (`app/services/train_service.py`).

    @note Validation rules:
    base validation enforces non-empty, minimum 2 points and strictly
    increasing timestamps; `validate_for_training()` additionally enforces
    configured minimum sample size and rejects constant-only values.
    """

    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @model_validator(mode="after")
    def validate_series_shape(self) -> "TimeSeries":
        """@brief Validate generic time-series structure constraints."""
        if len(self.data) < 2:
            raise ValueError("TimeSeries must contain at least 2 data points.")

        timestamps = [point.timestamp for point in self.data]
        if any(curr <= prev for prev, curr in zip(timestamps, timestamps[1:])):
            raise ValueError("TimeSeries timestamps must be strictly increasing.")

        return self

    def validate_for_training(self) -> "TimeSeries":
        """@brief Apply training preflight validation rules."""
        min_points = get_min_training_data_points()

        if len(self.data) < min_points:
            raise ValueError(
                f"Input list must contain at least {min_points} data points."
            )

        values = [point.value for point in self.data]
        first_value = values[0]
        if all(value == first_value for value in values):
            raise ValueError("Input list cannot contain constant values only.")

        return self
