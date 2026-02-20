from typing import Sequence

from pydantic import BaseModel, Field, model_validator

from app.schemas.data_point import DataPoint
from app.utils.env import get_min_training_data_points


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @model_validator(mode="after")
    def validate_series_shape(self) -> "TimeSeries":
        """@brief Validate generic time-series structure constraints.

        @return Validated TimeSeries instance.
        """
        min_points = get_min_training_data_points()
        if len(self.data) < min_points:
            raise ValueError(
                f"TimeSeries must contain at least {min_points} data points."
            )

        timestamps = [point.timestamp for point in self.data]
        if any(curr <= prev for prev, curr in zip(timestamps, timestamps[1:])):
            raise ValueError("TimeSeries timestamps must be strictly increasing.")

        return self

    def validate_for_training(self) -> "TimeSeries":
        """@brief Apply training preflight validation rules.

        @return Validated TimeSeries instance ready for training.
        """
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
