from math import isfinite
from typing import Sequence

from pydantic import BaseModel, Field, field_validator, model_validator
from utils.params import load_params


class DataPoint(BaseModel):
    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    value: float | None = Field(
        ..., description="Value of the time series measured at time `timestamp`"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: int) -> int:
        """@brief Validate that the timestamp is a non-negative Unix integer.

        @param v Input timestamp value to validate.
        @return The validated timestamp.
        @throws ValueError If the value is not an int or is negative.
        """
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("Input list must contain only integer Unix timestamps.")
        if v < 0:
            raise ValueError("Input list must contain only non-negative Unix timestamps.")
        return v

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float | None) -> float:
        """@brief Validate that the value is a finite numeric scalar.

        @param v Input value to validate.
        @return The validated value coerced to float.
        @throws ValueError If the value is None, not numeric, NaN, or infinite.
        """
        if v is None:
            raise ValueError("Input list cannot contain None, NaN, or infinite values.")
        if isinstance(v, bool) or not isinstance(v, (float, int)):
            raise ValueError("Input list must contain only float or int values.")
        if not isfinite(v):
            raise ValueError("Input list cannot contain None, NaN, or infinite values.")
        return float(v)


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_timestamps_values(cls, data: object) -> object:
        """@brief Coerce separate `timestamps`/`values` arrays into `data` objects.

        @param data Raw input to the model validator.
        @return Original input or a dict containing a `data` list of points.
        @throws ValueError If `timestamps` and `values` lengths differ.
        """
        if not isinstance(data, dict) or "data" in data:
            return data

        if "timestamps" in data or "values" in data:
            timestamps = data.get("timestamps")
            values = data.get("values")

            if timestamps is None or values is None:
                return data

            if len(timestamps) != len(values):
                raise ValueError("timestamps and values must have the same length.")

            return {
                "data": [
                    {"timestamp": timestamp, "value": value}
                    for timestamp, value in zip(timestamps, values)
                ]
            }

        return data

    @model_validator(mode="after")
    def validate_data(self) -> "TimeSeries":
        """@brief Validate dataset length and non-constant values.

        @return The validated model instance.
        @throws ValueError If there are too few points or all values are constant.
        """
        params = load_params()
        min_points = int(params.get("min_training_data_points"))

        if len(self.data) < min_points:
            raise ValueError(
                f"Input list must contain at least {min_points} data points."
            )

        if self.data:
            first_value = self.data[0].value
            if all(point.value == first_value for point in self.data):
                raise ValueError("Input list cannot contain constant values only.")

        return self
