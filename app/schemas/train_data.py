from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.data_point import DataPoint
from app.schemas.time_series import TimeSeries


class TrainData(BaseModel):

    timestamps: list[int] = Field(
        ...,
        description="Timestamp values should be in the unix timestamp format",
    )
    values: list[float] = Field(...)

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(cls, timestamps: list[int]) -> list[int]:
        """@brief Validate timestamps input for type and value constraints.

        @param timestamps List of Unix timestamps to validate.
        @return Validated timestamps list.
        """
        for timestamp in timestamps:
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise ValueError("Input list must contain only integer Unix timestamps.")
            if timestamp < 0:
                raise ValueError(
                    "Input list must contain only non-negative Unix timestamps."
                )
        return timestamps

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[float]) -> list[float]:
        """@brief Validate values input for type and finite constraints.

        @param values List of numeric values to validate.
        @return Validated values list.
        """
        validated: list[float] = []
        for value in values:
            if value is None:
                raise ValueError(
                    "Input list cannot contain None, NaN, or infinite values."
                )
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("Input list must contain only float or int values.")
            if not isfinite(value):
                raise ValueError(
                    "Input list cannot contain None, NaN, or infinite values."
                )
            validated.append(float(value))
        return validated

    @model_validator(mode="after")
    def validate_lengths(self) -> "TrainData":
        """@brief Validate length and content rules across fields.

        @description Ensures matching lengths across timestamps and values.

        @return The validated TrainData instance.
        """
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length.")

        return self

    def to_time_series(self) -> TimeSeries:
        """@brief Convert input lists into a TimeSeries model.

        @return TimeSeries instance containing the data points.
        """
        return TimeSeries(
            data=[
                DataPoint(timestamp=timestamp, value=value)
                for timestamp, value in zip(self.timestamps, self.values)
            ]
        ).validate_for_training()
