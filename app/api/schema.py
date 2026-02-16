from math import isfinite
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils.params import load_params
from app.core.schema import TimeSeries, DataPoint


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

        @description Ensures matching lengths, minimum data points, and non-constant values.

        @return The validated TrainData instance.
        """
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length.")

        params = load_params()
        min_points = int(params.get("min_training_data_points"))

        if len(self.timestamps) < min_points:
            raise ValueError(
                f"Input list must contain at least {min_points} data points."
            )

        first_value = self.values[0] if self.values else None
        if self.values and all(value == first_value for value in self.values):
            raise ValueError("Input list cannot contain constant values only.")

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
        )


class TrainResponse(BaseModel):
    """@brief Structured response outputed after a training attempt.

    @var series_id: Identifier of the series that was targeted.
    @var message: Summary of the training outcome.
    @var success: Flag telling if training succeeded.
    """
    series_id: str
    message: str
    success: bool


class PredictData(BaseModel):
    timestamp: str = Field(
        ...,
        description="Timestamp value in unix timestamp string format",
    )
    value: float = Field(...)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, timestamp: str) -> str:
        """@brief Validate timestamp input for type and value constraints.

        @param timestamp Unix timestamp string to validate.
        @return Validated timestamp string.
        """
        if isinstance(timestamp, bool) or not isinstance(timestamp, str):
            raise ValueError("Timestamp must be provided as a string.")
        if not timestamp.strip():
            raise ValueError("Timestamp must be a non-empty string.")
        if not timestamp.isdigit():
            raise ValueError("Timestamp must contain only digits.")
        return timestamp

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        """@brief Validate value input for type and finite constraints.

        @param value Numeric value to validate.
        @return Validated numeric value.
        """
        if value is None:
            raise ValueError("Value cannot be None, NaN, or infinite.")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Value must be a float or int.")
        if not isfinite(value):
            raise ValueError("Value cannot be None, NaN, or infinite.")
        return float(value)

    def to_data_point(self) -> DataPoint:
        """@brief Convert input payload into a DataPoint model.

        @return DataPoint instance with parsed timestamp and value.
        """
        return DataPoint(timestamp=int(self.timestamp), value=float(self.value))


class PredictVersion(BaseModel):
    version: str = Field(
        default="0",
        description="Model version in formats like 1, v1, or V1.",
    )

    @field_validator("version")
    @classmethod
    def sanitize_version(cls, version: str) -> str:
        """@brief Sanitize version input keeping only numeric characters.

        @param version Raw version value received from query string.
        @return Sanitized numeric version string.
        """
        if isinstance(version, bool):
            raise ValueError("Version must be a string or integer-like value.")

        # Keep only ASCII digits to neutralize querystring noise/injection payloads.
        value = str(version).strip()
        digits_only = re.sub(r"[^0-9]", "", value)
        if not digits_only:
            raise ValueError("Version must contain at least one digit.")

        return digits_only

    def to_int(self) -> int:
        """@brief Convert sanitized version string into integer.

        @return Integer model version.
        """
        return int(self.version)
