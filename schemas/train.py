from pydantic import BaseModel

from typing import List, Optional, Sequence
from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator
from utils.params import load_params


class TrainData(BaseModel):
    """@brief Represents the training payload consumed by the trainer.

    @var timestamps: Sequence of Unix timestamps (seconds) for each reading.
    @var values: Sequence of raw metric values supplied as the training set.
    """

    timestamps: List[int] = Field(
        ..., description="Unix timestamp (seconds) for each measurement."
    )
    values: List[Optional[float]] = Field(
        ..., description="Observed values aligned with the timestamps list."
    )

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(cls, v: List[int]) -> List[int]:
        """@brief Rejects invalid timestamps before model updates."""

        if any(isinstance(x, bool) or not isinstance(x, int) for x in v):
            raise ValueError("Input list must contain only integer Unix timestamps.")

        if any(x < 0 for x in v):
            raise ValueError("Input list must contain only non-negative Unix timestamps.")

        return v

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: List[Optional[float]]) -> List[float]:
        """@brief Rejects invalid training payloads before model updates."""

        # Reject if any value is not float or int
        if any(x is not None and not isinstance(x, (float, int)) for x in v):
            raise ValueError("Input list must contain only float or int values.")

        # Reject if any value is None, NaN, or infinity
        if any(x is None or not isfinite(x) for x in v):
            raise ValueError("Input list cannot contain None, NaN, or infinite values.")

        return v

    @model_validator(mode="after")
    def validate_lengths_and_variance(self) -> "TrainData":
        """@brief Validates list sizes and minimum variation constraints."""

        params = load_params()
        min_points = int(params.get("min_training_data_points"))

        if len(self.values) < min_points:
            raise ValueError(
                f"Input list must contain at least {min_points} data points."
            )

        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length.")

        # Reject if all values are constant
        if len(set(self.values)) == 1:
            raise ValueError("Input list cannot contain constant values only.")

        return self


class TrainResponse(BaseModel):
    """@brief Structured response outputed after a training attempt.

    @var series_id: Identifier of the series that was targeted.
    @var message: Summary of the training outcome.
    @var success: Flag telling if training succeeded.
    """

    series_id: str
    message: str
    success: bool
