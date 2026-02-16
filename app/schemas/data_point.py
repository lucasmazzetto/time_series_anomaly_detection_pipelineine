from datetime import datetime
from math import isfinite

from pydantic import BaseModel, Field, field_validator


class DataPoint(BaseModel):
    """@brief Canonical timestamped sample used across training and prediction.

    @details Used by `TimeSeries` (`app/schemas/time_series.py`), core model
    and trainer layers (`app/core/model.py`, `app/core/trainer.py`), services
    (`app/services/anomaly_detection_service.py`) and API adapters
    (`app/schemas/predict_data.py`, `app/schemas/train_data.py`).

    @note Validation rules:
    `timestamp` must be a non-negative integer Unix timestamp (0 is valid),
    and `value` must be a finite numeric value.
    """

    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    value: float = Field(
        ..., description="Value of the time series measured at time `timestamp`"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, timestamp: int) -> int:
        """@brief Validate timestamp type and Unix timestamp boundaries."""
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("timestamp must be an integer Unix timestamp.")
        if timestamp < 0:
            raise ValueError("timestamp must be greater than or equal to 0.")
        try:
            datetime.utcfromtimestamp(timestamp)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError("timestamp is not a valid Unix timestamp.") from exc
        return timestamp

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        """@brief Validate value as finite numeric measurement."""
        if value is None:
            raise ValueError("value cannot be None, NaN, or infinite.")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("value must be a float or int.")
        if not isfinite(value):
            raise ValueError("value cannot be None, NaN, or infinite.")
        return float(value)
