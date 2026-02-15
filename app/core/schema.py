from math import isfinite
from typing import Sequence, Any

from pydantic import BaseModel, Field, field_validator, model_validator
from app.utils.params import load_params


class DataPoint(BaseModel):
    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    value: float | None = Field(
        ..., description="Value of the time series measured at time `timestamp`"
    )


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity",
    )


class ModelState(BaseModel):
    model: str = Field(..., description="Model identifier.")
    parameters: dict[str, Any] = Field(
        ..., description="Serializable model parameters."
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Optional training metrics."
    )
