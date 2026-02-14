from typing import Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected."
    )
    value: float = Field(..., description="Observed value at `timestamp`.")


class TimeSeries(BaseModel):
    data: Sequence[DataPoint] = Field(
        ..., description="Ordered list of subsequent measurements."
    )
