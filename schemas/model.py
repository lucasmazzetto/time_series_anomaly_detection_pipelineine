from typing import Sequence

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """@brief A single recorded metric at a specific instant in time.
    
    @var timestamp (int): Unix time when the measurement was taken.
    @var value (float): Observed magnitude for the measurement.
    """

    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected."
    )
    value: float = Field(..., description="Observed value at `timestamp`.")


class TimeSeries(BaseModel):
    """@brief Ordered collection of readings representing a full time series.

    @var data (Sequence[DataPoint]): Chronological measurements that describe the
    series to be inspected.
    """

    data: Sequence[DataPoint] = Field(
        ..., description="Ordered list of subsequent measurements."
    )

