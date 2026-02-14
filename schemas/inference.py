from typing import List, Optional, Sequence
from math import isfinite

from pydantic import BaseModel, Field, field_validator, model_validator
from utils.params import load_params


class PredictData(BaseModel):
    """@brief A single recorded metric at a specific instant in time.
    
    @var timestamp (int): Unix time when the measurement was taken.
    @var value (float): Observed magnitude for the measurement.
    """

    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected."
    )
    
    value: float = Field(..., description="Observed value at `timestamp`.")