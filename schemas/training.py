from typing import List, Optional
from math import isfinite

from pydantic import BaseModel, field_validator
from utils.params import load_params

class TrainData(BaseModel):
    """@brief Represents the training payload consumed by the trainer.

    @var values: Sequence of raw metric values supplied as the training set.
    """

    values: List[Optional[float]]

    @field_validator('values')
    @classmethod
    def validate_values(cls, v: List[Optional[float]]) -> List[float]:
        """@brief Rejects invalid training payloads before model updates.

        @param v: Candidate list of floats or ints given from the client.
        @return: The same list after verifying the required invariants.
        @raise ValueError: When a validation rule is violated.
        """

        params = load_params()
        min_points = int(params.get("min_training_data_points"))

        if len(v) < min_points:
            raise ValueError(f"Input list must contain at least {min_points} data points.")

        # Reject if any value is not float or int
        if any(x is not None and not isinstance(x, (float, int)) for x in v):
            raise ValueError("Input list must contain only float or int values.")
        
        # Reject if any value is None, NaN, or infinity
        if any(x is None or not isfinite(x) for x in v):
            raise ValueError("Input list cannot contain None, NaN, or infinite values.")
        
        # Reject if all values are constant
        if len(set(v)) == 1:
            raise ValueError("Input list cannot contain constant values only.")
        
        return v


class TrainResponse(BaseModel):
    """@brief Structured response outputed after a training attempt.

    @var series_id: Identifier of the series that was targeted.
    @var message: Summary of the training outcome.
    @var success: Flag telling if training succeeded.
    """

    series_id: str
    message: str
    success: bool
