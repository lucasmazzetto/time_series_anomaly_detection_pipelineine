import os
from typing import Any


def load_params() -> dict[str, Any]:
    """@brief Load application parameters from environment variables.

    @return Dictionary of configuration parameters.
    """
    return {
        "min_training_data_points": os.getenv("MIN_TRAINING_DATA_POINTS", "3"),
    }
