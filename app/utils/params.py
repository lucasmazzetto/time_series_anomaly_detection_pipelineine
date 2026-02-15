import os
from typing import Any


def load_params() -> dict[str, Any]:
    return {
        "min_training_data_points": os.getenv("MIN_TRAINING_DATA_POINTS", "3"),
    }
