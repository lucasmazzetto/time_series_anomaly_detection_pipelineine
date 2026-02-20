import json
from typing import cast

from pydantic import ValidationError


def value_error_details(exc: ValueError) -> list[dict[str, object]]:
    """@brief Build a Pydantic-style 422 detail payload from a ValueError.

    @param exc Domain ValueError raised during payload handling.
    @return List-formatted validation details compatible with FastAPI/Pydantic errors.
    """
    return [
        {
            "type": "value_error",
            "loc": ["body"],
            "msg": str(exc),
            "input": None,
        }
    ]


def validation_error_details(exc: ValidationError) -> list[dict[str, object]]:
    """@brief Build a JSON-safe 422 detail payload from a Pydantic ValidationError.

    @param exc Pydantic validation error raised during payload conversion.
    @return List-formatted validation details safe to include in HTTPException detail.
    """
    return cast(list[dict[str, object]], json.loads(exc.json()))
