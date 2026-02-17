import re
from typing import Annotated

from pydantic import BeforeValidator


def _validate_series_id(series_id: object) -> str:
    """@brief Validate and normalize series identifier input."""
    if isinstance(series_id, bool):
        raise ValueError("series_id must be a string.")

    value = str(series_id).strip()
    if not value:
        raise ValueError("series_id must be a non-empty string.")

    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("series_id must contain only letters, numbers, '.', '_' or '-'.")

    if ".." in value:
        raise ValueError("series_id cannot contain consecutive dots.")

    return value


SeriesId = Annotated[str, BeforeValidator(_validate_series_id)]
