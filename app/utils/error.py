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
