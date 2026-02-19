import re

from pydantic import BaseModel, Field, field_validator, model_validator


class Version(BaseModel):

    version: str = Field(
        default="0",
        description="Model version in formats like 1, v1, or V1.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_scalar_input(cls, data: object) -> object:
        """@brief Accept scalar query values and coerce into model shape.

        @details FastAPI may pass a raw query scalar (e.g. `version=0`) for
        this model-typed parameter. This hook normalizes scalar input into
        `{\"version\": <value>}` while preserving dict/object inputs.

        @param data Raw value received from query parsing.
        @return Normalized payload suitable for model validation.
        """
        if isinstance(data, (str, int)) and not isinstance(data, bool):
            return {"version": str(data)}
        return data

    @field_validator("version")
    @classmethod
    def sanitize_version(cls, version: str) -> str:
        """@brief Validate and normalize version input.

        @param version Raw version value received from query string.
        @return Normalized numeric version string.
        """
        if isinstance(version, bool):
            raise ValueError("Version must be a string or integer-like value.")

        value = str(version).strip()
        if not re.fullmatch(r"[vV]?\d+", value):
            raise ValueError("Version must contain at least one digit.")

        if value.startswith(("v", "V")):
            return value[1:]

        return value

    def to_int(self) -> int:
        """@brief Convert sanitized version string into integer.

        @return Integer model version.
        """
        return int(self.version)
