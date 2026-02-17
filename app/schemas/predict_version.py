import re

from pydantic import BaseModel, Field, field_validator


class PredictVersion(BaseModel):
    
    version: str = Field(
        default="0",
        description="Model version in formats like 1, v1, or V1.",
    )

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
